from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from .backend import BackendError
from .data import canonical, file_digest
from .prompt import validate_response

# Included in the cache identity. Changes to CLI context/tools must invalidate it.
CODEX_POLICY = {
    "version": 1,
    "sandbox": "read-only",
    "ignore_user_config": True,
    "ephemeral": True,
    "disabled_features": ["shell_tool", "apps", "plugins", "hooks", "multi_agent",
                          "memories", "code_mode", "browser_use", "computer_use",
                          "image_generation", "view_image", "tool_suggest"],
    "overrides": {"approval_policy": "never", "web_search": "disabled",
                  "project_doc_max_bytes": 0, "skills.include_instructions": False,
                  "skills.bundled.enabled": False},
}
MIN_VERSION = (0, 165, 0)
MAX_OUTPUT_BYTES = 8 * 1024 * 1024


class CodexBackend:
    """Fresh isolated codex exec session per text, authenticated by the CLI.

    No shell interpolation, repository context, session resumption, or SDK usage.
    Nonzero exits/timeouts are never retried automatically: the CLI may already
    have consumed subscription limits before the caller observes the failure.
    """

    kind = "codex"

    def __init__(self, *, executable="codex", timeout=300, max_requests=2000,
                 min_interval=0.2, sleep=time.sleep):
        if (not math.isfinite(timeout) or timeout <= 0 or type(max_requests) is not int
                or max_requests < 0 or not math.isfinite(min_interval) or min_interval < 0):
            raise ValueError("Limites de execução Codex inválidos.")
        self.executable = executable
        self.timeout, self.max_requests, self.min_interval = timeout, max_requests, min_interval
        self.sleep = sleep
        self.attempts, self.last_call = 0, 0.0
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
        self._metadata = None

    def metadata(self):
        if self._metadata is not None:
            return self._metadata
        executable = shutil.which(self.executable)
        if not executable:
            raise BackendError("Codex CLI não encontrado. Instale-o e execute codex login; ou informe --codex-bin.")
        self.executable = str(Path(executable).resolve())
        try:
            result = subprocess.run([self.executable, "--version"], capture_output=True, text=True,
                                    encoding="utf-8", timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BackendError(f"Não foi possível consultar a versão do Codex ({type(exc).__name__}).") from None
        match = re.fullmatch(r"codex-cli (\d+)\.(\d+)\.(\d+)(?:[-+][\w.-]+)?", result.stdout.strip())
        if result.returncode or not match or tuple(map(int, match.groups())) < MIN_VERSION:
            raise BackendError("É necessário Codex CLI >=0.165.0 para os controles de isolamento utilizados.")
        self._metadata = {"kind": self.kind, "endpoint": "codex.exec", "cli_version": result.stdout.strip(),
                          "executable": self.executable, "executable_sha256": file_digest(Path(self.executable)),
                          "adapter_sha256": file_digest(Path(__file__)),
                          "policy": CODEX_POLICY}
        return self._metadata

    def command(self, request: dict, directory: Path) -> list[str]:
        args = [self.executable, "exec", "--ignore-user-config", "--strict-config", "--ephemeral",
                "--skip-git-repo-check", "--sandbox", "read-only", "--color", "never", "--json",
                "--model", request["model"], "--cd", str(directory),
                "--output-schema", str(directory / "schema.json"),
                "--output-last-message", str(directory / "answer.json")]
        for feature in CODEX_POLICY["disabled_features"]:
            args.extend(["--disable", feature])
        overrides = {**CODEX_POLICY["overrides"], "model_instructions_file": str(directory / "instructions.txt")}
        if request["reasoning_effort"] is not None:
            overrides["model_reasoning_effort"] = request["reasoning_effort"]
        for key, value in overrides.items():
            # JSON literals here are also TOML literals (only scalar values).
            args.extend(["--config", key + "=" + json.dumps(value, ensure_ascii=False)])
        args.append("-")
        return args

    def _execute(self, args, prompt: str, directory: Path):
        # Files instead of PIPE prevent unbounded memory from CLI diagnostics.
        with (directory / "events.jsonl").open("wb") as out, (directory / "stderr.log").open("wb") as err:
            process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                       cwd=directory, start_new_session=True)
            try:
                process.communicate(input=prompt.encode("utf-8"), timeout=self.timeout)
            except BaseException:
                # Terminate the process group too, so an interrupted run leaves no classifier behind.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
                raise
        if process.returncode:
            raise BackendError(f"Codex encerrou com código {process.returncode}. Confira codex login status, modelo e versão; cache preservado.")
        for name in ("events.jsonl", "stderr.log", "answer.json"):
            path = directory / name
            if not path.is_file() or path.stat().st_size > MAX_OUTPUT_BYTES:
                raise BackendError("Saída Codex ausente ou acima do limite; nenhum rótulo atribuído.")

    def create(self, request: dict) -> dict:
        metadata = self.metadata()
        if request.get("transport") != "codex" or request.get("runtime") != metadata:
            raise BackendError("Request não corresponde ao backend/runtime Codex configurado.")
        if self.attempts >= self.max_requests:
            raise BackendError("Limite de novas execuções Codex atingido. Retome com o mesmo cache.")
        self.sleep(max(0, self.min_interval - (time.monotonic() - self.last_call)))
        # The temporary working directory is outside the project: no train/gold files.
        with tempfile.TemporaryDirectory(prefix="ep2-codex-") as name:
            directory = Path(name)
            (directory / "schema.json").write_text(canonical(request["schema"]), encoding="utf-8")
            instructions = request["instructions"] + "\nClassifique exclusivamente os dados do prompt. Não use ferramentas, arquivos, skills ou rede. Não há tarefa de programação.\n"
            (directory / "instructions.txt").write_text(instructions, encoding="utf-8")
            args = self.command(request, directory)
            self.attempts += 1
            self.last_call = time.monotonic()
            try:
                self._execute(args, request["prompt"], directory)
            except subprocess.TimeoutExpired:
                raise BackendError("Timeout do Codex; processo encerrado e cache preservado. Aumente --timeout para retomar.") from None
            except OSError as exc:
                raise BackendError(f"Falha ao iniciar Codex ({type(exc).__name__}); cache preservado.") from None
            try:
                events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            except (ValueError, UnicodeError):
                raise BackendError("Codex produziu eventos JSONL inválidos.") from None
            if not all(isinstance(event, dict) and isinstance(event.get("type"), str) for event in events):
                raise BackendError("Codex produziu envelopes de eventos inválidos.")
            completed = [e for e in events if e.get("type") == "turn.completed"]
            # Tool use could read external context; refuse any such run.
            for event in events:
                if event.get("type") in {"turn.failed", "error"}:
                    raise BackendError("Codex reportou falha; nenhum rótulo atribuído.")
                if event["type"].startswith("item."):
                    item = event.get("item")
                    if not isinstance(item, dict) or item.get("type") not in {"reasoning", "agent_message"}:
                        raise BackendError("Codex tentou usar ferramenta/contexto externo; classificação rejeitada.")
            if len(completed) != 1:
                raise BackendError("Codex não confirmou exatamente um turno concluído.")
            usage = completed[0].get("usage")
            if not isinstance(usage, dict) or any(type(usage.get(key)) is not int or usage[key] < 0 for key in self.usage):
                raise BackendError("Codex não forneceu contagem válida de tokens.")
            for key in self.usage:
                self.usage[key] += usage[key]
            thread = next((e.get("thread_id") for e in events if e.get("type") == "thread.started"), None)
            answer = (directory / "answer.json").read_text(encoding="utf-8")
            result = {"id": f"codex:{thread}" if thread else None, "model": request["model"],
                      "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": answer}]}],
                      "usage": {"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
                                "input_tokens_details": {"cached_tokens": usage["cached_input_tokens"]}},
                      "codex": {"runtime": metadata, "model_is_requested_alias": True}}
            validate_response(request, result)
            return result
