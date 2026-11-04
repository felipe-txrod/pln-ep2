from __future__ import annotations

import os
import random
import sqlite3
import time
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from .data import canonical, digest
from .prompt import parse_response

API_BASE = "https://api.openai.com/v1"


@contextmanager
def file_lock(path: Path):
    """Linux/macOS advisory lock. Refuse concurrent runs instead of double charging."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Outro processo está usando {path}.") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, request TEXT NOT NULL, response TEXT NOT NULL)")
        self.connection.commit()

    @staticmethod
    def key(request: dict) -> str:
        endpoint = "codex.exec" if request.get("transport") == "codex" else API_BASE
        return digest(canonical({"endpoint": endpoint, "request": request}))

    def get(self, request: dict):
        import json
        row = self.connection.execute("SELECT request, response FROM responses WHERE key=?", (self.key(request),)).fetchone()
        if row is None:
            return None
        if row[0] != canonical(request):
            raise ValueError("Cache inconsistente: payload divergente.")
        response = json.loads(row[1])
        parse_response(response)
        return response

    def put(self, request: dict, response: dict):
        parse_response(response)
        self.connection.execute("INSERT OR IGNORE INTO responses VALUES (?, ?, ?)",
                                (self.key(request), canonical(request), canonical(response)))
        self.connection.commit()  # Each successful prediction survives an interrupted run.

    def close(self):
        self.connection.close()


class BackendError(RuntimeError):
    pass


class OpenAIBackend:
    kind = "api"

    def metadata(self):
        return {"kind": self.kind, "endpoint": API_BASE}

    def __init__(self, *, timeout: float = 60, retries: int = 2, max_requests: int = 2000,
                 min_interval: float = 0.2, client=None, sleep=time.sleep):
        if (not math.isfinite(timeout) or timeout <= 0 or type(retries) is not int or retries < 0
                or type(max_requests) is not int or max_requests < 0
                or not math.isfinite(min_interval) or min_interval < 0):
            raise ValueError("Limites de execução inválidos.")
        self.timeout, self.retries = timeout, retries
        self.max_requests, self.min_interval = max_requests, min_interval
        self.client, self.sleep = client, sleep
        self.attempts, self.last_call = 0, 0.0
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}

    def _client(self):
        if self.client is None:
            if not os.environ.get("OPENAI_API_KEY", "").strip():
                raise BackendError("Configure OPENAI_API_KEY no terminal para chamadas reais; dry-run funciona sem chave.")
            from openai import OpenAI
            # Own bounded retry loop: never stack SDK retries or inherit another endpoint.
            self.client = OpenAI(base_url=API_BASE, timeout=self.timeout, max_retries=0)
        return self.client

    def create(self, request: dict) -> dict:
        try:
            from openai import APIConnectionError, APIStatusError
        except ImportError:
            raise BackendError("O backend API requer o SDK: instale com pip install -e '.[api]'.") from None
        client = self._client()
        for attempt in range(self.retries + 1):
            if self.attempts >= self.max_requests:
                raise BackendError("Limite de novas tentativas atingido. Retome com o mesmo cache.")
            self.sleep(max(0, self.min_interval - (time.monotonic() - self.last_call)))
            self.attempts += 1
            self.last_call = time.monotonic()
            try:
                response = client.responses.create(**request).model_dump(mode="json")
                usage = response.get("usage") or {}
                self.usage["input_tokens"] += usage.get("input_tokens", 0)
                self.usage["output_tokens"] += usage.get("output_tokens", 0)
                self.usage["cached_input_tokens"] += (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
                return response
            except (APIConnectionError, APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                code = getattr(exc, "code", None)
                retryable = (status is None or status in {408, 409, 429} or status >= 500)
                if code in {"insufficient_quota", "billing_hard_limit_reached"}:
                    retryable = False
                if not retryable or attempt == self.retries:
                    # No exception body: it can contain input or credentials.
                    raise BackendError(f"Falha OpenAI ({type(exc).__name__}, HTTP {status}); cache preservado.") from None
                delay = min(30.0, 2 ** attempt + random.random())
                headers = getattr(getattr(exc, "response", None), "headers", {})
                retry_after = headers.get("retry-after")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        try:
                            retry_at = parsedate_to_datetime(retry_after)
                            delay = max(delay, (retry_at - datetime.now(timezone.utc)).total_seconds())
                        except (TypeError, ValueError):
                            pass
                if delay > 60:
                    raise BackendError("Retry-After excede 60s; aguarde o intervalo do servidor e retome.") from None
                self.sleep(delay)
        raise AssertionError("Unreachable")


def make_backend(config, *, codex_bin="codex", timeout=None, retries=None, max_requests=2000, min_interval=0.2):
    if config.backend == "codex":
        if retries is not None:
            raise ValueError("--retries é exclusivo da API. O backend Codex não reinicia processos automaticamente.")
        from .codex_backend import CodexBackend
        return CodexBackend(executable=codex_bin, timeout=300 if timeout is None else timeout,
                            max_requests=max_requests, min_interval=min_interval)
    return OpenAIBackend(timeout=60 if timeout is None else timeout, retries=2 if retries is None else retries,
                         max_requests=max_requests, min_interval=min_interval)
