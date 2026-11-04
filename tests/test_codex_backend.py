from __future__ import annotations

import builtins
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from gpt_classifier.backend import BackendError, Cache, make_backend
from gpt_classifier.cli import main
from gpt_classifier.codex_backend import CodexBackend
from gpt_classifier.data import Text, read_xlsx
from gpt_classifier.pipeline import run
from gpt_classifier.prompt import Config, build_request


@pytest.fixture
def codex_config():
    return Config(backend="codex", model="gpt-6.1-sol", max_output_tokens=None,
                  reasoning_effort="low", strategy="zero", per_class=0)


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    path = tmp_path / "fake-codex"
    # Actual executable subprocess: no network or real Codex invocation in tests.
    path.write_text(f"#!{sys.executable}\n" + '''
import json,os,sys,time
from pathlib import Path
if sys.argv[1:] == ["--version"]:
    print(os.environ.get("FAKE_CODEX_VERSION", "codex-cli 0.165.0"))
    raise SystemExit(0)
args=sys.argv[1:]
mode=os.environ.get("FAKE_CODEX_MODE", "ok")
assert args[-1]=="-"
prompt=sys.stdin.read()
root=Path.cwd()
out=Path(args[args.index("--output-last-message")+1])
schema=Path(args[args.index("--output-schema")+1])
assert json.loads(schema.read_text())["additionalProperties"] is False
with open(os.environ["FAKE_CODEX_LOG"],"a") as stream:
    stream.write(json.dumps({"args":args,"cwd":str(root),"prompt":prompt,"files":sorted(p.name for p in root.iterdir())})+"\\n")
if mode=="timeout":
    time.sleep(30)
if mode=="nonzero":
    print("secret diagnostic",file=sys.stderr)
    raise SystemExit(1)
answer='{"label":"c5"}'
if mode=="bad_answer": answer='{"label":"c6"}'
if mode=="duplicate_key": answer='{"label":"c1","label":"c5"}'
if mode!="missing_answer": out.write_text(answer)
print(json.dumps({"type":"thread.started","thread_id":"synthetic-thread"}))
if mode=="bad_event":
    print("null")
if mode=="bad_jsonl":
    print("not-json")
if mode=="tool":
    print(json.dumps({"type":"item.started","item":{"type":"command_execution","command":"cat gold.xlsx"}}))
if mode=="failed":
    print(json.dumps({"type":"turn.failed","error":{"message":"secret error"}}))
if mode!="not_completed":
    event={"type":"turn.completed","usage":{"input_tokens":40,"output_tokens":5,"cached_input_tokens":10}}
    if mode=="bad_usage": event["usage"]={"input_tokens":-1}
    print(json.dumps(event))
''', encoding="utf-8")
    path.chmod(0o755)
    monkeypatch.setenv("FAKE_CODEX_LOG", str(tmp_path / "calls.jsonl"))
    return path


def prepared_request(backend, config):
    request, _ = build_request(Text(2, 'Texto com $(touch /tmp/NAO_EXECUTAR), `comando`, aspas " e acentuação.'), [], config)
    request["runtime"] = backend.metadata()
    return request


def test_codex_uses_stdin_isolated_cwd_and_schema_without_key_or_sdk(fake_codex, codex_config, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    original_import = builtins.__import__
    def guard(name, *args, **kwargs):
        if name == "openai" or name.startswith("openai."):
            raise AssertionError("Codex must not import the API SDK")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    backend = CodexBackend(executable=str(fake_codex), min_interval=0)
    request = prepared_request(backend, codex_config)
    result = backend.create(request)
    assert result["output"][0]["content"][0]["text"] == '{"label":"c5"}'
    assert backend.attempts == 1 and backend.usage["input_tokens"] == 40
    log = json.loads((tmp_path / "calls.jsonl").read_text())
    assert log["prompt"] == request["prompt"]
    assert log["files"] == ["events.jsonl", "instructions.txt", "schema.json", "stderr.log"]
    assert not Path(log["cwd"]).exists()  # The isolated temporary workspace was removed.
    assert str(tmp_path) not in log["cwd"] and "EP2" not in log["cwd"]
    args = log["args"]
    assert "--ignore-user-config" in args and "--ephemeral" in args and "--strict-config" in args
    assert args[args.index("--sandbox")+1] == "read-only"
    for feature in ("shell_tool", "apps", "plugins", "multi_agent", "hooks"):
        assert any(args[i:i+2] == ["--disable", feature] for i in range(len(args)-1))
    assert 'web_search="disabled"' in args and "project_doc_max_bytes=0" in args
    assert "resp_text" not in " ".join(args) and "resume" not in args
    assert "max_output_tokens" not in request and "temperature" not in request


@pytest.mark.parametrize("mode", ["nonzero", "missing_answer", "bad_answer", "duplicate_key", "tool", "failed",
                                   "not_completed", "bad_event", "bad_jsonl", "bad_usage"])
def test_codex_failure_never_produces_prediction_or_automatic_retry(fake_codex, codex_config, tmp_path, monkeypatch, mode):
    monkeypatch.setenv("FAKE_CODEX_MODE", mode)
    backend = CodexBackend(executable=str(fake_codex), min_interval=0)
    with pytest.raises((BackendError, ValueError)) as error:
        run([Text(2,"Alvo")], [], codex_config, backend=backend,
            output_dir=tmp_path / "run", cache_path=tmp_path / "cache.sqlite3")
    assert "secret" not in str(error.value)
    assert backend.attempts == 1
    assert not (tmp_path / "run/predictions.jsonl").exists()
    cache = Cache(tmp_path / "cache.sqlite3")
    assert cache.connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0
    cache.close()


def test_codex_timeout_kills_process_and_preserves_cache(fake_codex, codex_config, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_MODE", "timeout")
    backend = CodexBackend(executable=str(fake_codex), timeout=0.1, min_interval=0)
    with pytest.raises(BackendError, match="Timeout"):
        backend.create(prepared_request(backend, codex_config))
    assert backend.attempts == 1


def test_codex_cache_resume_budget_and_backend_separation(fake_codex, codex_config, tmp_path):
    backend = CodexBackend(executable=str(fake_codex), max_requests=1, min_interval=0)
    kwargs = dict(backend=backend, output_dir=tmp_path / "run", cache_path=tmp_path / "cache.sqlite3")
    texts = [Text(2, "alvo"), Text(3, "alvo")]
    result = run(texts, [], codex_config, **kwargs)
    assert backend.attempts == 1 and result[1]["cache_hit"]
    result = run(texts, [], codex_config, **kwargs)
    assert all(r["cache_hit"] for r in result) and backend.attempts == 1
    api_request, _ = build_request(Text(2,"alvo"),[],Config(strategy="zero",per_class=0))
    codex_request = prepared_request(backend, codex_config)
    assert Cache.key(api_request) != Cache.key(codex_request)
    with pytest.raises(BackendError, match="Limite"):
        backend.create(codex_request)


def test_missing_old_or_changed_runtime_is_rejected(fake_codex, codex_config, tmp_path, monkeypatch):
    with pytest.raises(BackendError, match="não encontrado"):
        CodexBackend(executable=str(tmp_path / "absent")).metadata()
    monkeypatch.setenv("FAKE_CODEX_VERSION", "codex-cli 0.100.0")
    with pytest.raises(BackendError, match=">=0.165.0"):
        CodexBackend(executable=str(fake_codex)).metadata()
    monkeypatch.delenv("FAKE_CODEX_VERSION")
    backend = CodexBackend(executable=str(fake_codex))
    request = prepared_request(backend, codex_config)
    request["runtime"] = {}
    with pytest.raises(BackendError, match="runtime"):
        backend.create(request)
    assert backend.attempts == 0


def test_codex_target_labels_do_not_enter_prompt(make_xlsx, codex_config):
    a = make_xlsx("a.xlsx", [("Alvo", "c1", "SECRET")], headers=("resp_text","clarity","other"))
    b = make_xlsx("b.xlsx", [("Alvo", "c5", "SECRET2")], headers=("resp_text","clarity","other"))
    ra,_ = read_xlsx(a)
    rb,_ = read_xlsx(b)
    request,_ = build_request(ra[0],[],codex_config)
    assert build_request(ra[0],[],codex_config) == build_request(rb[0],[],codex_config)
    assert json.loads(request["prompt"]) == {"examples":[],"target":{"resp_text":"Alvo"}}


def test_codex_cli_selects_config_and_saves_dry_run(fake_codex, tmp_path, capsys):
    result = main(["predict", "--backend", "codex", "--model", "chosen-model", "--codex-bin", str(fake_codex),
                   "--dry-run", "--output-dir", str(tmp_path / "preview"), "--cache", str(tmp_path / "cache.sqlite3")])
    assert result == 0, capsys.readouterr().err
    manifest = json.loads((tmp_path / "preview/run.json").read_text())
    assert manifest["config"]["backend"] == "codex" and manifest["config"]["model"] == "chosen-model"
    assert manifest["backend"]["cli_version"] == "codex-cli 0.165.0"
    assert manifest["rows"] == 897 and manifest["status"] == "dry_run"
    assert not (tmp_path / "calls.jsonl").exists()


@pytest.mark.parametrize("changes", [{"temperature":0}, {"max_output_tokens":256}])
def test_codex_does_not_silently_ignore_unsupported_parameters(codex_config, changes):
    with pytest.raises(ValueError, match="não são expostos"):
        replace(codex_config, **changes)


def test_codex_rejects_api_retry_parameter(codex_config):
    with pytest.raises(ValueError, match="exclusivo da API"):
        make_backend(codex_config, retries=3)
