from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest
from openai import OpenAI

from gpt_classifier.backend import BackendError, Cache, OpenAIBackend
from gpt_classifier.data import Text
from gpt_classifier.pipeline import run
from gpt_classifier.prompt import Config, build_request
from conftest import response


def test_missing_key_fails_before_any_attempt(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = OpenAIBackend()
    with pytest.raises(BackendError, match="OPENAI_API_KEY"):
        backend.create({"model": "mock"})
    assert backend.attempts == 0


@pytest.mark.parametrize("kwargs", [{"timeout": float("nan")}, {"timeout": float("inf")},
                                     {"min_interval": float("nan")}, {"retries": -1}])
def test_execution_limits_are_finite(kwargs):
    with pytest.raises(ValueError):
        OpenAIBackend(**kwargs)


def make_backend(handler, **kwargs):
    client = OpenAI(api_key="test-placeholder", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return OpenAIBackend(client=client, sleep=lambda _: None, min_interval=0, **kwargs)


def test_real_sdk_wire_schema_and_usage():
    observed = []
    def handler(request):
        assert request.url.path == "/v1/responses"
        observed.append(json.loads(request.content))
        return httpx.Response(200, json=response())
    backend = make_backend(handler)
    config = Config(strategy="zero", per_class=0)
    payload, _ = build_request(Text(2, "Teste"), [], config)
    assert backend.create(payload)["status"] == "completed"
    assert observed == [payload]
    assert backend.usage == {"input_tokens": 20, "output_tokens": 5, "cached_input_tokens": 3}


@pytest.mark.parametrize("status,code,retry", [(429, "rate_limit_exceeded", True), (500, "server_error", True),
    (401, "invalid_api_key", False), (400, "bad_request", False), (429, "insufficient_quota", False)])
def test_transient_retries_and_no_auth_or_quota_retry(status, code, retry):
    calls = []
    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(status, json={"error": {"message": "secret input", "type": "error", "code": code}})
        return httpx.Response(200, json=response())
    backend = make_backend(handler)
    if retry:
        backend.create({"model": "mock", "input": "test"})
        assert len(calls) == 2
    else:
        with pytest.raises(BackendError) as error:
            backend.create({"model": "mock", "input": "test"})
        assert "secret input" not in str(error.value) and len(calls) == 1


def test_retry_after_and_request_budget():
    backend = make_backend(lambda _: httpx.Response(429, headers={"Retry-After": "120"}, json={"error": {"message": "wait"}}))
    with pytest.raises(BackendError, match="Retry-After"):
        backend.create({"model": "mock"})
    assert backend.attempts == 1
    backend = make_backend(lambda _: httpx.Response(500, json={"error": {"message": "failure"}}), max_requests=1)
    with pytest.raises(BackendError, match="Limite"):
        backend.create({"model": "mock"})
    assert backend.attempts == 1


def test_cache_resume_after_partial_failure_and_duplicates(tmp_path, bank):
    calls = []
    def handler(request):
        calls.append(request)
        if len(calls) == 2:
            return httpx.Response(401, json={"error": {"message": "invalid"}})
        return httpx.Response(200, json=response())
    backend = make_backend(handler)
    config = Config(strategy="zero", per_class=0)
    records = [Text(2, "A"), Text(3, "B"), Text(4, "A")]
    kwargs = dict(output_dir=tmp_path / "run", cache_path=tmp_path / "cache.sqlite3")
    with pytest.raises(BackendError):
        run(records, bank, config, backend=backend, **kwargs)
    assert not (tmp_path / "run/predictions.jsonl").exists()
    assert json.loads((tmp_path / "run/run.json").read_text())["status"] == "failed"
    next_backend = make_backend(lambda _: httpx.Response(200, json=response()))
    predictions = run(records, bank, config, backend=next_backend, **kwargs)
    assert next_backend.attempts == 1 and len(predictions) == 3
    assert [p["cache_hit"] for p in predictions] == [True, False, True]
    manifest = json.loads((tmp_path / "run/run.json").read_text())
    assert manifest["status"] == "completed" and len(manifest["attempt_history"]) == 2
    assert manifest["new_usage"]["input_tokens"] == 20
    with pytest.raises(ValueError, match="outro experimento"):
        run(records, bank, replace(config, model="different"), backend=next_backend, **kwargs)


def test_refusal_is_not_cached_or_exported(tmp_path, bank):
    value = response(status="incomplete")
    backend = make_backend(lambda _: httpx.Response(200, json=value))
    with pytest.raises(ValueError, match="incompleta"):
        run([Text(2, "alvo")], bank, Config(strategy="zero", per_class=0), backend=backend,
            output_dir=tmp_path / "run", cache_path=tmp_path / "cache.sqlite3")
    cache = Cache(tmp_path / "cache.sqlite3")
    assert cache.connection.execute("select count(*) from responses").fetchone()[0] == 0
    cache.close()
    assert (tmp_path / "run/failed-response.json").exists()


def test_dry_run_makes_no_calls_and_cannot_masquerade_as_predictions(tmp_path, bank):
    class Forbidden:
        def create(self, _):
            raise AssertionError("dry-run contacted API")
    assert run([Text(2, "alvo")], bank, Config(), backend=Forbidden(), dry_run=True,
               output_dir=tmp_path / "run", cache_path=tmp_path / "cache.sqlite3") == []
    assert json.loads((tmp_path / "run/run.json").read_text())["status"] == "dry_run"
    assert not (tmp_path / "run/predictions.jsonl").exists()
    assert not (tmp_path / "cache.sqlite3").exists()
