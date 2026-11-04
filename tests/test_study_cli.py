from __future__ import annotations

import json
from pathlib import Path

import httpx
from openai import OpenAI

from gpt_classifier.backend import OpenAIBackend
from gpt_classifier.cli import main
from gpt_classifier.data import group
from gpt_classifier.prompt import Config
from gpt_classifier.study import develop
from conftest import response


def test_development_end_to_end_selects_without_accessing_test_gold(tmp_path, bank):
    calls = []
    labels_by_text = {r.text: r.label for r in bank}
    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        # Transport simulates perfect predictions; no target label appears in
        # any final input message. This is a fixture, never a measured GPT score.
        text = json.loads(payload["input"][-1]["content"])["resp_text"]
        for message in payload["input"][1:-1]:
            if message["role"] == "user":
                example = json.loads(message["content"])["resp_text"]
                assert group(example) != group(text)
        return httpx.Response(200, json=response(labels_by_text[text]))
    client = OpenAI(api_key="test-placeholder", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    backend = OpenAIBackend(client=client, sleep=lambda _: None, min_interval=0, max_requests=50)
    directory = tmp_path / "study"
    result = develop(bank, Config(), output_dir=directory, cache_path=tmp_path / "cache.sqlite3", limit=6, backend=backend)
    assert result["candidate"] == "candidate-00"  # Prespecified tie rule.
    assert json.loads((directory / "selected-config.json").read_text())["strategy"] == "zero"
    assert len(calls) == 24
    for candidate in directory.glob("candidate-*"):
        manifest = json.loads((candidate / "run.json").read_text())
        assert manifest["new_api_attempts"] == 6
        assert manifest["new_usage"]["input_tokens"] == 120
    # Repeated invocation is served from cache; no additional remote attempts.
    develop(bank, Config(), output_dir=directory, cache_path=tmp_path / "cache.sqlite3", limit=6, backend=backend)
    assert len(calls) == 24


def test_cli_refuses_statistical_before_test2_and_refuses_dry_run_export(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    assert main(["predict", "--phase", "statistical", "--input",
                 str(root / "data/test1-rotulado-para-teste-estatistico.xlsx"),
                 "--output-dir", str(tmp_path / "run")]) == 2
    assert "Primeiro conclua teste2" in capsys.readouterr().err
    directory = tmp_path / "preview"
    directory.mkdir()
    (directory / "run.json").write_text(json.dumps({"status": "dry_run", "rows": 897}))
    assert main(["export", "--run", str(directory), "--output", str(tmp_path / "test2.xlsx")]) == 2
    assert not (tmp_path / "test2.xlsx").exists()


def test_standalone_test1_predict_export_and_evaluate(tmp_path, monkeypatch, capsys):
    from gpt_classifier import cli
    from gpt_classifier.data import read_xlsx
    from conftest import response
    root = Path(__file__).resolve().parents[1]
    seen = []
    def handler(request):
        payload = json.loads(request.content)
        target = json.loads(payload["input"][-1]["content"])
        assert set(target) == {"resp_text"}
        seen.append(target)
        return httpx.Response(200, json=response())
    client = OpenAI(api_key="test-placeholder", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    backend = OpenAIBackend(client=client, min_interval=0, sleep=lambda _: None)
    monkeypatch.setattr(cli, "make_backend", lambda *a, **k: backend)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(Config(strategy="zero", per_class=0).to_dict()))
    directory = tmp_path / "prediction"
    assert main(["predict", "--phase", "comparison", "--config", str(config),
                 "--input", str(root / "data/test1.xlsx"), "--output-dir", str(directory),
                 "--cache", str(tmp_path / "cache.sqlite3")]) == 0
    assert seen
    output = tmp_path / "test1-predito.xlsx"
    assert main(["export", "--purpose", "comparison", "--input", str(root / "data/test1.xlsx"),
                 "--run", str(directory), "--output", str(output)]) == 0
    exported, _ = read_xlsx(output, labeled=True)
    assert len(exported) == 900
    comparison = tmp_path / "evaluation"
    assert main(["evaluate", "--run", str(directory), "--output-dir", str(comparison)]) == 0
    report = json.loads((comparison / "metrics.json").read_text())
    assert report["ep1"]["accuracy"] == 0.48 and report["ep2"]["rows"] == 900
    assert "sem vínculo a teste2" in report["caveat"]
    # The comparison workbook cannot be exported through the official delivery path.
    assert main(["export", "--input", str(root / "data/test1.xlsx"), "--run", str(directory),
                 "--output", str(tmp_path / "wrong-delivery.xlsx")]) == 2
    assert not (tmp_path / "wrong-delivery.xlsx").exists()
