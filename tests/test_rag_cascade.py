"""Offline synthetic checks only: no real embeddings, datasets, GPU or classifier requests."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from cascade_classifier.config import CascadeConfig
from cascade_classifier.routing import RoutingRule, rank_targets
from ensemble_classifier.soft_voting import QWEN_MODELS
from gpt_classifier.backend import BackendError, Cache
from gpt_classifier.data import LABELS, Example, Text, digest, file_digest, read_jsonl, read_xlsx, write_json, write_jsonl
from rag_cascade.config import RagConfig
from rag_cascade.execution import RequestRunner
from rag_cascade.prompt import build_request
from rag_cascade.retrieval import Retriever, near_duplicate


class Embeddings:
    metadata = {"synthetic": True}

    def encode(self, texts):
        values = np.ones((len(texts), 384), dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)


class Backend:
    calls = []
    fail = False

    def __init__(self):
        self.usage = dict(input_tokens=0, output_tokens=0, cached_input_tokens=0)

    def metadata(self):
        return {"synthetic": True}

    def create(self, request):
        self.calls.append(request)
        if self.fail:
            raise BackendError("Synthetic unavailable transport")
        items = json.loads(request["prompt"])["items"]
        self.usage.update(input_tokens=100, output_tokens=20, cached_input_tokens=10)
        return {"status": "completed", "usage": {"input_tokens": 100, "output_tokens": 20,
                "input_tokens_details": {"cached_tokens": 10}}, "output": [{"type": "message", "content": [
                    {"type": "output_text", "text": json.dumps({"predictions": [{"id": item["id"], "label": "c5"} for item in items]})}]}]}


def model_rows(n):
    targets = [Text(index + 2, f"Synthetic public information response {index}") for index in range(n)]
    rng = np.random.default_rng(41)
    p = rng.dirichlet([5, 5, 5], n)
    parts = {name: rng.dirichlet([3, 3, 3], n) for name in QWEN_MODELS}
    return targets, rank_targets(targets, p, parts, RoutingRule())


def items(n, *, local=False):
    targets, rows = model_rows(n)
    return [{"excel_row": target.excel_row, "request": request, "request_key": Cache.key(request)}
            for target, row in zip(targets, rows)
            for request in [build_request(target, row, [], include_local=local,
                                          config=CascadeConfig(), runtime=Backend().metadata())]]


def test_retrieval_excludes_identical_near_duplicate_and_conflicting_examples():
    target = Text(900, "The information request concerns public budget allocation and the hospital service.")
    bank = [Example(2, target.text, "c1"), Example(3, target.text + " Thank you.", "c234")]
    sentences = ["Information about public budget explains financial limits and approvals.",
                 "A hospital service provides information with detailed patient guidance.",
                 "Public staff answer information requests about payment and legal duties.",
                 "Historical budget information contains tables about annual expenditure.",
                 "Information for citizens about the hospital service and its opening schedule.",
                 "Community projects have information describing eligibility and documentation.",
                 "Agency information shows which budget decisions need public consultation.",
                 "An information request was forwarded to the service responsible for records.",
                 "Published information explains the scope of a hospital funding program."]
    bank += [Example(index + 10, text, LABELS[index % 3]) for index, text in enumerate(sentences)]
    bank += [Example(30, "Conflicting information observation", "c1"), Example(31, "Conflicting information observation", "c5")]
    config = RagConfig(minimum_semantic_similarity=.1, minimum_lexical_similarity=.01)
    retriever = Retriever(bank, {row.excel_row: str(row.excel_row) for row in bank}, config, embeddings=Embeddings())
    for method in config.retrieval_methods:
        examples, report = retriever.retrieve(target, method=method, count=5)
        assert 3 <= len(examples) <= 5
        assert not {2, 3, 30, 31} & {row["excel_row"] for row in examples}
        assert len({row["group_id"] for row in examples}) == len(examples)
        assert all(not near_duplicate(target.text, row["resp_text"], threshold=.85) for row in examples)
        assert set(row["label"] for row in examples) == set(LABELS)
        assert report["returned"] == len(examples)
    assert retriever.identity["selection"]["conflicting_groups_excluded"] == 1


def test_prompt_separates_examples_and_predictions_and_keeps_original_text():
    targets, rows = model_rows(1)
    original = '  <p>Original\n$(do-not-run) `data`</p>\u2028'
    target = Text(2, original)
    rows[0]["text_sha256"] = digest(original)
    example = {"excel_row": 45, "resp_text": "A labeled training observation", "label": "c234",
               "text_sha256": digest("A labeled training observation")}
    plain = build_request(target, rows[0], [], include_local=False, config=CascadeConfig(), runtime={})
    rag = build_request(target, rows[0], [example], include_local=True, config=CascadeConfig(), runtime={})
    payload = json.loads(rag["prompt"])
    assert payload["target"]["resp_text"] == original
    assert payload["examples"] == [{"resp_text": example["resp_text"], "label": "c234"}]
    assert set(payload["target"]) == {"resp_text", "model_evidence"}
    assert set(payload["target"]["model_evidence"]["members"]) == set(QWEN_MODELS)
    assert set(payload["target"]["model_evidence"]["uncertainty"]) == {"maximum_probability", "margin", "entropy", "disagreement"}
    assert json.loads(plain["prompt"])["target"] == {"resp_text": original}
    assert plain["instructions"] == rag["instructions"]
    assert "excel_row" not in rag["prompt"]
    with pytest.raises(TypeError):
        build_request(Example(2, original, "c1"), rows[0], [], include_local=False, config=CascadeConfig(), runtime={})


def test_configuration_rejects_duplicate_arms_and_resolves_environment_before_freezing(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text("{}")
    monkeypatch.setenv("RAG_MAX_NEW_REQUESTS", "64")
    monkeypatch.setenv("CASCADE_MODEL", "gpt-6.1-sol")
    config = RagConfig.load(path)
    assert config.max_new_requests == 64 and config.transport.model == "gpt-6.1-sol"
    assert RagConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        replace(config, example_counts=(3, 3))
    with pytest.raises(ValueError):
        replace(config, retrieval_methods=("tfidf", "unknown"))
    with pytest.raises(ValueError):
        replace(config, transport=replace(CascadeConfig(), budgets=(.1, .5)))


def test_semantic_configuration_never_silently_substitutes_lexical_retrieval(monkeypatch):
    import gpt_classifier.embeddings as embeddings

    def missing(*args, **kwargs):
        raise ImportError("Synthetic missing local encoder")

    monkeypatch.setattr(embeddings, "LocalEmbeddings", missing)
    bank = [Example(i + 2, f"Synthetic labeled bank observation {i}", label) for i, label in enumerate(LABELS)]
    with pytest.raises(ImportError, match="missing local encoder"):
        Retriever(bank, {row.excel_row: str(row.excel_row) for row in bank}, RagConfig())


def test_runner_is_preview_by_default_and_preflight_refuses_over_budget(tmp_path):
    Backend.calls, Backend.fail = [], False
    runner = RequestRunner(CascadeConfig(), tmp_path / "cache.sqlite", backend_factory=Backend)
    _, preview = runner.run(items(9), tmp_path / "gpt", max_new_requests=0)
    assert preview["new_batches"] == 2 and preview["classification_executed"] is False
    assert Backend.calls == [] and not (tmp_path / "gpt").exists()
    with pytest.raises(ValueError, match="no request sent"):
        runner.run(items(9), tmp_path / "gpt", execute=True, max_new_requests=1)
    assert Backend.calls == []


def test_runner_reuses_individual_cache_across_batch_boundaries(tmp_path):
    Backend.calls, Backend.fail = [], False
    runner = RequestRunner(CascadeConfig(), tmp_path / "cache.sqlite", backend_factory=Backend)
    planned = items(10)
    decisions, first = runner.run(planned[:6], tmp_path / "one", execute=True, max_new_requests=1)
    assert first["new_attempts"] == len(Backend.calls) == 1
    _, second = runner.run(planned, tmp_path / "two", execute=True, max_new_requests=1)
    assert second["item_cache_hits"] == 6 and second["new_attempts"] == 1
    replayed, replay = runner.run(planned, tmp_path / "replay", execute=True, max_new_requests=0)
    assert len(Backend.calls) == 2 and replay["new_attempts"] == 0
    assert all(row["label"] == "c5" for row in replayed.values())


def test_retry_circuit_and_fallback_never_exceed_the_explicit_cap(tmp_path, monkeypatch):
    import rag_cascade.execution as execution

    monkeypatch.setattr(execution.time, "sleep", lambda _: None)
    Backend.calls, Backend.fail = [], True
    runner = RequestRunner(replace(CascadeConfig(), workers=1), tmp_path / "cache.sqlite", backend_factory=Backend)
    decisions, report = runner.run(items(17), tmp_path / "failed", execute=True, max_new_requests=3)
    assert len(Backend.calls) == report["new_attempts"] == 2
    assert report["unknown_usage_attempts"] == 2
    assert all(row["status"] == "fallback" and row["label"] is None for row in decisions.values())
    Backend.fail = False


def test_group_split_has_no_pilot_group_in_confirmation_and_uses_no_labels():
    from rag_cascade.data import split_cohorts

    _, rows = model_rows(300)
    groups = {row["excel_row"]: str(index // 2) for index, row in enumerate(rows)}
    split = split_cohorts(rows, groups, pilot_size=10, budget=.1)
    assert len(split["pilot_ids"]) == 10
    assert not {groups[row] for row in split["pilot_ids"]} & {groups[row] for row in split["confirmation_ids"]}
    assert len(split["confirmation_ids"]) == 280


@pytest.fixture
def study_fixture(tmp_path, monkeypatch):
    import rag_cascade.data as data

    Backend.calls, Backend.fail = [], False
    config = RagConfig(pilot_size=6, retrieval_methods=("tfidf",), example_counts=(3,), response_cache=str(tmp_path / "cache.sqlite"))
    directory = tmp_path / "study"
    targets, rows = model_rows(180)
    gold = [Example(row.excel_row, row.text, LABELS[index % 3]) for index, row in enumerate(targets)]
    groups = {row.excel_row: str(row.excel_row) for row in targets}
    split = data.split_cohorts(rows, groups, pilot_size=6, budget=.1)
    bank = [Example(1000 + index, f"Training-only observation {index}", LABELS[index % 3]) for index in range(6)]
    components = {row.excel_row: str(row.excel_row) for row in bank}
    identity = {"synthetic": True}
    monkeypatch.setattr(data, "training_bank", lambda config: (bank, components, identity))
    write_jsonl(directory / "targets.jsonl", targets)
    write_jsonl(directory / "oof-routing.jsonl", rows)
    write_jsonl(directory / "validation-gold.jsonl", gold)
    write_json(directory / "cohorts.json", split)
    write_json(directory / "study.json", {"config": config.to_dict(), "bank": identity, "routing_rule": RoutingRule().__dict__,
               "cohorts": split, "implementation": data.implementation(),
               "files": {name: file_digest(directory / name) for name in ("targets.jsonl", "oof-routing.jsonl", "validation-gold.jsonl", "cohorts.json")}})

    class Retrieval:
        def __init__(self, bank, components, config):
            self.bank = bank
            self.identity = {"synthetic_index": True}
        def prepare_targets(self, targets):
            assert all(type(target) is Text for target in targets)
        def retrieve(self, target, **kwargs):
            return [{"excel_row": row.excel_row, "resp_text": row.text, "label": row.label,
                     "text_sha256": digest(row.text), "relevance": .5} for row in self.bank[:3]], {"synthetic": True}

    factory = lambda config, cache: RequestRunner(config, cache, backend_factory=Backend)
    return directory, config, Retrieval, factory


def test_full_study_is_simulated_and_every_arm_precedes_gold_scoring(study_fixture, tmp_path, monkeypatch):
    import rag_cascade.study as study

    directory, config, retrieval, runner = study_fixture
    for phase in ("pilot", "confirmation"):
        planned = study.plan(directory, phase, retriever_factory=retrieval, runtime=Backend().metadata())
        before = len(Backend.calls)
        preview = study.run(directory, phase, runner_factory=runner)
        assert preview["classification_executed"] is False and len(Backend.calls) == before
        completed = study.run(directory, phase, execute=True, max_new_requests=128, runner_factory=runner)
        original_read = study.read_jsonl

        def checked_read(path):
            if Path(path).name == "validation-gold.jsonl":
                assert all((directory / phase / recipe["name"] / "predictions.jsonl").exists() for recipe in planned["recipes"])
            return original_read(path)

        monkeypatch.setattr(study, "read_jsonl", checked_read)
        selection = study.score(directory, phase)
        monkeypatch.setattr(study, "read_jsonl", original_read)
        assert selection["test1_accessed"] is False and selection["test2_accessed"] is False
        assert study.run(directory, phase, execute=True, runner_factory=runner) == completed
    artifact = study.freeze(directory, tmp_path / "model")
    assert artifact["budget"] in (0., .05, .1)
    assert artifact["test1_accessed"] is False


def test_empty_confirmation_route_uses_stacking_without_requests(study_fixture):
    import rag_cascade.study as study

    directory, _, retrieval, runner = study_fixture
    data = json.loads((directory / "study.json").read_text())
    pilot = set(data["cohorts"]["pilot_ids"])
    rows = read_jsonl(directory / "oof-routing.jsonl")
    for row in rows:
        row["eligible"] = row["excel_row"] in pilot
    write_jsonl(directory / "oof-routing.jsonl", rows)
    data["files"]["oof-routing.jsonl"] = file_digest(directory / "oof-routing.jsonl")
    write_json(directory / "study.json", data)
    for phase in ("pilot", "confirmation"):
        plan = study.plan(directory, phase, retriever_factory=retrieval, runtime=Backend().metadata())
        before = len(Backend.calls)
        study.run(directory, phase, execute=True, max_new_requests=128, runner_factory=runner)
        result = study.score(directory, phase)
        if phase == "confirmation":
            assert plan["targets"] == 0 and len(Backend.calls) == before
            assert result["budget"] == 0.


@pytest.mark.parametrize("execute", [False, True])
def test_test1_is_blocked_before_any_model_or_gpt_work_without_test2(tmp_path, monkeypatch, execute):
    import rag_cascade.prediction as prediction

    data = tmp_path / "data"
    data.mkdir()
    for name in ("test1.xlsx", "test1-rotulado-para-teste-estatistico.xlsx", "test2.xlsx"):
        (data / name).write_bytes(name.encode())
    model = tmp_path / "model.json"
    model.write_text("{}")
    monkeypatch.setattr(prediction, "ROOT", tmp_path)
    monkeypatch.setattr(prediction, "load_model", lambda path: ({}, RagConfig(), [], {}))
    monkeypatch.setattr(prediction, "frozen_inputs", lambda *args: pytest.fail("Must check the test2 guard first"))
    with pytest.raises(ValueError, match="requires a completed test2"):
        prediction.predict(model, data / "test1.xlsx", tmp_path / "unused", tmp_path / "out", execute=execute)


def test_actual_synthetic_test2_then_test1_chain_and_gold_access(tmp_path, make_xlsx, monkeypatch):
    import rag_cascade.prediction as prediction

    root = tmp_path / "data"
    root.mkdir()
    sources = {}
    for name, n in (("test2", 897), ("test1", 900)):
        original = make_xlsx(name + ".xlsx", [(f"Synthetic original {name} response {index}", None) for index in range(n)])
        target = root / original.name
        target.write_bytes(original.read_bytes())
        sources[name] = target
    gold = make_xlsx("test1-rotulado-para-teste-estatistico.xlsx", [(f"Synthetic original test1 response {i}", LABELS[i % 3]) for i in range(900)])
    (root / gold.name).write_bytes(gold.read_bytes())
    model = tmp_path / "model.json"
    model.write_text("{}")
    stack_run = tmp_path / "stack"
    write_json(stack_run / "run.json", {})
    config = RagConfig(response_cache=str(tmp_path / "cache.sqlite"))
    artifact = {"budget": 0., "routing_rule": RoutingRule().__dict__, "recipe": {"method": None, "count": 0, "include_local": False},
                "runtime": Backend().metadata()}
    monkeypatch.setattr(prediction, "ROOT", tmp_path)
    monkeypatch.setattr(prediction, "load_model", lambda path: (artifact, config, [], {}))

    def inputs(config, source, run):
        targets, metadata = read_xlsx(source)
        p = np.repeat([[.2, .3, .5]], len(targets), axis=0)
        members = {name: [{"label": "c5"} for row in targets] for name in QWEN_MODELS}
        return targets, metadata, dict.fromkeys(QWEN_MODELS, p), members, p

    monkeypatch.setattr(prediction, "frozen_inputs", inputs)
    runner = lambda config, cache: RequestRunner(config, cache, backend_factory=Backend)
    Backend.calls, Backend.fail = [], False
    test2 = tmp_path / "test2-run"
    preview = prediction.predict(model, sources["test2"], stack_run, test2, runner_factory=runner)
    assert preview["status"] == "preview" and not (test2 / "predictions.jsonl").exists()
    completed = prediction.predict(model, sources["test2"], stack_run, test2, execute=True, runner_factory=runner)
    assert completed["rows"] == 897 and (test2 / "test2.xlsx").exists()
    test1 = tmp_path / "test1-run"
    prediction.predict(model, sources["test1"], stack_run, test1, test2_run=test2, execute=True, runner_factory=runner)
    actual_read = prediction.read_xlsx

    def read_gold(path, **kwargs):
        assert (test2 / "test2.xlsx").exists() and (test1 / "predictions.jsonl").exists()
        return actual_read(path, **kwargs)

    monkeypatch.setattr(prediction, "read_xlsx", read_gold)
    result = prediction.evaluate(test1, gold, tmp_path / "evaluation")
    assert result["gold_read_after_test2_and_all_test1_predictions"] is True
    assert result["parameters_updated"] is False and Backend.calls == []
    (test2 / "test2.xlsx").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        prediction.evaluate(test1, gold, tmp_path / "tampered")


def test_no_encoder_download_without_explicit_flag(monkeypatch):
    from rag_cascade.cli import main

    assert main(["prepare-encoder"]) == 0


def test_previous_frozen_cascade_remains_loadable():
    from cascade_classifier.experiment import load_cascade

    root = Path(__file__).resolve().parents[1]
    artifact, _ = load_cascade(root / "models/stacking-gpt-cascade/model.json")
    assert artifact["selected_budget"] == 0.
