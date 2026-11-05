"""Exercise isolation, percentage caps, persistent caching, retries and frozen evaluation."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from cascade_classifier.config import CascadeConfig
from cascade_classifier.dataset import nested_predictions
from cascade_classifier.experiment import apply_decisions, comparison
from cascade_classifier.gpt import SelectiveGPT, build_request
from cascade_classifier.routing import RoutingRule, choose_rule, rank_targets, select_targets
from ensemble_classifier.soft_voting import QWEN_MODELS
from gpt_classifier.backend import BackendError
from gpt_classifier.data import LABELS, Example, Text, digest, file_digest, write_json, write_jsonl


def fixture_rows(n=200):
    rng = np.random.default_rng(17)
    records = [Text(i + 2, f"Original text {i}") for i in range(n)]
    members = {name: rng.dirichlet([2, 2, 2], n) for name in QWEN_MODELS}
    stacking = rng.dirichlet([5, 5, 5], n)
    return records, stacking, members


def test_routing_is_label_free_and_percentage_caps_are_nested():
    records, stacking, members = fixture_rows(1972)
    rows = rank_targets(records, stacking, members, RoutingRule())
    selections = [select_targets(rows, budget) for budget in (.05, .1, .2)]
    assert [len(items) for items in selections] == [98, 197, 394]
    assert {row["excel_row"] for row in selections[0]} < {row["excel_row"] for row in selections[1]} < {row["excel_row"] for row in selections[2]}
    changed = rank_targets(records, stacking, members, RoutingRule())
    assert rows == changed
    with pytest.raises(TypeError):
        rank_targets([Example(r.excel_row, r.text, "c1") for r in records], stacking, members, RoutingRule())
    for budget in (True, -.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            select_targets(rows, budget)


def test_disagreement_uses_all_four_members_and_changes_priority():
    records = [Text(2, "one"), Text(3, "two")]
    p = np.array([[.34, .33, .33]] * 2)
    members = {name: np.array([[.8, .1, .1]] * 2) for name in QWEN_MODELS}
    members["qwen"][1] = [.1, .8, .1]
    rows = rank_targets(records, p, members, RoutingRule(.5))
    assert rows[0]["disagreement"] == 0 and rows[1]["disagreement"] == .5
    assert rows[1]["uncertainty_rank"] == 1


def test_validation_rule_search_records_only_oof_error_capture():
    targets, p, members = fixture_rows()
    labels = [LABELS[i % 3] for i in range(len(targets))]
    rule, trials = choose_rule(targets, p, members, labels)
    assert len(trials) == 12
    assert max(trial["captured_errors"] for trial in trials) == next(
        trial["captured_errors"] for trial in trials if trial["rule"] == rule.__dict__)


def test_nested_oof_excludes_outer_targets_from_inner_regularization(monkeypatch):
    import cascade_classifier.dataset as dataset

    targets, _, members = fixture_rows(150)
    records = [Example(row.excel_row, row.text, LABELS[i % 3]) for i, row in enumerate(targets)]
    calls = []
    original_fit = dataset.fit_meta

    def fit(matrices, labels, groups):
        calls.append(set(int(group) for group in groups))
        return original_fit(matrices, labels, groups)

    monkeypatch.setattr(dataset, "fit_meta", fit)
    probabilities, assignments, audits = nested_predictions(records, np.arange(150), members)
    assert probabilities.shape == (150, 3) and set(assignments) == set(range(5))
    for audit, training_groups in zip(audits, calls):
        valid = {row - 2 for row in audit["validation_ids"]}
        assert not training_groups & valid
        assert audit["model"]["C"] == audit["inner_search"]["selected_C"]
    assert sorted(row for audit in audits for row in audit["validation_ids"]) == [row.excel_row for row in records]


def test_gpt_payload_preserves_original_text_and_never_sends_gold():
    config = CascadeConfig()
    targets, p, members = fixture_rows(3)
    text = '  <p>Original\n$(secret) `command`</p>\u2028'
    targets[0] = Text(2, text)
    rows = rank_targets(targets, p, members, RoutingRule())
    plain = build_request(targets[0], rows[0], False, config, {})
    local = build_request(targets[0], rows[0], True, config, {})
    assert json.loads(plain["prompt"])["target"] == {"resp_text": text}
    assert "local_predictions" in json.loads(local["prompt"])["target"]
    assert "temperature" not in plain and plain["reasoning_effort"] == "low"
    with pytest.raises(TypeError):
        build_request(Example(2, text, "c5"), rows[0], False, config, {})


class FakeBackend:
    calls = []
    fail = False

    def __init__(self):
        self.usage = dict(input_tokens=0, output_tokens=0, cached_input_tokens=0)

    def metadata(self):
        return {"kind": "codex", "synthetic": True}

    def create(self, request):
        self.calls.append(request)
        if self.fail:
            raise BackendError("Synthetic failure")
        items = json.loads(request["prompt"])["items"]
        answer = {"predictions": [{"id": row["id"], "label": "c5"} for row in items]}
        self.usage.update(input_tokens=100, output_tokens=20, cached_input_tokens=10)
        return {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(answer)}]}],
                "usage": {"input_tokens": 100, "output_tokens": 20, "input_tokens_details": {"cached_tokens": 10}}}


@pytest.mark.parametrize("workers", [1, 2])
def test_persistent_item_cache_survives_budget_and_batch_boundaries(tmp_path, workers):
    FakeBackend.calls, FakeBackend.fail = [], False
    records, p, members = fixture_rows(23)
    rows = rank_targets(records, p, members, RoutingRule())
    service = SelectiveGPT(replace(CascadeConfig(), workers=workers), tmp_path / "cache.sqlite", backend_factory=FakeBackend)
    decisions, report = service.execute(records, rows[:10], tmp_path / "first", include_local=False)
    assert len(FakeBackend.calls) == 2 and report["attempts"] == 2
    assert report["usage"]["input_tokens"] == 200
    decisions, second = service.execute(records, rows[:15], tmp_path / "second", include_local=False)
    assert len(FakeBackend.calls) == 3 and second["initial_item_cache_hits"] == 10
    decisions, third = service.execute(records, rows[:15], tmp_path / "third", include_local=False)
    assert len(FakeBackend.calls) == 3 and third["attempts"] == 0
    assert all(decision["label"] == "c5" for decision in decisions.values())
    service.execute(records, rows[:2], tmp_path / "different-prompt", include_local=True)
    assert len(FakeBackend.calls) == 4


def test_failures_retry_with_cap_and_fall_back_without_poisoning_cache(tmp_path, monkeypatch):
    import cascade_classifier.gpt as module

    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    FakeBackend.calls, FakeBackend.fail = [], True
    records, p, members = fixture_rows(20)
    rows = rank_targets(records, p, members, RoutingRule())
    service = SelectiveGPT(replace(CascadeConfig(), workers=1), tmp_path / "cache.sqlite", backend_factory=FakeBackend)
    decisions, report = service.execute(records, rows, tmp_path / "failed", include_local=False)
    assert len(FakeBackend.calls) == 2  # The circuit breaker stops subsequent batches.
    assert report["unknown_usage_attempts"] == 2 and report["failed_attempts"] == 2
    result = apply_decisions(rows, rows, decisions)
    assert all(row["decision_source"] == "fallback" for row in result)
    assert [row["label"] for row in result] == [row["label"] for row in rows]
    FakeBackend.fail = False
    recovered, _ = service.execute(records, rows, tmp_path / "recovery", include_local=False)
    assert all(row["label"] == "c5" for row in recovered.values())


def test_metrics_count_corrections_harm_and_preserve_unrouted_decisions():
    records = [Example(2, "a", "c1"), Example(3, "b", "c5"), Example(4, "c", "c234")]
    baseline = [{"excel_row": row.excel_row, "text_sha256": digest(row.text), "label": "c1"} for row in records]
    decisions = {2: {"label": "c5", "request_key": "one"}, 3: {"label": "c5", "request_key": "two"}}
    predicted = apply_decisions(baseline, baseline[:2], decisions)
    score, changes = comparison(records, baseline, predicted, baseline[:2], decisions, CascadeConfig())
    assert score["changes"] == {"corrected": 1, "harmed": 1, "both_wrong_changed_label": 0}
    assert predicted[2]["label"] == "c1" and predicted[2]["decision_source"] == "stacking"
    assert len(changes) == 2


def test_environment_overrides_validate_model_and_budget(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text("{}")
    monkeypatch.setenv("CASCADE_MODEL", "gpt-6.1-sol")
    monkeypatch.setenv("CASCADE_REASONING_EFFORT", "low")
    monkeypatch.setenv("CASCADE_WORKERS", "2")
    config = CascadeConfig.load(path)
    assert config.model == "gpt-6.1-sol" and config.workers == 2
    with pytest.raises(ValueError):
        replace(config, budgets=(.2, .1))


def test_native_svm_vote_is_preserved_when_calibration_changes_argmax():
    records, p, members = fixture_rows(3)
    for name in members:
        members[name][:] = [.8, .1, .1]
    native = {name: ["c1"] * 3 for name in QWEN_MODELS}
    native["svm"][0] = "c5"
    rows = rank_targets(records, p, members, RoutingRule(), native_labels=native)
    assert rows[0]["local_predictions"]["svm"]["label"] == "c5"
    assert rows[0]["disagreement"] == .5


def test_final_prediction_never_reads_gold_and_freezes_all_budget_outputs(tmp_path, monkeypatch):
    import cascade_classifier.experiment as module

    FakeBackend.calls, FakeBackend.fail = [], False
    records, p, members = fixture_rows(900)
    config = CascadeConfig(stacking_model=str(tmp_path / "stack.json"))
    stack_path = Path(config.stacking_model)
    stack_path.write_text("{}")
    cascade_path = tmp_path / "cascade.json"
    cascade_path.write_text("{}")
    source_path = tmp_path / "input.xlsx"
    source_path.write_bytes(b"synthetic workbook identity")
    source = {"sha256": file_digest(source_path)}
    artifact = {"config": config.to_dict(), "routing_rule": RoutingRule().__dict__, "selected_budget": .1,
                "comparison_budgets": [0., .05, .1, .2], "prompt_mode": "text_only", "codex_runtime": FakeBackend().metadata()}
    monkeypatch.setattr(module, "load_cascade", lambda path: (artifact, config))
    monkeypatch.setattr(module, "load_stacking_artifact", lambda path: ({"soft_model": "unused", "meta_model": {}}, {}))
    native = {name: [{"label": LABELS[int(part.argmax())]} for part in members[name]] for name in QWEN_MODELS}
    monkeypatch.setattr(module, "saved_soft_inputs", lambda *args: (records, source, members, native, {}))
    monkeypatch.setattr(module, "meta_probabilities", lambda *args: p)
    stacking_run = tmp_path / "stacking-run"
    predictions = [{"excel_row": row.excel_row, "text_sha256": digest(row.text), "label": LABELS[int(part.argmax())],
                    "probabilities": dict(zip(LABELS, part.tolist()))} for row, part in zip(records, p)]
    write_jsonl(stacking_run / "predictions.jsonl", predictions)
    write_json(stacking_run / "run.json", {"status": "completed", "artifact_sha256": file_digest(stack_path),
               "soft_run": "unused", "soft_files": {}, "input": source,
               "predictions_sha256": file_digest(stacking_run / "predictions.jsonl")})

    def forbid_gold(*args, **kwargs):
        raise AssertionError("Prediction must never load final gold")

    monkeypatch.setattr(module, "read_xlsx", forbid_gold)
    service = SelectiveGPT(config, tmp_path / "cache.sqlite", backend_factory=FakeBackend)
    default = module.predict_cascade(cascade_path, source_path, stacking_run, tmp_path / "default", tmp_path / "cache.sqlite", service=service)
    assert default["budgets"] == [0., .1] and default["calls"]["targets"] == 90
    final = tmp_path / "final"
    report = module.predict_cascade(cascade_path, source_path, stacking_run, final, tmp_path / "cache.sqlite", service=service, compare_budgets=True)
    assert report["calls"]["targets"] == 180 and report["calls"]["initial_item_cache_hits"] == 90
    assert report["gold_accessed"] is False
    before = {name: file_digest(final / name) for name in report["files"]}

    def read_gold(path, *, labeled):
        assert labeled and all((final / f"predictions-{budget}.jsonl").exists() for budget in artifact["comparison_budgets"])
        return [Example(row.excel_row, row.text, LABELS[i % 3]) for i, row in enumerate(records)], {"sha256": file_digest(path)}

    monkeypatch.setattr(module, "read_xlsx", read_gold)
    result = module.evaluate_cascade(final, source_path, tmp_path / "evaluation")
    assert result["selected_budget_from_validation"] == .1
    assert [result["scores"][str(budget)]["routed_instances"] for budget in [0., .05, .1, .2]] == [0, 45, 90, 180]
    assert before == {name: file_digest(final / name) for name in report["files"]}
    assert result["parameters_updated"] is False
    (final / "routing.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="changed"):
        module.evaluate_cascade(final, source_path, tmp_path / "tampered-evaluation")
