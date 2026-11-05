from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import f1_score

from ensemble_classifier import soft_pipeline as pipeline
from ensemble_classifier.soft_voting import (MODELS, FrozenScores, aligned_scores, apply_calibration,
                                              choose_weights, fit_calibrations, soft_vote)
from gpt_classifier.data import LABELS, Example, Text, digest, file_digest, write_json, write_jsonl


def sample(n=180):
    rng = np.random.default_rng(42)
    labels = np.asarray([LABELS[i % 3] for i in range(n)])
    raw = {"svm": rng.normal(size=(n, 3)), "rl": rng.dirichlet([2, 2, 2], n),
           "norberto": rng.dirichlet([1, 1, 1], n)}
    return labels, raw


def test_aligned_classes_reject_missing_duplicates_and_invalid_values():
    assert np.allclose(aligned_scores([{"probabilities": [.6, .1, .3]}], "probabilities", ["c5", "c1", "c234"]), [[.1, .3, .6]])
    assert np.allclose(aligned_scores([{"probabilities": {"c5": .6, "c234": .3, "c1": .1}}], "probabilities"), [[.1, .3, .6]])
    for row in ({"label": "c1"}, {"probabilities": [1, 1, 1]}, {"probabilities": [float("nan"), 0, 1]},
                {"probabilities": {"c1": 1, "c5": 0, "unknown": 0}}):
        with pytest.raises(ValueError):
            aligned_scores([row], "probabilities")
    with pytest.raises(ValueError):
        aligned_scores([{"probabilities": [.6, .1, .3]}], "probabilities", ["c1", "c1", "c5"])


def test_disjoint_stratified_group_roles():
    labels = np.asarray([LABELS[(i // 2) % 3] for i in range(600)])
    groups = np.arange(600) // 2
    calibration, weights = pipeline.separate_holdout(labels, groups, 42)
    assert not set(calibration) & set(weights)
    assert not set(groups[calibration]) & set(groups[weights])
    assert set(calibration) | set(weights) == set(range(600))
    for indices in (calibration, weights):
        assert set(labels[indices]) == set(LABELS)
        assert max(np.mean(labels[indices] == label) for label in LABELS) < .4


def test_sigmoid_adds_probabilities_without_base_fit_and_temperature_stays_in_calibration():
    labels, raw = sample()
    with pytest.raises(RuntimeError, match="prohibited"):
        FrozenScores().fit(raw["svm"], labels)
    calibrators = fit_calibrations(raw, labels, np.arange(len(labels)))
    assert calibrators["svm"]["method"] == "sigmoid_ovr"
    for model in MODELS:
        p = apply_calibration(raw[model], calibrators[model])
        assert np.isfinite(p).all() and np.all(p >= 0) and np.allclose(p.sum(axis=1), 1)
    for model in ("rl", "norberto"):
        assert calibrators[model]["calibration_only_check"]["folds"] == 3
        assert np.array_equal(raw[model].argmax(axis=1), apply_calibration(raw[model], calibrators[model]).argmax(axis=1))


def test_fixed_sixteen_weight_candidates_optimize_macro_f1_and_keep_uniform_tie():
    labels, raw = sample()
    p = {m: raw[m] if m != "svm" else np.ones((len(labels), 3)) / 3 for m in MODELS}
    weights, trials = choose_weights(p, labels)
    assert len(trials) == 16 and np.isclose(sum(weights.values()), 1)
    predicted = np.asarray(LABELS)[soft_vote(p, weights).argmax(axis=1)]
    assert f1_score(labels, predicted, labels=list(LABELS), average="macro") == max(t["f1_macro"] for t in trials)
    identical = dict.fromkeys(MODELS, np.tile([.1, .3, .6], (len(labels), 1)))
    tied, _ = choose_weights(identical, labels)
    assert tied == dict.fromkeys(MODELS, 1 / 3)
    for bad in ({"svm": -1, "rl": 1, "norberto": 1}, {"svm": 1, "rl": 1, "norberto": 1},
                {"svm": float("inf"), "rl": 0, "norberto": 0}):
        with pytest.raises(ValueError):
            soft_vote(p, bad)


@pytest.fixture
def offline_soft(tmp_path, monkeypatch):
    records = [Example(i + 2, f"holdout {i}", LABELS[i % 3]) for i in range(180)]
    config = {"seed": 42, "train": str(tmp_path / "training.xlsx"), "split": "fixture", "recorded_norberto": {}}
    source = {"sha256": pipeline.DATASET_SHA256}
    identities = {m: {"artifact_sha256": digest(m)} for m in MODELS}
    (tmp_path / "training.xlsx").write_bytes(b"fixture")
    source["sha256"] = file_digest(tmp_path / "training.xlsx")
    monkeypatch.setattr(pipeline, "load_soft_config", lambda _: config)
    monkeypatch.setattr(pipeline, "holdout_records", lambda _: (records, np.arange(180), source, records, {}))
    monkeypatch.setattr(pipeline, "model_identities", lambda: identities)
    monkeypatch.setattr(pipeline, "verify_lr_training", lambda *args: None)
    reads = []
    def scoring(config, targets, source, models, directory, **kwargs):
        assert all(type(r) is Text and not hasattr(r, "label") for r in targets)
        reads.extend(targets)
        _, raw = sample(len(targets))
        for name in MODELS:
            root = directory / name
            root.mkdir(parents=True, exist_ok=True)
            field = "decision_scores" if name == "svm" else "probabilities"
            write_jsonl(root / "predictions.jsonl", [
                {"excel_row": target.excel_row, "text_sha256": digest(target.text),
                 "label": LABELS[int(scores.argmax())], field: dict(zip(LABELS, scores.tolist()))}
                for target, scores in zip(targets, raw[name])])
            write_json(root / "run.json", {"status": "completed", "predictions_sha256": file_digest(root / "predictions.jsonl")})
        return raw, {}
    monkeypatch.setattr(pipeline, "collect_scores", scoring)
    output = tmp_path / "fit"
    return output, reads, records


def test_fit_routes_nonoverlapping_arrays_to_calibration_and_macro_f1_search(offline_soft, monkeypatch):
    output, reads, records = offline_soft
    seen = {}
    original_calibrate, original_choose = pipeline.fit_calibrations, pipeline.choose_weights
    def calibrate(raw, labels, groups, **kwargs):
        seen["calibration"] = set(groups)
        return original_calibrate(raw, labels, groups, **kwargs)
    def choose(raw, labels):
        seen["weight_rows"] = len(labels)
        return original_choose(raw, labels)
    monkeypatch.setattr(pipeline, "fit_calibrations", calibrate)
    monkeypatch.setattr(pipeline, "choose_weights", choose)
    report = pipeline.fit_soft(output / "unused.json", output)
    partition = [json.loads(line) for line in (output / "partition.jsonl").read_text().splitlines()]
    cal = {r["excel_row"] - 2 for r in partition if r["role"] == "calibration"}
    weights = {r["excel_row"] - 2 for r in partition if r["role"] == "weight_selection"}
    assert seen["calibration"] == cal and not cal & weights and seen["weight_rows"] == len(weights)
    assert report["fixed_candidate_count"] == 16 and report["test1_accessed"] is False
    assert len(reads) == len(records) and report["base_training_performed"] is False


def test_predict_then_final_gold_read_never_refits_and_detects_tampering(offline_soft, make_xlsx, tmp_path, monkeypatch):
    output, _, _ = offline_soft
    pipeline.fit_soft(output / "unused.json", output)
    source = make_xlsx("test-unlabeled.xlsx", [(f"text {i}", None) for i in range(900)])
    gold = make_xlsx("gold.xlsx", [(f"text {i}", LABELS[i % 3]) for i in range(900)])
    monkeypatch.setattr(pipeline, "fit_calibrations", lambda *a, **k: pytest.fail("Refitting calibration during prediction"))
    monkeypatch.setattr(pipeline, "choose_weights", lambda *a, **k: pytest.fail("Retuning weights on final test"))
    artifact_before = file_digest(output / "model.json")
    real_reader = pipeline.read_xlsx
    final = tmp_path / "final"
    calls = []
    def guarded_reader(path, **kwargs):
        if kwargs.get("labeled"):
            calls.append(path)
            assert all((final / "members" / m / "predictions.jsonl").is_file() for m in MODELS)
            assert all((final / "calibrated" / m / "predictions.jsonl").is_file() for m in MODELS)
            assert (final / "predictions.jsonl").is_file()
        return real_reader(path, **kwargs)
    monkeypatch.setattr(pipeline, "read_xlsx", guarded_reader)
    result = pipeline.predict_soft(output / "model.json", source, final)
    assert result["gold_accessed"] is False and not calls
    evaluated = pipeline.evaluate_soft(final, gold, tmp_path / "evaluation")
    assert calls == [gold] and evaluated["parameters_updated"] is False
    assert set(evaluated["metrics"]) == {"ensemble", *MODELS, *(m + "_calibrated" for m in MODELS)}
    assert file_digest(output / "model.json") == artifact_before
    (final / "calibrated/svm/predictions.jsonl").write_text("{}\n")
    with pytest.raises(ValueError):
        pipeline.evaluate_soft(final, gold, tmp_path / "tampered")
    assert calls == [gold]  # Tampering fails before opening gold again.
