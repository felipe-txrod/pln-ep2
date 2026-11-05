from __future__ import annotations

import numpy as np
import pytest

from ensemble_classifier import stacking_pipeline as pipeline
from ensemble_classifier.soft_voting import MODELS, apply_calibration, soft_vote
from ensemble_classifier.stacking import (CANDIDATES, FEATURES, feature_matrix, fit_meta,
                                          meta_probabilities, select_regularization)
from gpt_classifier.data import LABELS, Example, Text, canonical, digest, file_digest, read_xlsx, write_json, write_jsonl


def sample(n):
    rng = np.random.default_rng(17)
    return {"svm": rng.normal(size=(n, 3)), "rl": rng.dirichlet([2, 2, 2], n),
            "norberto": rng.dirichlet([2, 2, 2], n)}


def test_feature_order_and_portable_multinomial_export():
    p = {m: np.tile([.1 + i * .1, .3, .6 - i * .1], (90, 1)) for i, m in enumerate(MODELS)}
    assert feature_matrix(dict(reversed(list(p.items()))))[0].tolist() == pytest.approx([.1, .3, .6, .2, .3, .5, .3, .3, .4])
    y, groups = np.asarray(LABELS * 30), np.arange(90)
    model, search, folds, oof = fit_meta(p, y, groups)
    assert model["features"] == list(FEATURES) and len(search["trials"]) == len(CANDIDATES)
    assert meta_probabilities(p, model).shape == oof.shape == (90, 3)
    assert np.allclose(meta_probabilities(p, model).sum(axis=1), 1)
    assert set(folds) == set(range(5))
    with pytest.raises(ValueError):
        meta_probabilities(p, dict(model, features=list(reversed(FEATURES))))
    with pytest.raises(ValueError):
        feature_matrix(dict(p, svm=np.ones((90, 3))))
    with pytest.raises(ValueError):
        feature_matrix(dict(p, svm=p["svm"][:-1]))


def test_one_standard_error_prefers_stronger_regularization():
    trials = [{"C": .01, "mean_f1_macro": .45, "standard_error_f1_macro": .01},
              {"C": .1, "mean_f1_macro": .49, "standard_error_f1_macro": .02},
              {"C": 1., "mean_f1_macro": .50, "standard_error_f1_macro": .03},
              {"C": 10., "mean_f1_macro": .495, "standard_error_f1_macro": .01}]
    selected, threshold = select_regularization(trials)
    assert selected == .1 and threshold == pytest.approx(.47)


def test_meta_cv_never_splits_similarity_groups():
    n = 180
    raw = sample(n)
    p = dict(raw, svm=raw["rl"])
    groups = np.arange(n) // 2
    y = np.asarray([LABELS[(i // 2) % 3] for i in range(n)])
    _, search, fold_ids, _ = fit_meta(p, y, groups)
    for group in set(groups):
        assert len(set(fold_ids[groups == group])) == 1
    assert sum(f["validation_rows"] for f in search["folds"]) == n
    assert all(f["shared_groups"] == 0 for f in search["folds"])


@pytest.fixture
def experiment(tmp_path, monkeypatch, make_xlsx):
    records = [Example(i + 2, f"holdout {i}", LABELS[(i // 2) % 3]) for i in range(180)]
    groups = np.arange(180) // 2
    train = make_xlsx("train.xlsx", [(r.text, r.label) for r in records])
    _, source = read_xlsx(train)
    identities = {m: {"artifact_sha256": digest(m)} for m in MODELS}
    calibrators = {"svm": {"method": "sigmoid_ovr", "a": [-1., -1., -1.], "b": [0., 0., 0.]},
                   "rl": {"method": "identity"}, "norberto": {"method": "identity"}}
    calibration, meta = pipeline.separate_holdout([r.label for r in records], groups, 42)
    partition_path = tmp_path / "soft-partition.jsonl"
    write_jsonl(partition_path, [{"excel_row": r.excel_row, "text_sha256": digest(r.text), "group_id": str(groups[i]),
                                "role": "calibration" if i in calibration else "weight_selection"} for i, r in enumerate(records)])
    soft_path = tmp_path / "soft-model.json"
    write_json(soft_path, {"fixture": True})
    soft = {"score_configuration": {"seed": 42, "train": str(train)}, "partition_sha256": file_digest(partition_path),
            "model_identities": identities, "calibration": calibrators, "weights": {"svm": .75, "rl": 0., "norberto": .25}}
    monkeypatch.setattr(pipeline, "load_soft_artifact", lambda _: soft)
    monkeypatch.setattr(pipeline, "holdout_records", lambda _: (records, groups, source, records, {}))
    monkeypatch.setattr(pipeline, "verify_lr_training", lambda *a: None)
    monkeypatch.setattr(pipeline, "model_identities", lambda: identities)

    def cache(directory, targets, source):
        raw = sample(len(targets))
        for m in MODELS:
            root = directory / m
            root.mkdir(parents=True)
            field = "decision_scores" if m == "svm" else "probabilities"
            write_jsonl(root / "predictions.jsonl", [{"excel_row": t.excel_row, "text_sha256": digest(t.text),
                "label": LABELS[int(p.argmax())], field: dict(zip(LABELS, p.tolist()))} for t, p in zip(targets, raw[m])])
            write_json(root / "run.json", {"status": "completed", "predictions_sha256": file_digest(root / "predictions.jsonl")})
        return raw

    def collect(config, targets, source, identities, directory, **kwargs):
        assert all(type(t) is Text for t in targets)
        return sample(len(targets)), {}
    monkeypatch.setattr(pipeline, "collect_scores", collect)
    score_dir = tmp_path / "scores"
    cache(score_dir, records, source)
    config_path = tmp_path / "stacking.json"
    write_json(config_path, {"soft_model": str(soft_path), "soft_partition": str(partition_path), "holdout_scores": str(score_dir)})
    return {"config": config_path, "output": tmp_path / "fit", "calibration": calibration, "meta": meta,
            "records": records, "raw": sample(180), "cache": cache, "soft": soft, "soft_path": soft_path,
            "groups": groups}


def test_fit_uses_only_noncalibration_rows_and_preserves_sources(experiment, monkeypatch):
    e = experiment
    original = pipeline.fit_meta
    def guarded(matrices, labels, groups):
        assert np.array_equal(groups, e["groups"][e["meta"]])
        assert not set(groups) & set(e["groups"][e["calibration"]])
        for m in MODELS:
            assert np.array_equal(matrices[m], apply_calibration(e["raw"][m][e["meta"]], e["soft"]["calibration"][m]))
        return original(matrices, labels, groups)
    monkeypatch.setattr(pipeline, "fit_meta", guarded)
    before = file_digest(e["soft_path"])
    result = pipeline.fit_stacking(e["config"], e["output"])
    assert result["test1_accessed"] is result["base_training_performed"] is result["calibration_fitted"] is False
    assert result["partition"]["meta_training_rows"] == len(e["meta"])
    assert file_digest(e["soft_path"]) == before
    pipeline.load_stacking_artifact(e["output"] / "model.json")
    with pytest.raises(FileExistsError):
        pipeline.fit_stacking(e["config"], e["output"])


def test_predict_evaluate_read_gold_late_no_refitting_and_reject_tampering(experiment, make_xlsx, tmp_path, monkeypatch):
    e = experiment
    pipeline.fit_stacking(e["config"], e["output"])
    source = make_xlsx("test.xlsx", [(f"test {i}", None) for i in range(900)])
    gold = make_xlsx("gold.xlsx", [(f"test {i}", LABELS[i % 3]) for i in range(900)])
    targets, info = read_xlsx(source)
    soft_run = tmp_path / "soft-run"
    raw = e["cache"](soft_run / "members", targets, info)
    calibrated = {m: apply_calibration(raw[m], e["soft"]["calibration"][m]) for m in MODELS}
    for m in MODELS:
        root = soft_run / "calibrated" / m
        root.mkdir(parents=True)
        write_jsonl(root / "predictions.jsonl", pipeline.prediction_rows(targets, calibrated[m]))
        write_json(root / "run.json", {"predictions_sha256": file_digest(root / "predictions.jsonl")})
    write_jsonl(soft_run / "predictions.jsonl", pipeline.prediction_rows(targets, soft_vote(calibrated, e["soft"]["weights"])))
    write_json(soft_run / "run.json", {"status": "completed", "rows": 900, "labels": list(LABELS), "input": info,
        "artifact_sha256": file_digest(e["soft_path"]), "weights": e["soft"]["weights"], "gold_accessed": False,
        "base_training_performed": False, "calibration_fitted": False,
        "targets_sha256": digest(canonical([(r.excel_row, digest(r.text)) for r in targets])),
        "predictions_sha256": file_digest(soft_run / "predictions.jsonl")})
    monkeypatch.setattr(pipeline, "fit_meta", lambda *a, **k: pytest.fail("Meta refit on test1"))
    final = tmp_path / "final"
    seen = []
    original_reader = pipeline.read_xlsx
    def guarded(path, **kwargs):
        if path == gold:
            assert (final / "predictions.jsonl").is_file()
            assert kwargs.get("labeled") is True
            seen.append(path)
        return original_reader(path, **kwargs)
    monkeypatch.setattr(pipeline, "read_xlsx", guarded)
    before = file_digest(e["output"] / "model.json")
    prediction = pipeline.predict_stacking(e["output"] / "model.json", source, final, soft_run=soft_run)
    assert not seen and prediction["gold_accessed"] is False
    result = pipeline.evaluate_stacking(final, gold, tmp_path / "evaluation")
    assert seen == [gold] and result["parameters_updated"] is False
    assert set(result["metrics"]) == {"stacking", "soft_voting", *MODELS, *(m + "_calibrated" for m in MODELS)}
    paired = result["paired_stacking_vs"]["soft_voting"]
    assert result["changes_vs_soft"]["corrected"] == paired["only_ep2_correct"]
    assert result["changes_vs_soft"]["harmed"] == paired["only_ep1_correct"]
    assert file_digest(e["output"] / "model.json") == before
    with pytest.raises(FileExistsError):
        pipeline.evaluate_stacking(final, gold, tmp_path / "evaluation")
    (soft_run / "calibrated/svm/predictions.jsonl").write_text("{}\n")
    with pytest.raises(ValueError):
        pipeline.evaluate_stacking(final, gold, tmp_path / "tampered")
    assert seen == [gold]


def test_missing_cache_never_launches_inference(experiment, monkeypatch, tmp_path):
    e = experiment
    monkeypatch.setattr(pipeline, "collect_scores", lambda *a, **k: pytest.fail("Unexpected base inference"))
    with pytest.raises(FileNotFoundError):
        pipeline.cached_scores({}, [], {}, {}, tmp_path / "missing", "holdout")
