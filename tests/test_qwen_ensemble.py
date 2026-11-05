"""Verify four-member probabilities, fixed selection, cache provenance, and row order."""
from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import pytest

from ensemble_classifier.soft_voting import (QWEN_MODELS, apply_calibration, choose_weights,
                                              fit_calibrations, soft_vote)
from ensemble_classifier.stacking import feature_matrix, feature_names, fit_meta, meta_probabilities
from gpt_classifier.data import LABELS, Text, digest, file_digest, write_json, write_jsonl


def sample(n=180):
    rng = np.random.default_rng(13)
    y = np.asarray([LABELS[index % 3] for index in range(n)])
    raw = {name: rng.dirichlet([2, 2, 2], n) for name in QWEN_MODELS}
    raw["svm"] = rng.normal(size=(n, 3))
    return raw, y


def test_four_member_calibration_and_unique_quarter_simplex_grid():
    raw, labels = sample()
    calibration = fit_calibrations(raw, labels, np.arange(len(labels)))
    assert set(calibration) == set(QWEN_MODELS)
    calibrated = {name: apply_calibration(raw[name], calibration[name]) for name in QWEN_MODELS}
    weights, trials = choose_weights(calibrated, labels)
    assert len(trials) == len({tuple(row["weights"].values()) for row in trials}) == 35
    assert set(weights) == set(QWEN_MODELS) and sum(weights.values()) == 1
    assert any(row["weights"] == dict.fromkeys(QWEN_MODELS, .25) for row in trials)
    assert np.allclose(soft_vote(calibrated, weights).sum(axis=1), 1)
    assert calibration["qwen"]["calibration_only_check"]["folds"] == 3
    with pytest.raises(ValueError):
        soft_vote(calibrated, {name: 1 / 3 for name in QWEN_MODELS[:-1]})


def test_stacking_exports_twelve_features_and_requires_every_member():
    raw, labels = sample()
    raw["svm"] = raw["rl"].copy()
    model, search, folds, oof = fit_meta(raw, labels, np.arange(len(labels)))
    assert len(feature_names(QWEN_MODELS)) == feature_matrix(raw).shape[1] == 12
    assert model["features"][-3:] == ["qwen:c1", "qwen:c234", "qwen:c5"]
    assert np.asarray(model["coef"]).shape == (3, 12)
    assert meta_probabilities(raw, model).shape == oof.shape == (180, 3)
    assert len(search["trials"]) == 4 and len(set(folds)) == 5
    with pytest.raises(ValueError):
        meta_probabilities({k: v for k, v in raw.items() if k != "qwen"}, model)


def test_recorded_qwen_rejects_wrong_artifact_and_reordered_rows(tmp_path):
    from ensemble_classifier.soft_sources import recorded_qwen

    directory = tmp_path / "scores"
    directory.mkdir()
    records = [Text(2, "first"), Text(3, "second")]
    write_jsonl(directory / "predictions.jsonl", [
        {"excel_row": r.excel_row, "text_sha256": digest(r.text), "label": "c5",
         "probabilities": {"c1": .1, "c234": .2, "c5": .7}} for r in records])
    write_json(directory / "run.json", {"status": "completed", "input": {"sha256": "source"},
               "labels": list(LABELS), "adapter_sha256": "selected", "gold_accessed": False,
               "training_performed": False, "predictions_sha256": file_digest(directory / "predictions.jsonl")})
    specification = {"run": str(directory), "run_sha256": file_digest(directory / "run.json"),
                     "predictions_sha256": file_digest(directory / "predictions.jsonl")}
    config = {"recorded_qwen": {"holdout": specification}}
    identities = {"qwen": {"artifact_sha256": "selected"}}
    p, _ = recorded_qwen(config, records, {"sha256": "source"}, identities, "holdout")
    assert p.shape == (2, 3)
    with pytest.raises(ValueError, match="another model"):
        recorded_qwen(config, records, {"sha256": "source"}, {"qwen": {"artifact_sha256": "old"}}, "holdout")
    with pytest.raises(ValueError, match="alignment"):
        recorded_qwen(config, records[::-1], {"sha256": "source"}, identities, "holdout")


def test_batched_qwen_inference_preserves_all_source_ids_including_tail(tmp_path, monkeypatch):
    import torch
    import transformers
    import qwen_classifier.inference as inference
    from qwen_classifier.config import Config

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    write_json(adapter / "experiment.json", {"labels": list(LABELS), "config": Config(mode="fp16").to_dict()})

    class Tokenizer:
        eos_token_id = 0
        def encode(self, text, add_special_tokens=False):
            return [int(text)] * int(text)

    batches = []
    class Model:
        def eval(self):
            return self
        def __call__(self, input_ids, **kwargs):
            from types import SimpleNamespace
            batches.append(len(input_ids))
            marker = input_ids[:, 0].float()
            return SimpleNamespace(logits=torch.stack([marker, marker * 0, marker * 0 - 1], dim=-1))

    def collate(rows):
        return {"input_ids": torch.tensor([[row["input_ids"][0]] for row in rows])}

    monkeypatch.setattr(inference, "tokenizer_for", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(inference, "build_model", lambda *a, **k: Model())
    monkeypatch.setattr(transformers, "DataCollatorWithPadding", lambda *a, **k: collate)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())
    records = [Text(index + 2, str(index + 1)) for index in range(7)]
    rows = inference.predict_records(adapter, tmp_path, records, batch_size=3)
    assert batches == [3, 3, 1]
    assert [row["excel_row"] for row in rows] == [r.excel_row for r in records]
    assert [row["logits"][0] for row in rows] == list(range(1, 8))
    assert [row["text_sha256"] for row in rows] == [digest(r.text) for r in records]


def test_original_three_member_artifacts_remain_loadable():
    from ensemble_classifier.soft_pipeline import load_soft_artifact
    from ensemble_classifier.stacking_pipeline import load_stacking_artifact

    root = Path(__file__).resolve().parents[1]
    soft = load_soft_artifact(root / "models/ensemble-soft/model.json")
    stacking, reference = load_stacking_artifact(root / "models/ensemble-stacking/model.json")
    assert soft["models"] == stacking["models"] == ["svm", "rl", "norberto"]
    assert reference["weights"] == soft["weights"]


def test_four_member_soft_and_stacking_reuse_disjoint_fit_roles(tmp_path, monkeypatch, make_xlsx):
    from ensemble_classifier import soft_pipeline as soft
    from ensemble_classifier import stacking_pipeline as stacking
    from gpt_classifier.data import Example, read_xlsx

    raw, labels = sample()
    records = [Example(index + 2, f"response {index}", label) for index, label in enumerate(labels)]
    groups = np.arange(len(records))
    train = make_xlsx("training.xlsx", [(row.text, row.label) for row in records])
    _, info = read_xlsx(train)
    identities = {name: {"artifact_sha256": digest(name)} for name in QWEN_MODELS}
    config = {"seed": 42, "train": str(train), "split": "fixture", "models": list(QWEN_MODELS), "recorded_norberto": {}}
    monkeypatch.setattr(soft, "load_soft_config", lambda path: config)
    for module in (soft, stacking):
        monkeypatch.setattr(module, "holdout_records", lambda config: (records, groups, info, records, {}))
        monkeypatch.setattr(module, "verify_lr_training", lambda *args: None)
        monkeypatch.setattr(module, "model_identities", lambda *args: identities)

    def collect(config, targets, source, model_ids, directory, **kwargs):
        assert all(type(target) is Text for target in targets)
        for name in QWEN_MODELS:
            destination = directory / name
            destination.mkdir(parents=True, exist_ok=True)
            field = "decision_scores" if name == "svm" else "probabilities"
            write_jsonl(destination / "predictions.jsonl", [
                {"excel_row": target.excel_row, "text_sha256": digest(target.text),
                 "label": LABELS[int(values.argmax())], field: dict(zip(LABELS, values.tolist()))}
                for target, values in zip(targets, raw[name])])
            write_json(destination / "run.json", {"status": "completed"})
        return raw, {}

    monkeypatch.setattr(soft, "collect_scores", collect)
    soft_output = tmp_path / "soft-fit"
    fitted = soft.fit_soft(tmp_path / "unused", soft_output)
    assert fitted["fixed_candidate_count"] == 35 and set(fitted["weights"]) == set(QWEN_MODELS)
    monkeypatch.setattr(stacking, "collect_scores", lambda *args, **kwargs: (raw, {}))
    config_path = tmp_path / "stacking.json"
    write_json(config_path, {"soft_model": str(soft_output / "model.json"),
               "soft_partition": str(soft_output / "partition.jsonl"), "holdout_scores": str(soft_output / "scores")})
    result = stacking.fit_stacking(config_path, tmp_path / "stack-fit")
    artifact, reference = stacking.load_stacking_artifact(tmp_path / "stack-fit/model.json")
    assert artifact["models"] == reference["models"] == list(QWEN_MODELS)
    assert len(artifact["features"]) == 12
    assert result["partition"]["shared_rows"] == result["partition"]["shared_groups"] == 0
    assert result["partition"]["meta_training_rows"] == fitted["partition"]["weight_selection_rows"]
