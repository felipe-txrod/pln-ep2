"""Reject full-corpus ensemble artifacts and keep reserved rows outside logistic fitting."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ensemble_classifier.config import EnsembleConfig
from ensemble_classifier.files import member_run
from gpt_classifier.data import LABELS, Example, Text, digest, file_digest, write_json, write_jsonl
from rl_classifier import RLClassifier

ROOT = Path(__file__).resolve().parents[1]


def training_module():
    path = ROOT / "scripts/rl/train-development.py"
    spec = importlib.util.spec_from_file_location("rl_development_training", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partition_rejects_group_leakage_and_reordered_rows():
    module = training_module()
    records = [Example(i + 2, f"response {i}", LABELS[i % 3]) for i in range(6)]
    split = [{"excel_row": str(row.excel_row), "group_id": str(i),
              "role": "development" if i < 3 else "holdout"} for i, row in enumerate(records)]
    selected, report = module.select_development(records, split, expected_development=3, expected_holdout=3)
    assert selected == records[:3] and report["holdout_used_for_fit"] is False
    with pytest.raises(ValueError, match="in order"):
        module.select_development(records, split[::-1], expected_development=3, expected_holdout=3)
    split[-1]["group_id"] = split[0]["group_id"]
    with pytest.raises(ValueError, match="overlap"):
        module.select_development(records, split, expected_development=3, expected_holdout=3)


def test_rl_does_not_fall_back_to_full_corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_BUNDLE_DIR", str(tmp_path / "missing-development"))
    with pytest.raises(FileNotFoundError, match="not selected automatically"):
        RLClassifier()


def test_ensemble_rejects_predictions_from_another_artifact(tmp_path):
    records = [Text(2, "response")]
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    write_json(prediction_dir / "run.json", {"status": "completed", "input": {"sha256": "source"},
               "artifact_sha256": "a" * 64, "labels": list(LABELS)})
    write_jsonl(prediction_dir / "predictions.jsonl", [
        {"excel_row": 2, "text_sha256": digest("response"), "label": "c1"}])
    manifest = tmp_path / "frozen_manifest.json"
    write_json(manifest, {"artifact_sha256": "b" * 64})
    specification = {"run": str(prediction_dir), "identity_field": "artifact_sha256", "manifest": str(manifest)}
    with pytest.raises(ValueError, match="selected model manifest"):
        member_run(specification, records, "source")
    write_json(manifest, {"artifact_sha256": "a" * 64})
    rows, provenance = member_run(specification, records, "source")
    assert len(rows) == 1
    assert provenance["identity"]["selected_manifest_sha256"] == file_digest(manifest)


def test_default_ensemble_variants_require_development_manifests():
    for filename in ("ensemble.json", "ensemble-test1.json"):
        active = EnsembleConfig.load(ROOT / "configs" / filename)
        assert active.models == ("norberto", "svm", "rl")
        assert active.strategy == "majority" and active.tiebreaker == "rl"
        assert "qwen" not in active.members and active.gpt_config is None
        config = EnsembleConfig.load(ROOT / "configs" / filename, models=["svm", "rl"], tiebreaker="svm")
        assert config.members["rl"]["manifest"] == "models/rl-development/frozen_manifest.json"
        assert config.members["svm"]["manifest"] == "src/svm_classifier/frozen_manifest.json"
        assert "development" in config.members["rl"]["run"]


def test_synthetic_fit_excludes_holdout_vocabulary_and_exports_loadable_bundle(tmp_path, monkeypatch):
    from openpyxl import Workbook
    from lexical_classifier.inference import FrozenLexicalClassifier

    module = training_module()
    ep1 = tmp_path / "ep1"
    (ep1 / "data").mkdir(parents=True)
    (ep1 / "src").symlink_to(ROOT.parent / "EP1/src", target_is_directory=True)
    source = ep1 / "data/train.xlsx"
    workbook = Workbook()
    workbook.active.append(["resp_text", "clarity"])
    for label in LABELS:
        for _ in range(3):
            workbook.active.append([f"shared request content distinctive{label}", label])
    for label in LABELS:
        workbook.active.append(["reservedholdoutmarker exclusively reserved", label])
    workbook.save(source)
    workbook.close()
    split_path = ep1 / "results/research/dataset-hypotheses-20260923/splits.csv"
    split_path.parent.mkdir(parents=True)
    with split_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["excel_row", "group_id", "role"])
        writer.writeheader()
        writer.writerows({"excel_row": i + 2, "group_id": str(i),
                         "role": "development" if i < 9 else "holdout"} for i in range(12))
    monkeypatch.setattr(module, "DATASET_SHA256", file_digest(source))
    monkeypatch.setattr(module, "SPLIT_SHA256", file_digest(split_path))
    select = module.select_development
    monkeypatch.setattr(module, "select_development", lambda records, split: select(
        records, split, expected_development=9, expected_holdout=3))
    args = SimpleNamespace(ep1_root=ep1, output_dir=tmp_path / "model", dry_run=False)
    report = module.run(args)
    model = FrozenLexicalClassifier(args.output_dir / "frozen_manifest.json", bundle_dir=args.output_dir).load()
    vocabulary = model._model.named_steps["features"].transformer_list[0][1].vocabulary_
    assert not any("reservedholdoutmarker" in term for term in vocabulary)
    assert report["training_rows"] == 9 and report["reserved_holdout_rows"] == 3
    assert report["holdout_evaluated"] is False
    assert len((args.output_dir / "training-rows.jsonl").read_text().splitlines()) == 9
    monkeypatch.setenv("RL_BUNDLE_DIR", str(args.output_dir))
    assert len(RLClassifier().predict(["shared request content distinctivec1"])) == 1
