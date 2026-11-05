"""Verify test1 evaluation boundaries with synthetic predictions, without fitting."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpt_classifier.data import LABELS, Example, Text


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "scripts/svm/evaluate-test1.py"
    spec = importlib.util.spec_from_file_location("svm_test1", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"synthetic artifact")
    artifact.with_suffix(".json").write_text("{}")
    args = SimpleNamespace(artifact=artifact, input=tmp_path / "input.xlsx",
                           gold=tmp_path / "gold.xlsx", output_dir=tmp_path / "output", threads=2)
    records = [Text(i + 2, f"response {i}") for i in range(900)]
    gold = [Example(row.excel_row, row.text, LABELS[i % 3]) for i, row in enumerate(records)]
    labels = [row.label for row in gold]
    model = SimpleNamespace(predict=lambda texts: labels)
    monkeypatch.setattr(module, "load_model", lambda path: (model, {"dataset": {"rows": 20092}}))

    def read(path, labeled=False):
        if labeled:
            assert (args.output_dir / "predictions.jsonl").exists()
            run = json.loads((args.output_dir / "run.json").read_text())
            assert run["gold_accessed"] is False
            assert run["phase"] == "prediction"
        return (gold if labeled else records), {"sha256": "synthetic"}

    monkeypatch.setattr(module, "read_xlsx", read)
    return module, args, labels, gold


def test_predictions_are_saved_before_gold_access(evaluation):
    module, args, _, _ = evaluation
    scores = module.evaluate(args)
    assert scores["correct"] == 900
    assert scores["accuracy"] == 1
    assert scores["precision_macro"] == 1
    assert scores["recall_macro"] == 1
    run = json.loads((args.output_dir / "run.json").read_text())
    assert run["device"] == "cpu"
    assert run["training_rows"] == 20092
    with pytest.raises(FileExistsError):
        module.evaluate(args)


def test_wrong_prediction_count_is_rejected(evaluation):
    module, args, labels, _ = evaluation
    labels.pop()
    with pytest.raises(ValueError, match="prediction count"):
        module.evaluate(args)
    assert not (args.output_dir / "predictions.jsonl").exists()


def test_misaligned_gold_preserves_text_only_predictions(evaluation):
    module, args, _, gold = evaluation
    gold[0] = Example(2, "changed text", "c1")
    with pytest.raises(ValueError, match="Gold row IDs or texts differ"):
        module.evaluate(args)
    assert (args.output_dir / "predictions.jsonl").exists()
    assert not (args.output_dir / "metrics.json").exists()
