"""Ensure scoring consumes frozen predictions without loading a model."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpt_classifier.data import LABELS, Example, Text, digest, file_digest, write_json, write_jsonl
from norberto_classifier.artifacts import ADAPTER_SHA256, load_bundle


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "scripts/norberto/evaluate-test1.py"
    spec = importlib.util.spec_from_file_location("norberto_test1_stages", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    input_path, gold_path = tmp_path / "test1.xlsx", tmp_path / "gold.xlsx"
    input_path.write_bytes(b"input fixture")
    gold_path.write_bytes(b"gold fixture")
    records = [Text(i + 2, f"response {i}") for i in range(900)]
    gold = [Example(r.excel_row, r.text, LABELS[i % 3]) for i, r in enumerate(records)]
    rows = [{"excel_row": r.excel_row, "text_sha256": digest(r.text), "label": truth.label}
            for r, truth in zip(records, gold)]
    write_jsonl(predictions / "predictions.jsonl", rows)
    write_json(predictions / "run.json", {
        "status": "completed", "phase": "prediction", "rows": 900, "labels": list(LABELS),
        "gold_accessed": False, "input": {"sha256": file_digest(input_path)},
        "adapter_weights_sha256": ADAPTER_SHA256,
        "artifact_manifest_sha256": file_digest(source.parents[2] / "src/norberto_classifier/frozen_manifest.json"),
        "predictions_sha256": file_digest(predictions / "predictions.jsonl")})

    def read(path, labeled=False):
        return (gold if labeled else records), {"sha256": file_digest(path)}

    def forbidden(*args, **kwargs):
        raise AssertionError("Scoring must not verify/load a training model.")

    monkeypatch.setattr(module, "read_xlsx", read)
    monkeypatch.setattr(module, "NorbertoClassifier", forbidden)
    args = SimpleNamespace(stage="compare", prediction_dir=predictions, input=input_path,
                           gold=gold_path, output_dir=tmp_path / "scores")
    return module, args, gold


def test_compare_saved_predictions_without_model_access(comparison):
    module, args, _ = comparison
    result = module.evaluate(args)
    assert result["metrics"]["correct"] == 900
    assert result["metrics"]["accuracy"] == 1
    assert result["comparison_device"] == "cpu"
    assert result["inference_performed"] is False
    assert json.loads((args.output_dir / "metrics.json").read_text())["confusion_matrix"] == [
        [300, 0, 0], [0, 300, 0], [0, 0, 300]]


def test_rejects_predictions_changed_after_inference(comparison):
    module, args, _ = comparison
    path = args.prediction_dir / "predictions.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after inference"):
        module.evaluate(args)
    assert not args.output_dir.exists()


def test_rejects_gold_with_changed_text(comparison):
    module, args, gold = comparison
    gold[0] = Example(2, "different response", "c1")
    with pytest.raises(ValueError, match="Gold row IDs or texts differ"):
        module.evaluate(args)
    assert not args.output_dir.exists()


def test_rejects_experimental_adapter_predictions(comparison):
    module, args, _ = comparison
    path = args.prediction_dir / "run.json"
    metadata = json.loads(path.read_text())
    metadata["adapter_weights_sha256"] = "f" * 64
    write_json(path, metadata)
    with pytest.raises(ValueError, match="original EP1 adapter"):
        module.evaluate(args)
    assert not args.output_dir.exists()


def test_delivery_bundle_keeps_original_configuration_and_weights():
    bundle = load_bundle()
    assert file_digest(bundle.adapter / "adapter_model.safetensors") == ADAPTER_SHA256
    assert bundle.metadata["training_rows"] == 16093
    assert bundle.metadata["config"]["epochs"] == 2
    assert bundle.config["lora_rank"] == 4
    assert bundle.config["lora_alpha"] == 8
    assert bundle.config["learning_rate"] == .001
