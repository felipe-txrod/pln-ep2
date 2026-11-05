"""Predict with the unchanged EP1 delivery adapter and score saved test1 predictions."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

from gpt_classifier.data import LABELS, digest, file_digest, read_jsonl, read_xlsx, write_json, write_jsonl
from gpt_classifier.evaluation import align, metrics
from norberto_classifier import NorbertoClassifier
from norberto_classifier.artifacts import ADAPTER_SHA256

ROOT = Path(__file__).resolve().parents[2]


def compare_saved(args: argparse.Namespace) -> dict[str, Any]:
    """Score persisted predictions on CPU without importing model libraries."""
    metadata_path = args.prediction_dir / "run.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("status") != "completed" or metadata.get("phase") != "prediction"
            or metadata.get("rows") != 900 or metadata.get("gold_accessed") is not False
            or metadata.get("labels") != list(LABELS)
            or metadata.get("adapter_weights_sha256") != ADAPTER_SHA256
            or metadata.get("artifact_manifest_sha256") != file_digest(
                ROOT / "src/norberto_classifier/frozen_manifest.json")):
        raise ValueError("A completed text-only prediction run from the original EP1 adapter is required.")
    prediction_path = args.prediction_dir / "predictions.jsonl"
    if file_digest(prediction_path) != metadata["predictions_sha256"]:
        raise ValueError("The saved predictions changed after inference.")
    records, info = read_xlsx(args.input)
    if len(records) != 900 or info["sha256"] != metadata["input"]["sha256"]:
        raise ValueError("The test1 source differs from the prediction input.")
    predictions = read_jsonl(prediction_path)
    align(records, predictions)
    gold, gold_info = read_xlsx(args.gold, labeled=True)
    if len(gold) != 900 or any((a.excel_row, a.text) != (b.excel_row, b.text) for a, b in zip(records, gold)):
        raise ValueError("Gold row IDs or texts differ from the inference source.")
    scores = metrics(gold, predictions)
    scores["precision_macro"] = scores["classification_report"]["macro avg"]["precision"]
    scores["recall_macro"] = scores["classification_report"]["macro avg"]["recall"]
    scores["correct"] = sum(row["label"] == truth.label for row, truth in zip(predictions, gold))
    scores["incorrect"] = 900 - scores["correct"]
    result = {**metadata, "phase": "comparison", "training_performed": False,
              "evaluation_performed": True, "inference_performed": False, "gold_accessed": True,
              "comparison_device": "cpu",
              "gold_read_after_predictions": True, "gold_sha256": gold_info["sha256"],
              "prediction_run_sha256": file_digest(metadata_path),
              "comparison_script_sha256": file_digest(Path(__file__)), "metrics": scores}
    if (file_digest(args.input) != info["sha256"] or file_digest(args.gold) != gold_info["sha256"]
            or file_digest(prediction_path) != metadata["predictions_sha256"]):
        raise ValueError("An evaluation input changed during scoring.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "metrics.json", scores)
    write_json(args.output_dir / "run.json", result)
    return result


def predict_original(args: argparse.Namespace) -> dict[str, Any]:
    """Use the immutable EP1 inference package; no training implementation is loaded."""
    if args.output_dir.exists():
        raise FileExistsError(f"Use a new output directory: {args.output_dir}")
    records, info = read_xlsx(args.input)
    if len(records) != 900:
        raise ValueError("Test1 requires all 900 original rows.")
    started = time.monotonic()
    classifier = NorbertoClassifier(args.artifact_dir, device=args.device)
    values = classifier.predict([row.text for row in records])
    predictions = [{"excel_row": row.excel_row, "text_sha256": digest(row.text), **asdict(value)}
                   for row, value in zip(records, values)]
    align(records, predictions)
    if file_digest(args.input) != info["sha256"]:
        raise ValueError("The test1 input changed during inference.")
    if classifier.bundle is None:
        raise RuntimeError("The original EP1 model was not loaded.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prediction_path = args.output_dir / "predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    result = {"status": "completed", "phase": "prediction", "model_type": "norberto-ep1",
              "rows": len(records), "labels": list(LABELS), "input": info,
              "gold_accessed": False, "training_performed": False, "evaluation_performed": False,
              "device": args.device, "adapter_weights_sha256": ADAPTER_SHA256,
              "artifact_manifest_sha256": classifier.bundle.identity_sha256,
              "source_ep1_metadata_sha256": file_digest(classifier.bundle.root / "run.json"),
              "config": classifier.bundle.config, "source_training_config": classifier.bundle.metadata["config"],
              "tokenization": classifier.tokenization_statistics,
              "predictions_sha256": file_digest(prediction_path),
              "inference_seconds": time.monotonic() - started,
              "prediction_script_sha256": file_digest(Path(__file__))}
    write_json(args.output_dir / "run.json", result)
    return result


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    return compare_saved(args) if args.stage == "compare" else predict_original(args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("predict", "compare"), required=True)
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--input", type=Path, default=ROOT / "data/test1.xlsx")
    parser.add_argument("--gold", type=Path, default=ROOT / "data/test1-rotulado-para-teste-estatistico.xlsx")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "compare" and args.prediction_dir is None:
        parser.error("--prediction-dir is required for comparison.")
    try:
        result = evaluate(args)
    except (ValueError, OSError, RuntimeError, ImportError, TypeError, KeyError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
