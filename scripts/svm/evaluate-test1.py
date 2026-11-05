"""Evaluate an explicitly selected, trusted EP1 SVM artifact on test1 using CPU."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from gpt_classifier.data import LABELS, digest, file_digest, read_jsonl, read_xlsx, write_json, write_jsonl
from gpt_classifier.evaluation import align, metrics
from lexical_classifier.inference import FrozenLexicalClassifier, load_legacy_artifact

ROOT = Path(__file__).resolve().parents[2]


def load_model(artifact: Path) -> tuple[Any, dict[str, Any]]:
    """Validate the fixed SVM recipe without replacing the published artifact."""
    metadata = json.loads(artifact.with_suffix(".json").read_text())
    contract = ROOT / "src/svm_classifier/frozen_manifest.json"
    validator = FrozenLexicalClassifier(contract)
    for package, expected in validator.manifest["runtime_versions"].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"The trained artifact requires {package}=={expected}.")
    bundle = load_legacy_artifact(artifact)
    if (not isinstance(bundle, dict) or bundle.get("format_version") != 1
            or tuple(bundle.get("labels", ())) != LABELS
            or bundle.get("metadata") != metadata
            or metadata.get("selected") != validator.manifest["selected"]):
        raise ValueError("The artifact and adjacent metadata do not match the SVM contract.")
    # Reuse recipe validation while keeping the published artifact hash pinned.
    validator._validate_model(bundle["model"])
    return bundle["model"], metadata


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads < 1:
        raise ValueError("threads must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    artifact_hash = file_digest(args.artifact)
    metadata_hash = file_digest(args.artifact.with_suffix(".json"))
    model, metadata = load_model(args.artifact)
    records, info = read_xlsx(args.input)
    if len(records) != 900:
        raise ValueError("test1 must contain all 900 original rows.")
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=args.threads):
        labels = model.predict([row.text for row in records])
    if len(labels) != len(records):
        raise ValueError("The model returned an incorrect prediction count.")
    predictions = [{"excel_row": row.excel_row, "text_sha256": digest(row.text), "label": str(label)}
                   for row, label in zip(records, labels)]
    align(records, predictions)
    prediction_path = args.output_dir / "predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    run = {"status": "completed", "phase": "prediction", "gold_accessed": False,
           "device": "cpu", "rows": len(records), "labels": list(LABELS),
           "artifact_sha256": artifact_hash, "metadata_sha256": metadata_hash,
           "training_rows": metadata["dataset"]["rows"], "input": info,
           "predictions_sha256": file_digest(prediction_path)}
    write_json(args.output_dir / "run.json", run)
    # Read gold only after text-only predictions have been persisted.
    gold, gold_info = read_xlsx(args.gold, labeled=True)
    if len(gold) != 900 or any((a.excel_row, a.text) != (b.excel_row, b.text)
                              for a, b in zip(records, gold)):
        raise ValueError("Gold row IDs or texts differ from the inference source.")
    scores = metrics(gold, read_jsonl(prediction_path))
    scores["correct"] = sum(p["label"] == row.label for p, row in zip(predictions, gold))
    scores["incorrect"] = len(gold) - scores["correct"]
    scores["precision_macro"] = scores["classification_report"]["macro avg"]["precision"]
    scores["recall_macro"] = scores["classification_report"]["macro avg"]["recall"]
    write_json(args.output_dir / "metrics.json", scores)
    run.update(phase="evaluation", gold_accessed=True, gold=gold_info)
    write_json(args.output_dir / "run.json", run)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True, help="Trusted local joblib artifact only.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/test1.xlsx")
    parser.add_argument("--gold", type=Path, default=ROOT / "data/test1-rotulado-para-teste-estatistico.xlsx")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate(args), indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
