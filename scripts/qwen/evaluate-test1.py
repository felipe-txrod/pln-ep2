"""Evaluate a completed Qwen adapter on test1 without training or test2 export."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from gpt_classifier.data import LABELS, file_digest, read_xlsx, write_json, write_jsonl
from gpt_classifier.evaluation import align, metrics
from qwen_classifier.hardware import environment, isolated_probe, memory, runtime_identity
from qwen_classifier.inference import adapter_identity, predict_records
from qwen_classifier.artifacts import resolve_adapter


def verify_training(root: Path | None) -> dict[str, Any]:
    adapter = resolve_adapter() if root is None else root / "adapter"
    metadata_path = adapter.parent / "training.json" if root is None else root / "run.json"
    report = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (report.get("status") != "completed" or report.get("smoke") is not False
            or report.get("pilot") is not False or not report.get("verified_lora_update")):
        raise ValueError("A completed full training run with verified LoRA updates is required.")
    saved = json.loads((adapter / "experiment.json").read_text(encoding="utf-8"))
    if saved["config"] != report["config"] or saved["labels"] != list(LABELS):
        raise ValueError("The exported adapter configuration differs from its training run.")
    if root is None:
        # The package manifest pins the selected weights and reviewed training provenance.
        return report
    checkpoint = root / "checkpoints" / Path(report["best_checkpoint"]).name
    if file_digest(adapter / "adapter_model.safetensors") != file_digest(checkpoint / "adapter_model.safetensors"):
        raise ValueError("The exported adapter differs from the checkpoint selected in validation.")
    states = [json.loads(path.read_text(encoding="utf-8"))
              for path in (root / "checkpoints").glob("checkpoint-*/trainer_state.json")]
    final = max(states, key=lambda state: state["global_step"])
    if (final["global_step"] != final["max_steps"]
            or not math.isclose(final["epoch"], report["config"]["epochs"])):
        raise ValueError("Training did not reach its configured final epoch and step.")
    return report


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    training = verify_training(args.training_run)
    adapter = resolve_adapter() if args.training_run is None else args.training_run / "adapter"
    training_path = adapter.parent / "training.json" if args.training_run is None else args.training_run / "run.json"
    identity = adapter_identity(adapter)
    training_digest = file_digest(training_path)
    current = environment()
    if (args.training_run is not None
            and current["implementation_sha256"] != training["environment"]["implementation_sha256"]):
        raise ValueError("Qwen source code changed since training; review before evaluating.")
    records, source_info = read_xlsx(args.input)
    if len(records) != 900:
        raise ValueError("The test1 evaluation requires all 900 original rows.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "run.json", {"status": "running", "phase": "comparison",
                                             "training_performed": False})
    try:
        kind = "nf4" if training["config"]["mode"] == "qlora" else "fp16"
        check = isolated_probe(kind)
        write_json(args.output_dir / "compatibility.json", check)
        if check["status"] != "passed":
            raise RuntimeError(f"Current inference kernels failed compatibility checks: {check}")
        device = runtime_identity()
        import torch
        torch.cuda.reset_peak_memory_stats()
        print(f"Reloading the selected adapter and predicting {len(records)} unlabeled test1 rows.", flush=True)
        started = time.monotonic()
        predictions = predict_records(adapter, args.cache_dir, records)
        elapsed = time.monotonic() - started
        align(records, predictions)
        write_jsonl(args.output_dir / "predictions.jsonl", predictions)

        # Gold labels become available only after text-only predictions are persisted.
        gold, gold_info = read_xlsx(args.gold, labeled=True)
        if len(gold) != len(records) or any((row.excel_row, row.text) != (truth.excel_row, truth.text)
                                           for row, truth in zip(records, gold)):
            raise ValueError("Gold workbook row IDs or texts differ from the inference source.")
        scores = metrics(gold, predictions)
        scores["precision_macro"] = scores["classification_report"]["macro avg"]["precision"]
        scores["recall_macro"] = scores["classification_report"]["macro avg"]["recall"]
        scores["correct"] = sum(row["label"] == truth.label for row, truth in zip(predictions, gold))
        scores["incorrect"] = len(records) - scores["correct"]
        if (adapter_identity(adapter) != identity
                or file_digest(training_path) != training_digest
                or file_digest(args.input) != source_info["sha256"]
                or file_digest(args.gold) != gold_info["sha256"]):
            raise ValueError("A model or evaluation input changed during inference.")
        write_json(args.output_dir / "metrics.json", scores)
        result = {
            "status": "completed", "phase": "comparison", "dataset": "test1",
            "scope": "Isolated test1 comparison; no test2 delivery was performed by this evaluation.",
            "input": source_info, "gold_sha256": gold_info["sha256"], "rows": len(records),
            "labels": list(LABELS), "adapter_sha256": identity,
            "adapter_weights_sha256": file_digest(adapter / "adapter_model.safetensors"),
            "training_run_sha256": training_digest, "config": training["config"],
            "best_checkpoint": Path(training["best_checkpoint"]).name,
            "implementation_sha256": current["implementation_sha256"],
            "evaluation_script_sha256": file_digest(Path(__file__)),
            "predictions_sha256": file_digest(args.output_dir / "predictions.jsonl"),
            "device": device, "environment": current, "memory": memory(),
            "training_performed": False, "evaluation_performed": True,
            "gold_read_after_predictions": True, "adapter_reloaded_for_inference": True,
            "inference_seconds": elapsed, "metrics": scores,
        }
        write_json(args.output_dir / "run.json", result)
        return result
    except Exception as error:
        failure = {"status": "failed", "phase": "comparison",
                   "error": f"{type(error).__name__}: {error}", "training_performed": False}
        write_json(args.output_dir / "failure.json", failure)
        write_json(args.output_dir / "run.json", failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path,
                        help="Explicit historical run override; defaults to the verified selected models/qwen package")
    parser.add_argument("--input", type=Path, default=Path("data/test1.xlsx"))
    parser.add_argument("--gold", type=Path, default=Path("data/test1-rotulado-para-teste-estatistico.xlsx"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/qwen-model"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = evaluate(args)
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    print(json.dumps({"status": result["status"], "metrics": result["metrics"],
                      "inference_seconds": result["inference_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
