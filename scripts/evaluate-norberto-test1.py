"""Evaluate a completed full-corpus NorBERTo adapter on the original test1 rows."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from gpt_classifier.data import LABELS, canonical, digest, file_digest, read_jsonl, read_xlsx, write_json, write_jsonl
from gpt_classifier.evaluation import align, metrics

ROOT = Path(__file__).resolve().parents[1]


def verify_training(root: Path, ep1: Path) -> dict[str, Any]:
    report = json.loads((root / "run.json").read_text(encoding="utf-8"))
    original = json.loads((ROOT / "models/norberto/run.json").read_text(encoding="utf-8"))
    training = report["training"]
    if (report.get("status") != "completed" or report.get("mode") != "full-training"
            or report.get("training_rows") != 20092 or report.get("labels") != list(LABELS)
            or report.get("holdout_excluded") is not False or report.get("test_data_used") is not False
            or report.get("validation_used") is not False):
        raise ValueError("A completed full-corpus NorBERTo training run is required.")
    if (report["config"] != original["config"] or report["dataset_sha256"] != original["dataset_sha256"]
            or not math.isclose(training["epochs_executed"], 2) or training["training_stopped_by_budget"]
            or training["training_rows"] != 20092 or not math.isfinite(training["training_loss"])):
        raise ValueError("Training data, parameters, epochs, or completion checks differ from the fixed recipe.")
    if file_digest(ep1 / "src/clarity/finetune.py") != report["ep1_trainer_sha256"]:
        raise ValueError("The original trainer implementation changed since full training.")
    records, info = read_xlsx(ep1 / "data/train.xlsx", labeled=True)
    saved = read_jsonl(root / "training-rows.jsonl")
    if info["sha256"] != report["dataset_sha256"] or len(records) != len(saved) or len(saved) != 20092:
        raise ValueError("Full training-row coverage cannot be verified.")
    for record, row in zip(records, saved):
        if row != {"excel_row": record.excel_row, "text_sha256": digest(record.text), "label": record.label}:
            raise ValueError(f"The full training row differs at Excel row {record.excel_row}.")
    for name, expected in report["adapter_files_sha256"].items():
        path = root / "final_model" / name
        if not path.resolve().is_relative_to((root / "final_model").resolve()) or file_digest(path) != expected:
            raise ValueError(f"The trained adapter file changed: {name}")
    states = [json.loads(path.read_text()) for path in (root / "checkpoints").glob("checkpoint-*/trainer_state.json")]
    final = max(states, key=lambda state: state["global_step"])
    if final["global_step"] != final["max_steps"] or not math.isclose(final["epoch"], 2):
        raise ValueError("The trainer did not reach its configured final update.")
    return report


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    training = verify_training(args.training_run, args.ep1_root)
    adapter = args.training_run / "final_model"
    source_digest = file_digest(args.training_run / "run.json")
    identity = digest(canonical({"files": training["adapter_files_sha256"], "config": training["config"]}))
    records, info = read_xlsx(args.input)
    if len(records) != 900:
        raise ValueError("Test1 evaluation requires all 900 original rows.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "run.json", {"status": "running", "training_performed": False})
    try:
        sys.path.insert(0, str(args.ep1_root / "src"))
        from clarity import finetune as original
        import numpy as np
        import torch

        for package, expected in (("transformers", "4.57.1"), ("peft", "0.17.1"), ("accelerate", "1.10.1")):
            if importlib.metadata.version(package) != expected:
                raise RuntimeError(f"Use the original runtime: {package}=={expected}.")
        config = training["config"]
        use_cpu, device = original.select_execution_device("gpu")
        torch.set_num_threads(config["threads"])
        print("Verified all 20,092 training rows and exported adapter hashes. Loading the new adapter.", flush=True)
        started = time.monotonic()
        tokenizer = original.AutoTokenizer.from_pretrained(adapter, local_files_only=True)
        features, statistics = original.tokenize_corpus(
            np.asarray([record.text for record in records]), tokenizer, config,
            cache_dir=args.output_dir / "token-cache")
        model = original.make_model(config, config["seed"], adapter_path=adapter, allow_download=False)
        model.requires_grad_(False)
        trainer = original.Trainer(
            model=model, args=original.TrainingArguments(
                output_dir=str(args.output_dir / "inference"),
                per_device_eval_batch_size=config["eval_batch_size"],
                fp16=False, use_cpu=use_cpu, report_to=[], disable_tqdm=True,
                dataloader_num_workers=0, eval_accumulation_steps=32),
            processing_class=tokenizer,
            data_collator=original.DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))
        torch.cuda.reset_peak_memory_stats()
        print("Predicting 900 unlabeled test1 responses in batches of 8.", flush=True)
        raw = trainer.predict(original.TokenizedRows(features, None, np.arange(len(features))))
        logits = np.asarray(raw.predictions)
        if logits.shape != (900, 3) or not np.isfinite(logits).all():
            raise FloatingPointError("NorBERTo returned invalid or nonfinite logits.")
        probabilities = torch.softmax(torch.from_numpy(logits).float(), dim=-1).numpy()
        predictions = [{"excel_row": record.excel_row, "text_sha256": digest(record.text),
                        "label": LABELS[int(probabilities[i].argmax())],
                        "probabilities": probabilities[i].tolist(), "logits": logits[i].tolist()}
                       for i, record in enumerate(records)]
        elapsed = time.monotonic() - started
        align(records, predictions)
        write_jsonl(args.output_dir / "predictions.jsonl", predictions)
        # Gold labels enter only after text-only predictions have been persisted.
        gold, gold_info = read_xlsx(args.gold, labeled=True)
        if len(gold) != 900 or any((a.excel_row, a.text) != (b.excel_row, b.text) for a, b in zip(records, gold)):
            raise ValueError("Gold row IDs or texts differ from the inference source.")
        scores = metrics(gold, predictions)
        scores["precision_macro"] = scores["classification_report"]["macro avg"]["precision"]
        scores["recall_macro"] = scores["classification_report"]["macro avg"]["recall"]
        scores["correct"] = sum(row["label"] == truth.label for row, truth in zip(predictions, gold))
        scores["incorrect"] = 900 - scores["correct"]
        if (file_digest(args.input) != info["sha256"] or file_digest(args.gold) != gold_info["sha256"]
                or file_digest(args.training_run / "run.json") != source_digest
                or file_digest(adapter / "adapter_model.safetensors") != training["adapter_files_sha256"]["adapter_model.safetensors"]):
            raise ValueError("A model or evaluation input changed during inference.")
        result = {"status": "completed", "phase": "comparison", "model_type": "norberto-full",
                  "scope": "Isolated test1 comparison of the full-corpus refit; no test2 delivery performed.",
                  "rows": 900, "labels": list(LABELS), "input": info, "gold_sha256": gold_info["sha256"],
                  "config": config, "adapter_sha256": identity,
                  "adapter_weights_sha256": training["adapter_files_sha256"]["adapter_model.safetensors"],
                  "training_run_sha256": source_digest, "training_rows": training["training_rows"],
                  "training_rows_verified": True, "ep1_trainer_sha256": training["ep1_trainer_sha256"],
                  "evaluation_script_sha256": file_digest(Path(__file__)),
                  "predictions_sha256": file_digest(args.output_dir / "predictions.jsonl"),
                  "training_performed": False, "evaluation_performed": True, "gold_read_after_predictions": True,
                  "device": device, "runtime_versions": {name: importlib.metadata.version(name) for name in (
                      "torch", "transformers", "peft", "accelerate", "numpy", "scikit-learn", "openpyxl")},
                  "memory": {"peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                             "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20},
                  "inference_seconds": elapsed, "tokenization": statistics, "metrics": scores}
        write_json(args.output_dir / "metrics.json", scores)
        write_json(args.output_dir / "run.json", result)
        return result
    except Exception as error:
        failure = {"status": "failed", "error": f"{type(error).__name__}: {error}", "training_performed": False}
        write_json(args.output_dir / "failure.json", failure)
        write_json(args.output_dir / "run.json", failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, default=ROOT / "results/norberto-full-training")
    parser.add_argument("--ep1-root", type=Path, default=ROOT.parent / "EP1")
    parser.add_argument("--input", type=Path, default=ROOT / "data/test1.xlsx")
    parser.add_argument("--gold", type=Path, default=ROOT / "data/test1-rotulado-para-teste-estatistico.xlsx")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate(args)
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    print(json.dumps({"status": report["status"], "metrics": report["metrics"],
                      "inference_seconds": report["inference_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
