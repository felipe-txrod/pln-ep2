"""Evaluate development-only NorBERTo on the pinned EP1 holdout without training."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path

from gpt_classifier.data import LABELS, canonical, digest, file_digest, read_jsonl, read_xlsx, write_json, write_jsonl
from gpt_classifier.evaluation import metrics
from norberto_classifier.partitions import development_partition

ROOT = Path(__file__).resolve().parents[1]


def evaluate(args: argparse.Namespace) -> dict:
    training = json.loads((args.training_run / "run.json").read_text())
    config = training["config"]
    original = json.loads((ROOT / "models/norberto/run.json").read_text())
    if (training.get("status") != "completed" or training.get("mode") != "development-training"
            or training.get("data_partition") != "development" or not training.get("holdout_excluded")
            or training.get("training_rows") != 16093 or training.get("reserved_holdout_rows") != 3999
            or training.get("recipe_version") != 2 or config != original["config"]
            or not math.isclose(training["training"]["epochs_executed"], 2)
            or training["training"]["training_stopped_by_budget"]):
        raise ValueError("A completed corrected-loss development-only training run is required.")
    if training["split_sha256"] != config["holdout_split_sha256"]:
        raise ValueError("The training split differs from the original frozen holdout.")
    for path, field in ((args.ep1_root / "src/clarity/finetune.py", "ep1_trainer_sha256"),
                        (ROOT / "src/norberto_classifier/partitions.py", "partition_implementation_sha256"),
                        (ROOT / "src/norberto_classifier/training_loss.py", "loss_implementation_sha256")):
        if file_digest(path) != training[field]:
            raise ValueError(f"The recorded implementation changed: {path.name}")
    adapter = args.training_run / "final_model"
    for name, expected in training["adapter_files_sha256"].items():
        path = adapter / name
        if not path.resolve().is_relative_to(adapter.resolve()) or file_digest(path) != expected:
            raise ValueError(f"The adapter file changed: {name}")
    source_digest = file_digest(args.training_run / "run.json")
    states = [json.loads(path.read_text()) for path in (args.training_run / "checkpoints").glob("checkpoint-*/trainer_state.json")]
    state = max(states, key=lambda row: row["global_step"])
    if state["global_step"] != state["max_steps"] or state["global_step"] != 2012 or state["epoch"] != 2:
        raise ValueError("Training did not reach the fixed final update.")
    source = args.ep1_root / "data/train.xlsx"
    records, info = read_xlsx(source)  # Labels remain unavailable to inference.
    if info["sha256"] != training["dataset_sha256"] or len(records) != 20092:
        raise ValueError("The source differs from the original EP1 training workbook.")
    split_path = args.holdout_splits or args.ep1_root / "results/research/dataset-hypotheses-20260923/splits.csv"
    if file_digest(split_path) != training["split_sha256"]:
        raise ValueError("The holdout split checksum changed.")
    with split_path.open(encoding="utf-8", newline="") as stream:
        split = list(csv.DictReader(stream))
    if len(split) != 20092 or [int(row["excel_row"]) for row in split] != [r.excel_row for r in records]:
        raise ValueError("The split row IDs do not match the workbook.")
    if any(row["role"] not in {"development", "holdout"} for row in split):
        raise ValueError("Unknown split role.")
    groups = {role: {row["group_id"] for row in split if row["role"] == role} for role in ("development", "holdout")}
    development = [r for r, row in zip(records, split) if row["role"] == "development"]
    holdout = [r for r, row in zip(records, split) if row["role"] == "holdout"]
    saved = read_jsonl(args.training_run / "training-rows.jsonl")
    if len(development) != len(saved) or len(saved) != 16093 or len(holdout) != 3999 or groups["development"] & groups["holdout"]:
        raise ValueError("Holdout coverage or development group isolation failed.")
    for record, row in zip(development, saved):
        if row["excel_row"] != record.excel_row or row["text_sha256"] != digest(record.text):
            raise ValueError("Recorded training rows differ from the development partition.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "run.json", {"status": "running", "training_performed": False})
    try:
        sys.path.insert(0, str(args.ep1_root / "src"))
        from clarity import finetune as engine
        import numpy as np
        import torch

        for package, expected in (("transformers", "4.57.1"), ("peft", "0.17.1"), ("accelerate", "1.10.1")):
            if importlib.metadata.version(package) != expected:
                raise RuntimeError(f"Use {package}=={expected}.")
        use_cpu, device = engine.select_execution_device("gpu")
        torch.set_num_threads(config["threads"])
        print("Verified 16,093 training rows and 3,999 disjoint holdout rows. Loading adapter.", flush=True)
        started = time.monotonic()
        tokenizer = engine.AutoTokenizer.from_pretrained(adapter, local_files_only=True)
        features, tokenization = engine.tokenize_corpus(np.asarray([r.text for r in holdout]), tokenizer, config,
                                                      cache_dir=args.output_dir / "token-cache")
        model = engine.make_model(config, config["seed"], adapter_path=adapter, allow_download=False)
        model.requires_grad_(False)

        class Progress(engine.TrainerCallback):
            def __init__(self):
                self.batches = 0

            def on_prediction_step(self, args, state, control, **kwargs):
                self.batches += 1
                if self.batches % 50 == 0:
                    print(f"Holdout inference: {min(self.batches * 8, len(holdout))}/{len(holdout)} rows", flush=True)

        trainer = engine.Trainer(model=model, args=engine.TrainingArguments(
            output_dir=str(args.output_dir / "inference"), per_device_eval_batch_size=8,
            fp16=False, use_cpu=use_cpu, report_to=[], disable_tqdm=True,
            dataloader_num_workers=0, eval_accumulation_steps=32), processing_class=tokenizer,
            data_collator=engine.DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8), callbacks=[Progress()])
        torch.cuda.reset_peak_memory_stats()
        raw = trainer.predict(engine.TokenizedRows(features, None, np.arange(len(features))))
        logits = np.asarray(raw.predictions)
        if logits.shape != (3999, 3) or not np.isfinite(logits).all():
            raise FloatingPointError("Holdout logits are invalid or nonfinite.")
        probabilities = torch.softmax(torch.from_numpy(logits).float(), dim=-1).numpy()
        predictions = [{"excel_row": r.excel_row, "text_sha256": digest(r.text),
                        "label": LABELS[int(probabilities[i].argmax())],
                        "probabilities": probabilities[i].tolist(), "logits": logits[i].tolist()}
                       for i, r in enumerate(holdout)]
        elapsed = time.monotonic() - started
        write_jsonl(args.output_dir / "predictions.jsonl", predictions)
        # Only now read gold labels, after persisting all text-only predictions.
        labeled, gold_info = read_xlsx(source, labeled=True)
        selected, partition_info = development_partition(labeled, split_path, training["split_sha256"],
                                                         development_rows=16093, holdout_rows=3999)
        if any(row["label"] != record.label for row, record in zip(saved, selected)):
            raise ValueError("Recorded training labels do not match the source.")
        held_ids = {r.excel_row for r in holdout}
        gold = [r for r in labeled if r.excel_row in held_ids]
        scores = metrics(gold, predictions)
        scores["precision_macro"] = scores["classification_report"]["macro avg"]["precision"]
        scores["recall_macro"] = scores["classification_report"]["macro avg"]["recall"]
        scores["correct"] = sum(row["label"] == truth.label for row, truth in zip(predictions, gold))
        scores["incorrect"] = len(gold) - scores["correct"]
        if (gold_info["sha256"] != info["sha256"] or file_digest(source) != info["sha256"]
                or file_digest(args.training_run / "run.json") != source_digest
                or file_digest(adapter / "adapter_model.safetensors") != training["adapter_files_sha256"]["adapter_model.safetensors"]):
            raise ValueError("A model or evaluation input changed during inference.")
        report = {"status": "completed", "phase": "holdout-validation", "dataset": "ep1-frozen-holdout",
                  "rows": 3999, "labels": list(LABELS), "input": info, "split_sha256": training["split_sha256"],
                  "config": config, "recipe_version": 2, "loss_normalization": training["loss_normalization"],
                  "adapter_sha256": digest(canonical({"files": training["adapter_files_sha256"], "config": config})),
                  "adapter_weights_sha256": training["adapter_files_sha256"]["adapter_model.safetensors"],
                  "training_run_sha256": source_digest, "training_rows_verified": True, "partition": partition_info,
                  "evaluation_script_sha256": file_digest(Path(__file__)),
                  "predictions_sha256": file_digest(args.output_dir / "predictions.jsonl"),
                  "training_performed": False, "evaluation_performed": True, "gold_read_after_predictions": True,
                  "test1_accessed": False, "device": device, "inference_seconds": elapsed, "tokenization": tokenization,
                  "memory": {"peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                             "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}, "metrics": scores,
                  "caveat": "Historically reused EP1 holdout; disjoint from this model's training, not historically untouched."}
        write_json(args.output_dir / "metrics.json", scores)
        write_json(args.output_dir / "run.json", report)
        return report
    except Exception as error:
        failure = {"status": "failed", "training_performed": False, "error": f"{type(error).__name__}: {error}"}
        write_json(args.output_dir / "failure.json", failure)
        write_json(args.output_dir / "run.json", failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, default=ROOT / "results/norberto-development-normalized")
    parser.add_argument("--ep1-root", type=Path, default=ROOT.parent / "EP1")
    parser.add_argument("--holdout-splits", type=Path)
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
