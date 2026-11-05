"""Refit NorBERTo on the full EP1 training workbook using the original trainer."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_report(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], list]:
    from gpt_classifier.data import LABELS, read_xlsx

    source = ROOT / "models/norberto"
    contract = json.loads((ROOT / "src/norberto_classifier/frozen_manifest.json").read_text())
    for name in ("run.json", "config.json"):
        if checksum(source / name) != contract["files"][name]:
            raise ValueError(f"The original NorBERTo training reference changed: {name}")
    original = json.loads((source / "run.json").read_text())
    config = json.loads((source / "config.json").read_text())
    if config != original["config"] or config["epochs"] != 2:
        raise ValueError("Use the original effective two-epoch training configuration.")
    if checksum(args.input) != original["dataset_sha256"]:
        raise ValueError("The input must be the unchanged EP1 training workbook.")
    records, info = read_xlsx(args.input, labeled=True)
    if len(records) != original["source_rows"] or len(records) != 20092:
        raise ValueError("Full refitting requires all 20,092 original training rows.")
    if args.output_dir.exists():
        raise FileExistsError(f"Use a new output directory: {args.output_dir}")
    if not args.output_dir.resolve().is_relative_to((ROOT / "results").resolve()):
        raise ValueError("Training outputs must use a new directory under EP2/results.")
    trainer_source = args.ep1_root / "src/clarity/finetune.py"
    if not trainer_source.is_file():
        raise FileNotFoundError(f"The original EP1 trainer is required: {trainer_source}")
    plan = {"mode": "full-training", "protocol": "fresh_base_all_ep1_training_rows_fixed_parameters",
            "recipe_version": 2, "loss_normalization": "sample_count_across_accumulation",
            "training_wrapper_sha256": checksum(Path(__file__)),
            "loss_implementation_sha256": checksum(ROOT / "src/norberto_classifier/training_loss.py"),
            "training_rows": len(records), "labels": list(LABELS), "dataset_sha256": info["sha256"],
            "config": config, "effective_batch_size": config["batch_size"] * config["gradient_accumulation"],
            "validation_used": False, "holdout_excluded": False, "test_data_used": False,
            "initialization": "fresh_pinned_base; original adapter is not loaded",
            "original_training_run_sha256": checksum(source / "run.json"),
            "ep1_trainer_sha256": checksum(trainer_source)}
    return plan, records


def train(args: argparse.Namespace, plan: dict[str, Any], records: list) -> int:
    sys.path.insert(0, str(args.ep1_root / "src"))
    from clarity import finetune as original_trainer
    from gpt_classifier.data import LABELS, digest, write_jsonl
    from norberto_classifier.training_loss import sample_normalized_trainer
    import numpy as np
    import torch

    for package, expected in (("transformers", "4.57.1"), ("peft", "0.17.1"), ("accelerate", "1.10.1")):
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"The original trainer requires {package}=={expected}.")
    config = original_trainer.read_finetune_config(ROOT / "models/norberto/config.json")
    if config != plan["config"]:
        raise ValueError("The training configuration changed after preparation.")
    use_cpu, accelerator = original_trainer.select_execution_device("gpu")
    torch.set_num_threads(config["threads"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {**plan, "status": "running", "accelerator": accelerator}
    write_report(args.output_dir / "run.json", metadata)
    write_report(args.output_dir / "config.json", config)
    write_jsonl(args.output_dir / "training-rows.jsonl", [
        {"excel_row": row.excel_row, "text_sha256": digest(row.text), "label": row.label}
        for row in records])
    try:
        tokenizer = original_trainer.AutoTokenizer.from_pretrained(
            config["model_name"], revision=config["revision"], cache_dir=".cache/huggingface",
            local_files_only=True, do_lower_case=False)
        features, tokenization = original_trainer.tokenize_corpus(
            np.asarray([row.text for row in records]), tokenizer, config)
        labels = np.asarray([LABELS.index(row.label) for row in records])
        print(f"FULL TRAINING: rows={len(records)}; epochs={config['epochs']}; "
              "holdout_excluded=False; validation_used=False; initialization=fresh_base; "
              "recipe_version=2; loss_normalization=sample_count_across_accumulation", flush=True)
        with sample_normalized_trainer(original_trainer):
            trainer, training = original_trainer.train_one(
                config, tokenizer, features, labels, np.arange(len(records)), None,
                args.output_dir, config["seed"], final=True, allow_download=False, use_cpu=use_cpu)
        if checksum(args.input) != plan["dataset_sha256"]:
            raise ValueError("The EP1 training workbook changed during training.")
        complete = (not training["training_stopped_by_budget"]
                    and math.isclose(training["epochs_executed"], config["epochs"])
                    and math.isfinite(training["training_loss"]))
        adapter = args.output_dir / "final_model"
        report = {**metadata, "status": "completed" if complete else "incomplete",
                  "training": training, "tokenization": tokenization,
                  "adapter_files_sha256": {str(path.relative_to(adapter)): checksum(path)
                                           for path in sorted(adapter.rglob("*")) if path.is_file()}}
        write_report(args.output_dir / "run.json", report)
        del trainer
        print(json.dumps({"status": report["status"], "training_rows": len(records),
                          "epochs": training["epochs_executed"], "adapter": str(adapter)}, indent=2), flush=True)
        return 0 if complete else 1
    except Exception as error:
        write_report(args.output_dir / "run.json", {**metadata, "status": "failed",
                                                    "error": f"{type(error).__name__}: {error}"})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep1-root", type=Path, default=ROOT.parent / "EP1")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/norberto-full-training")
    parser.add_argument("--dry-run", action="store_true", help="Validate all rows and fixed parameters without loading models")
    args = parser.parse_args()
    args.input = args.input if args.input is not None else args.ep1_root / "data/train.xlsx"
    try:
        plan, records = prepare(args)
        if args.dry_run:
            if any(name in sys.modules for name in ("torch", "transformers", "peft")):
                raise RuntimeError("Dry-run unexpectedly imported model libraries.")
            print(json.dumps({**plan, "training_performed": False}, indent=2))
            return 0
        return train(args, plan, records)
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
