"""Command-line entry points; GPU dependencies remain optional for data preparation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gpt_classifier.data import write_json
from .config import Config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verified Qwen3 NF4 QLoRA / FP16 LoRA classification")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "download", "smoke", "train", "_smoke-worker"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path)
        if name != "download":
            command.add_argument("--output-dir", type=Path, required=True)
        if name in {"prepare", "train"}:
            command.add_argument("--input", type=Path, default=Path("data/train.xlsx"))
        if name != "prepare":
            command.add_argument("--cache-dir", type=Path, default=Path(".cache/qwen-model"))
        if name == "train":
            command.add_argument("--smoke-report", type=Path)
            command.add_argument("--pilot", action="store_true",
                                 help="Run two optimizer steps on 96 training and 48 validation examples")
    diagnostic = commands.add_parser("probe")
    diagnostic.add_argument("--output", type=Path, required=True)
    performance = commands.add_parser("benchmark")
    performance.add_argument("--config", type=Path, required=True)
    performance.add_argument("--output-dir", type=Path, required=True)
    performance.add_argument("--cache-dir", type=Path, default=Path(".cache/qwen-model"))
    performance.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    performance.add_argument("--checkpointing", action="store_true")
    performance.add_argument("--measured-steps", type=int, default=3)
    inference = commands.add_parser("predict")
    inference.add_argument("--adapter", type=Path, required=True)
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--output-dir", type=Path, required=True)
    inference.add_argument("--cache-dir", type=Path, default=Path(".cache/qwen-model"))
    inference.add_argument("--phase", choices=("inference", "statistical"), default="inference")
    inference.add_argument("--test2-run", type=Path)
    inference.add_argument("--workbook", type=Path)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = Config.load(args.config) if getattr(args, "config", None) else Config()
        if args.command == "probe":
            from .hardware import probe
            result = probe()
            write_json(args.output, result)
            print(json.dumps(result, indent=2))
            return 0 if result["recommended_mode"] else 1
        if args.command == "benchmark":
            from .benchmark import benchmark
            result = benchmark(config, args.output_dir, args.cache_dir, batches=args.batches,
                               checkpointing=args.checkpointing, measured_steps=args.measured_steps)
        elif args.command == "download":
            from huggingface_hub import snapshot_download
            path = snapshot_download(config.model_id, revision=config.revision, cache_dir=str(args.cache_dir),
                                     allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
            result = {"snapshot": path, "revision": config.revision}
        elif args.command == "prepare":
            from .dataset import prepare
            manifest = prepare(args.input, args.output_dir, config)
            result = {key: value for key, value in manifest.items() if key not in {"train_rows", "validation_rows"}}
            result.update(train_rows=len(manifest["train_rows"]), validation_rows=len(manifest["validation_rows"]))
        elif args.command == "smoke":
            from .training import smoke_test
            result = smoke_test(config, args.output_dir, args.cache_dir)
        elif args.command == "_smoke-worker":
            from .training import fit
            result = fit(config, args.output_dir, args.cache_dir, smoke=True)
        elif args.command == "train":
            from .training import train
            result = train(config, args.input, args.output_dir, args.cache_dir,
                           smoke_report=args.smoke_report, pilot=args.pilot)
        elif args.command == "predict":
            from .inference import predict
            result = predict(args.adapter, args.cache_dir, args.input, args.output_dir, phase=args.phase,
                             test2_run=args.test2_run, workbook=args.workbook)
        else:
            from .config import LABEL2ID
            from .training import classification_metrics
            saved = json.loads((args.run / "validation-predictions.json").read_text(encoding="utf-8"))
            result = classification_metrics([LABEL2ID[r["gold"]] for r in saved],
                                            [LABEL2ID[r["label"]] for r in saved])
            write_json(args.run / "validation-metrics.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
