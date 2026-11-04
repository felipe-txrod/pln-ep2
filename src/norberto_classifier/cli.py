"""Predict workbook rows with the frozen EP1 model, without training or evaluation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .artifacts import LABELS
from .inference import NorbertoClassifier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen EP1 NorBERTo inference; no training commands")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("predict")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--artifact-dir", type=Path)
    command.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args(argv)
    try:
        from gpt_classifier.data import digest, file_digest, read_xlsx, write_json, write_jsonl
        if args.output_dir.exists():
            raise FileExistsError(f"Use a new output directory: {args.output_dir}")
        classifier = NorbertoClassifier(args.artifact_dir, device=args.device)
        records, source = read_xlsx(args.input)
        started = time.monotonic()
        predictions = classifier.predict([record.text for record in records])
        if file_digest(args.input) != source["sha256"]:
            raise ValueError("The input workbook changed during inference.")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_jsonl(args.output_dir / "predictions.jsonl", [
            {"excel_row": record.excel_row, "text_sha256": digest(record.text), **asdict(prediction)}
            for record, prediction in zip(records, predictions)])
        assert classifier.bundle is not None
        report = {"status": "completed", "training_performed": False, "evaluation_performed": False,
                  "input": source, "labels": list(LABELS), "device": args.device,
                  "artifact_manifest_sha256": classifier.bundle.identity_sha256,
                  "source_ep1_metadata_sha256": file_digest(classifier.bundle.root / "run.json"),
                  "config": classifier.bundle.config,
                  "source_training_config": classifier.bundle.metadata["config"],
                  "tokenization": classifier.tokenization_statistics,
                  "duration_seconds": time.monotonic() - started}
        write_json(args.output_dir / "run.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
