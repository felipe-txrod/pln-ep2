"""Shared command-line inference for frozen SVM and logistic classifiers."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .inference import FrozenLexicalClassifier


def main(factory: Callable[..., FrozenLexicalClassifier], argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen EP1 lexical inference, without training")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info")
    predict = commands.add_parser("predict")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--artifact", type=Path)
    predict.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        if args.command == "info":
            print(json.dumps(factory().describe(), indent=2))
            return 0
        from gpt_classifier.data import canonical, digest, file_digest, read_xlsx, write_json, write_jsonl
        if args.output_dir.exists():
            raise FileExistsError(f"Use a new output directory: {args.output_dir}")
        records, source = read_xlsx(args.input)
        classifier = factory(args.artifact, threads=args.threads)
        predictions = classifier.predict([record.text for record in records])
        if file_digest(args.input) != source["sha256"]:
            raise ValueError("The source workbook changed during inference.")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_jsonl(args.output_dir / "predictions.jsonl", [
            {"excel_row": record.excel_row, "text_sha256": digest(record.text),
             **{key: value for key, value in asdict(prediction).items() if value is not None}}
            for record, prediction in zip(records, predictions)])
        implementation = {f"{directory.name}/{path.name}": file_digest(path)
                          for directory in (Path(__file__).parent, classifier.manifest_path.parent)
                          for path in sorted(directory.glob("*.py"))}
        report = {"status": "completed", "input": source, "rows": len(records),
                  "training_performed": False, "evaluation_performed": False,
                  "model_type": classifier.manifest["model_type"],
                  "artifact_sha256": classifier.manifest["artifact_sha256"],
                  "artifact_manifest_sha256": file_digest(classifier.manifest_path),
                  "implementation_sha256": digest(canonical(implementation)),
                  "labels": classifier.manifest["labels"], "threads": args.threads}
        write_json(args.output_dir / "run.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
