"""Expose preparation, development freezing and a separate retrospective evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpt_classifier.backend import file_lock
from .pipeline import develop, evaluate_test1, load_frozen, predict_test1, prepare, validate_plan
from .reporting import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-dir", type=Path, default=Path("results/svm-final-budgeted-20261106"))
    parser.add_argument("--search-config", type=Path, default=Path("configs/svm-budgeted.json"))
    parser.add_argument("command", choices=("prepare", "develop", "predict-test1", "evaluate-test1", "report"))
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.search_dir, args.search_config, args.output_dir)
        else:
            with file_lock(args.output_dir / ".finalization.lock"):
                if args.command == "develop":
                    result = develop(args.output_dir)
                elif args.command == "predict-test1":
                    manifest = predict_test1(args.output_dir)
                    result = {"status": manifest["status"], "rows": manifest["rows"], "gold_accessed": False}
                elif args.command == "evaluate-test1":
                    final = evaluate_test1(args.output_dir)
                    result = {"status": final["status"], "recommended_configuration": final["recommended_configuration"],
                              "test1_macro_f1": {name: final["metrics"][name]["f1_macro"] for name in ("A", "B", "C")}}
                else:
                    validate_plan(args.output_dir)
                    report(args.output_dir)
                    result = {"status": "report_updated"}
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
