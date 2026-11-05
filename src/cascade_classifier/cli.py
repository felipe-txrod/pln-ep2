"""Separate, explicit stages for a reproducible selective cascade experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen stacking with selective GPT decisions")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, default=Path("configs/cascade.json"))
    prepare.add_argument("--output-dir", type=Path, required=True)
    for name in ("validate-initial", "validate-extended"):
        command = commands.add_parser(name)
        command.add_argument("--validation-dir", type=Path, required=True)
        command.add_argument("--cache", type=Path, default=Path(os.environ.get("CASCADE_CACHE", ".cache/cascade/responses.sqlite3")))
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--validation-dir", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--stacking-run", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--compare-budgets", action="store_true", help="Evaluate the frozen 0/5/10/20% variants; otherwise use only the validation-selected budget")
    predict.add_argument("--cache", type=Path, default=Path(os.environ.get("CASCADE_CACHE", ".cache/cascade/responses.sqlite3")))
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--gold", type=Path, default=Path("data/test1-rotulado-para-teste-estatistico.xlsx"))
    evaluate.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        # Every remote stage is explicit; preparation and scoring never execute GPT implicitly.
        if args.command.startswith("validate-"):
            from .experiment import validate_extended, validate_initial
            function = validate_initial if args.command == "validate-initial" else validate_extended
            result = function(args.validation_dir, args.cache)
        else:
            from .experiment import evaluate_cascade, freeze, predict_cascade, prepare_validation
            from .config import CascadeConfig
            if args.command == "prepare":
                result = prepare_validation(CascadeConfig.load(args.config), args.output_dir)
            elif args.command == "freeze":
                result = freeze(args.validation_dir, args.output_dir)
            elif args.command == "predict":
                result = predict_cascade(args.model, args.input, args.stacking_run, args.output_dir, args.cache,
                                         compare_budgets=args.compare_budgets)
            else:
                result = evaluate_cascade(args.run, args.gold, args.output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, TypeError, KeyError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
