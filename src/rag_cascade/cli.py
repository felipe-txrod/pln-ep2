"""Prepare and inspect freely; classification always requires --execute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train-only RAG context experiments over frozen selective stacking")
    commands = parser.add_subparsers(dest="command", required=True)
    encoder = commands.add_parser("prepare-encoder", help="Explicitly download the pinned local encoder")
    encoder.add_argument("--cache", type=Path, default=Path(".cache/retrieval"))
    encoder.add_argument("--download", action="store_true")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, default=Path("configs/rag-cascade.json"))
    prepare.add_argument("--output-dir", type=Path, required=True)
    for name in ("plan", "run", "score"):
        command = commands.add_parser(name)
        command.add_argument("--study", type=Path, required=True)
        command.add_argument("--stage", choices=("pilot", "confirmation"), required=True)
        if name == "run":
            command.add_argument("--execute", action="store_true")
            command.add_argument("--max-new-requests", type=int)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--study", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--stacking-run", type=Path, required=True)
    predict.add_argument("--test2-run", type=Path)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--execute", action="store_true")
    predict.add_argument("--max-new-requests", type=int)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--gold", type=Path, default=Path("data/test1-rotulado-para-teste-estatistico.xlsx"))
    evaluate.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-encoder":
            if not args.download:
                result = {"status": "preview", "download_executed": False,
                          "message": "Add --download explicitly to fetch the pinned local E5 encoder; no classifier call is made."}
            else:
                from gpt_classifier.embeddings import prepare_model
                result = prepare_model(args.cache)
        elif args.command == "prepare":
            from .config import RagConfig
            from .data import prepare
            result = prepare(RagConfig.load(args.config), args.output_dir)
        elif args.command in {"plan", "run", "score", "freeze"}:
            from . import study
            if args.command == "freeze":
                result = study.freeze(args.study, args.output_dir)
            elif args.command == "run":
                result = study.run(args.study, args.stage, execute=args.execute, max_new_requests=args.max_new_requests)
            else:
                result = getattr(study, args.command)(args.study, args.stage)
        else:
            from . import prediction
            if args.command == "predict":
                result = prediction.predict(args.model, args.input, args.stacking_run, args.output_dir,
                                             test2_run=args.test2_run, execute=args.execute,
                                             max_new_requests=args.max_new_requests)
            else:
                result = prediction.evaluate(args.run, args.gold, args.output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, TypeError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
