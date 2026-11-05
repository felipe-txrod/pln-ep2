"""Select and combine saved model runs without executing any classifier."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import EnsembleConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline ensemble with selectable classifiers")
    commands = parser.add_subparsers(dest="command", required=True)
    fit_stack_parser = commands.add_parser("fit-stacking", help="Fit a meta-LR on the disjoint frozen holdout role only")
    fit_stack_parser.add_argument("--config", type=Path, default=Path("configs/ensemble-stacking.json"))
    fit_stack_parser.add_argument("--output-dir", type=Path, required=True)
    predict_stack_parser = commands.add_parser("predict-stacking", help="Apply frozen meta-LR to calibrated member probabilities")
    predict_stack_parser.add_argument("--model", type=Path, default=Path("models/ensemble-stacking-qwen/model.json"))
    predict_stack_parser.add_argument("--input", type=Path, required=True)
    predict_stack_parser.add_argument("--output-dir", type=Path, required=True)
    predict_stack_parser.add_argument("--soft-run", type=Path, help="Reuse an existing soft prediction run, without inference")
    predict_stack_parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    predict_stack_parser.add_argument("--live-norberto", action="store_true")
    predict_stack_parser.add_argument("--live-qwen", action="store_true")
    evaluate_stack_parser = commands.add_parser("evaluate-stacking", help="Final stacking/soft/member comparison; no tuning")
    evaluate_stack_parser.add_argument("--run", type=Path, required=True)
    evaluate_stack_parser.add_argument("--gold", type=Path, default=Path("data/test1-rotulado-para-teste-estatistico.xlsx"))
    evaluate_stack_parser.add_argument("--output-dir", type=Path, required=True)
    fit_soft_parser = commands.add_parser("fit-soft", help="Calibrate frozen scores and choose soft weights on disjoint holdout roles")
    fit_soft_parser.add_argument("--config", type=Path, default=Path("configs/ensemble-soft.json"))
    fit_soft_parser.add_argument("--output-dir", type=Path, required=True)
    fit_soft_parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    predict_soft_parser = commands.add_parser("predict-soft", help="Apply a frozen calibrated soft ensemble without reading labels")
    predict_soft_parser.add_argument("--model", type=Path, default=Path("models/ensemble-soft-qwen/model.json"))
    predict_soft_parser.add_argument("--input", type=Path, required=True)
    predict_soft_parser.add_argument("--output-dir", type=Path, required=True)
    predict_soft_parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    predict_soft_parser.add_argument("--live-norberto", action="store_true", help="Use local NorBERTo inference instead of the pinned historical score files")
    predict_soft_parser.add_argument("--live-qwen", action="store_true", help="Use the selected Qwen on GPU instead of pinned score files")
    evaluate_soft_parser = commands.add_parser("evaluate-soft", help="Final comparison on the 900 professor labels; no parameter updates")
    evaluate_soft_parser.add_argument("--run", type=Path, required=True)
    evaluate_soft_parser.add_argument("--gold", type=Path, default=Path("data/test1-rotulado-para-teste-estatistico.xlsx"))
    evaluate_soft_parser.add_argument("--output-dir", type=Path, required=True)
    combine_parser = commands.add_parser("combine")
    combine_parser.add_argument("--config", type=Path, default=Path("configs/ensemble.json"))
    combine_parser.add_argument("--models", nargs="+")
    combine_parser.add_argument("--strategy", choices=("consensus", "majority", "consensus_gpt", "majority_gpt"))
    combine_parser.add_argument("--tiebreaker", help="Arbitrator identifier; defaults to gpt")
    combine_parser.add_argument("--input", type=Path, required=True)
    combine_parser.add_argument("--output-dir", type=Path, required=True)
    combine_parser.add_argument("--tiebreaker-run", type=Path, help="Saved predictions from the selected arbitrator")
    combine_parser.add_argument("--gpt-run", type=Path, help="Legacy GPT-only alias for --tiebreaker-run")
    combine_parser.add_argument("--phase", choices=("inference", "statistical"), default="inference")
    combine_parser.add_argument("--test2-run", type=Path)
    comparison_parser = commands.add_parser(
        "compare-majority", help="Exploratory local majority comparison; never changes delivery defaults")
    comparison_parser.add_argument("--config", type=Path, default=Path("configs/ensemble-majority-test1.json"))
    comparison_parser.add_argument("--input", type=Path, default=Path("data/test1.xlsx"))
    comparison_parser.add_argument("--gold", type=Path,
                                   default=Path("data/test1-rotulado-para-teste-estatistico.xlsx"))
    comparison_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in {"fit-stacking", "predict-stacking", "evaluate-stacking"}:
            from .stacking_pipeline import evaluate_stacking, fit_stacking, predict_stacking
            if args.command == "fit-stacking":
                report = fit_stacking(args.config, args.output_dir)
            elif args.command == "predict-stacking":
                report = predict_stacking(args.model, args.input, args.output_dir, soft_run=args.soft_run,
                                          device=args.device, live_norberto=args.live_norberto, live_qwen=args.live_qwen)
            else:
                report = evaluate_stacking(args.run, args.gold, args.output_dir)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.command in {"fit-soft", "predict-soft", "evaluate-soft"}:
            from .soft_pipeline import evaluate_soft, fit_soft, predict_soft
            if args.command == "fit-soft":
                report = fit_soft(args.config, args.output_dir, device=args.device)
            elif args.command == "predict-soft":
                report = predict_soft(args.model, args.input, args.output_dir,
                                      device=args.device, live_norberto=args.live_norberto, live_qwen=args.live_qwen)
            else:
                report = evaluate_soft(args.run, args.gold, args.output_dir)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.command == "compare-majority":
            from .comparison import compare_majority
            report = compare_majority(EnsembleConfig.load(args.config), args.input, args.gold, args.output_dir)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        from .pipeline import combine
        config = EnsembleConfig.load(args.config, models=args.models, strategy=args.strategy,
                                     tiebreaker=args.tiebreaker)
        report = combine(config, args.input, args.output_dir, gpt_run=args.gpt_run,
                         tiebreaker_run=args.tiebreaker_run,
                         phase=args.phase, test2_run=args.test2_run)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2 if report["status"] == "pending_tiebreaks" else 0
    except (ValueError, OSError, RuntimeError, ImportError, TypeError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
