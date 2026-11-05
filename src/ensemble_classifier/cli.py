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
