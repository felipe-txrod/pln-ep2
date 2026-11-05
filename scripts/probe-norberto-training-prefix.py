"""Run a bounded development-training prefix with the original full-run schedule."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ep1-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.updates < 2012:
        parser.error('Choose a positive prefix shorter than the complete 2,012-update run.')
    source = Path(__file__).with_name('train-norberto-full.py')
    spec = importlib.util.spec_from_file_location('training_runner', source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    args.input = args.ep1_root / 'data/train.xlsx'
    args.partition = 'development'
    args.holdout_splits = None
    plan, records = runner.prepare(args)
    plan.update(diagnostic_prefix=True, requested_updates=args.updates,
                diagnostic_script_sha256=runner.checksum(Path(__file__)))
    sys.path.insert(0, str(args.ep1_root / 'src'))
    from clarity import finetune as engine

    class StopPrefix(engine.TrainerCallback):
        def on_step_end(self, training_args, state, control, **kwargs):
            if state.global_step >= args.updates:
                control.should_training_stop = True
            return control

    previous = engine.Trainer

    class PrefixTrainer(previous):
        def __init__(self, *positional, **kwargs):
            kwargs['callbacks'] = [*kwargs.get('callbacks', []), StopPrefix()]
            super().__init__(*positional, **kwargs)

    engine.Trainer = PrefixTrainer
    try:
        # An intentionally incomplete run preserves the full 2,012-step schedule.
        result = runner.train(args, plan, records)
        return 0 if result == 1 else 1
    finally:
        engine.Trainer = previous


if __name__ == '__main__':
    raise SystemExit(main())
