"""Fit the fixed EP1 logistic recipe on development only; never fit or score holdout."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import warnings

from gpt_classifier.data import LABELS, Example, canonical, digest, file_digest, read_xlsx, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
DATASET_SHA256 = "0e9219233664ac675fc47982bbc822acd3bb15d67fb7bfc26b67bec047e46818"
SPLIT_SHA256 = "0f78e7a809c7b7584db05b33cf03a73ff586d5de02ba824c16562d5070d41494"


def select_development(records: list[Example], split: list[dict], *,
                       expected_development: int = 16093, expected_holdout: int = 3999) -> tuple[list[Example], dict]:
    if ([row.excel_row for row in records] != [int(row["excel_row"]) for row in split]
            or len({row.excel_row for row in records}) != len(records)
            or Counter(row["role"] for row in split) != {
                "development": expected_development, "holdout": expected_holdout}
            or any(not row["group_id"] for row in split)):
        raise ValueError("The fixed partition must cover all source rows exactly, in order.")
    groups = {role: {row["group_id"] for row in split if row["role"] == role}
              for role in ("development", "holdout")}
    if groups["development"] & groups["holdout"]:
        raise ValueError("Development and holdout similarity groups overlap.")
    development = [record for record, row in zip(records, split) if row["role"] == "development"]
    if {row.label for row in development} != set(LABELS):
        raise ValueError("Development must contain all three classes.")
    return development, {"data_partition": "development", "training_rows": len(development),
                         "reserved_holdout_rows": expected_holdout,
                         "training_groups": len(groups["development"]),
                         "holdout_groups": len(groups["holdout"]),
                         "holdout_row_overlap": 0, "holdout_group_overlap": 0,
                         "holdout_used_for_fit": False, "holdout_evaluated": False}


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    source = args.ep1_root / "data/train.xlsx"
    split_path = args.ep1_root / "results/research/dataset-hypotheses-20260923/splits.csv"
    if file_digest(source) != DATASET_SHA256 or file_digest(split_path) != SPLIT_SHA256:
        raise ValueError("The EP1 workbook or frozen split differs from the approved source.")
    records, _ = read_xlsx(source, labeled=True)
    with split_path.open(encoding="utf-8", newline="") as stream:
        split = list(csv.DictReader(stream))
    development, partition = select_development(records, split)
    reference = json.loads((ROOT / "src/rl_classifier/frozen_manifest.json").read_text())
    for name, expected in reference["runtime_versions"].items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f"Use the pinned lexical runtime: {name}=={expected}.")
    if args.output_dir.exists():
        raise FileExistsError(f"Use a new output directory: {args.output_dir}")
    plan = {"status": "planned", **partition, "dataset_sha256": DATASET_SHA256,
            "split_sha256": SPLIT_SHA256, "classifier": reference["classifier_parameters"],
            "output_dir": str(args.output_dir), "parameter_search_performed": False,
            "test_data_accessed": False, "training_performed": False}
    if args.dry_run:
        return plan

    sys.path.insert(0, str(args.ep1_root / "src"))
    from model_research.models import build_model
    from lexical_classifier.inference import FrozenLexicalClassifier
    import joblib
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits

    model = build_model("lexical", seed=42, cache_dir=None)
    model.set_params(classifier__C=.25)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "run.json", plan | {"status": "running"})
    try:
        print(f"Fitting logistic regression on {len(development)} development rows; holdout excluded.", flush=True)
        with threadpool_limits(limits=2), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit([row.text for row in development], [row.label for row in development])
        config = {"model": "word_char_logistic", "classifier__C": .25, "seed": 42,
                  "data_partition": "development", "dataset_sha256": DATASET_SHA256,
                  "split_sha256": SPLIT_SHA256}
        identities = [{"excel_row": row.excel_row, "text_sha256": digest(row.text), "label": row.label}
                      for row in development]
        metadata = {"selected": "development_word_char_logistic", **partition,
                    "dataset": {"rows": len(development), "class_counts": dict(Counter(row.label for row in development))},
                    "dataset_sha256": DATASET_SHA256, "split_sha256": SPLIT_SHA256,
                    "training_rows_identity_sha256": digest(canonical(identities)),
                    "config": config, "runtime_versions": reference["runtime_versions"],
                    "iterations": model.named_steps["classifier"].n_iter_.tolist(), "converged": True,
                    "parameter_search_performed": False, "test_data_accessed": False,
                    "training_script_sha256": file_digest(Path(__file__)),
                    "ep1_model_builder_sha256": file_digest(args.ep1_root / "src/model_research/models.py")}
        write_json(args.output_dir / "config.json", config)
        write_json(args.output_dir / "metadata.json", metadata)
        write_jsonl(args.output_dir / "training-rows.jsonl", identities)
        artifact = args.output_dir / "model.joblib"
        joblib.dump({"format_version": 1, "labels": LABELS, "model": model, "metadata": metadata}, artifact, compress=3)
        contract = {**reference, "artifact_sha256": file_digest(artifact),
                    "source_artifact": str(artifact), "source_config": "config.json",
                    "source_metadata": "metadata.json", "selected": metadata["selected"],
                    "data_partition": "development", "training_rows": len(development),
                    "reserved_holdout_rows": partition["reserved_holdout_rows"], "split_sha256": SPLIT_SHA256,
                    "files": {name: file_digest(args.output_dir / name) for name in ("metadata.json", "config.json")}}
        contract_path = args.output_dir / "frozen_manifest.json"
        write_json(contract_path, contract)
        FrozenLexicalClassifier(contract_path, bundle_dir=args.output_dir).load()
        write_json(args.output_dir / "manifest.json", {
            "schema_version": 1, "model": "rl", "format": "joblib", "labels": list(LABELS),
            "data_partition": "development", "training_supported": False,
            "files": {name: file_digest(args.output_dir / name) for name in (
                "model.joblib", "config.json", "metadata.json", "frozen_manifest.json")}})
        report = plan | {"status": "completed", "training_performed": True,
                         "artifact_sha256": contract["artifact_sha256"], "converged": True,
                         "safe_reload_verified": True, "elapsed_seconds": time.monotonic() - started}
        write_json(args.output_dir / "run.json", report)
        return report
    except BaseException as error:
        write_json(args.output_dir / "run.json", plan | {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep1-root", type=Path, default=ROOT.parent / "EP1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/rl-development")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        report = run(args)
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
