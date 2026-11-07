"""Amend the search budget without rewriting the original protocol or fold artifacts."""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from ensemble_classifier.files import read_json
from gpt_classifier.backend import file_lock
from gpt_classifier.data import LABELS, atomic_text, file_digest, write_json
from svm_research.cli import validate_plan as validate_original
from svm_research.data import load_data, location, runtime
from svm_research.search import broad_candidates, candidate_id, classification_metrics, grouped_folds
from .engine import candidates_a, identity, search


def implementation():
    source = Path(__file__).resolve().parents[1]
    paths = [*list((source / "svm_research").glob("*.py")), *list((source / "svm_budgeted").glob("*.py"))]
    paths += [source / relative for relative in ("lexical_classifier/inference.py", "gpt_classifier/data.py",
                                               "ensemble_classifier/files.py", "ensemble_classifier/soft_sources.py",
                                               "ensemble_classifier/soft_voting.py")]
    return {str(path.relative_to(source)): file_digest(path) for path in sorted(paths)}


def load_config(path):
    config = read_json(path)
    expected = {"source_run", "seed", "jobs", "fit_timeout_seconds", "phase_a_folds", "phase_b_folds",
                "phase_b_candidate_cap", "train", "split", "group_manifest", "group_audit", "original_split",
                "baseline", "baseline_scores"}
    if set(config) != expected:
        raise ValueError("Unknown or missing amended configuration fields.")
    for field in ("seed", "jobs", "fit_timeout_seconds", "phase_a_folds", "phase_b_folds", "phase_b_candidate_cap"):
        if type(config[field]) is not int:
            raise ValueError(f"Expected an integer for {field}.")
    if (config["seed"] != 42 or not 1 <= config["jobs"] <= 4 or not 10 <= config["fit_timeout_seconds"] <= 180
            or config["phase_a_folds"] != 3 or config["phase_b_folds"] != 5 or config["phase_b_candidate_cap"] != 25):
        raise ValueError("Invalid amended seed, folds, parallelism or compute budget.")
    return config


def base_config(config):
    return {k: v for k, v in config.items() if k not in {"source_run", "fit_timeout_seconds", "phase_b_candidate_cap"}}


def import_folds(source, output, data, *, perform_copy=True):
    """Import exact matching folds only after validating IDs, predictions and metrics."""
    source_plan = read_json(source / "phase-a/plan.json")
    old_candidates = {candidate_id(c): c for c in source_plan["candidates"]}
    selected = {candidate_id(c): c for c in candidates_a()}
    rows = np.asarray([r.excel_row for r in data.training])
    labels = np.asarray([r.label for r in data.training])
    folds = grouped_folds(labels, data.groups, 3, 42)
    if source_plan["seed"] != 42 or source_plan["folds"] != 3:
        raise ValueError("Original folds differ from the amended phase A.")
    imported, inventory, excluded = [], {}, []
    for report_path in sorted((source / "phase-a/trials").glob("*/fold-*.json")):
        report = read_json(report_path)
        identifier = report_path.parent.name
        if identifier not in old_candidates:
            raise ValueError("An original result has an unknown candidate identity.")
        fold = report["identity"]["fold"]
        if type(fold) is not int or fold not in range(3) or report.get("status") != "completed":
            raise ValueError("Original completed-fold metadata is invalid.")
        train, valid = folds[fold]
        expected = identity(old_candidates[identifier], fold, train, valid, rows, 42)
        array_path = report_path.with_suffix(".npz")
        if report["identity"] != expected or file_digest(array_path) != report["predictions_sha256"]:
            raise ValueError("Original fold provenance or checksum failed.")
        with np.load(array_path, allow_pickle=False) as values:
            if (set(values.files) != {"excel_rows", "predictions"}
                    or not np.array_equal(values["excel_rows"], rows[valid])
                    or values["predictions"].shape != (len(valid),)
                    or values["predictions"].dtype.kind not in "iu"
                    or np.any((values["predictions"] < 0) | (values["predictions"] > 2))):
                raise ValueError("Original fold predictions are incomplete or misaligned.")
            metrics = classification_metrics(labels[valid], np.asarray(LABELS)[values["predictions"]])
        if metrics != report["metrics"]:
            raise ValueError("Original fold metrics do not match its saved predictions.")
        for path in (report_path, array_path):
            inventory[str(path.resolve())] = file_digest(path)
        if identifier not in selected:
            excluded.append({"candidate_id": identifier, "fold": fold,
                             "reason": "Not in amended compute-bounded design; historical output preserved"})
            continue
        if perform_copy:
            destination = output / "phase-a/trials" / identifier
            destination.mkdir(parents=True, exist_ok=True)
            for path in (report_path, array_path):
                shutil.copy2(path, destination / path.name)
        imported.append({"candidate_id": identifier, "fold": fold,
                         "report_sha256": file_digest(report_path), "predictions_sha256": file_digest(array_path)})
    return {"source_completed_folds": len(imported) + len(excluded), "imported_folds": imported,
            "excluded_historical_folds": excluded, "source_files": inventory,
            "reuse_rule": "Identical estimator/feature parameters, seed, runtime, train/validation IDs, labels and fold predictions; max_iter remains 20000"}


def prepare(config_path, output):
    config = load_config(config_path)
    source = location(config["source_run"])
    if output.exists() or output.resolve() == source.resolve():
        raise FileExistsError("Amended preparation requires a new output directory.")
    with file_lock(source / ".research.lock"):
        old_config, old_plan = validate_original(location("configs/svm-research.json"), source)
        if any(config.get(k) != v for k, v in old_config.items() if k not in {"jobs", "cache_gib"}):
            raise ValueError("Data or CV settings differ from the immutable original run.")
        data = load_data(base_config(config))
        old_audit = read_json(source / "audit.json")
        if old_audit["label_identity"] != data.audit["label_identity"]:
            raise ValueError("Original training labels or text identities changed.")
        inventory = import_folds(source, output, data, perform_copy=False)
        original_files = {str((source / name).resolve()): file_digest(source / name)
                          for name in ("plan.json", "audit.json", "baseline.json", "training-membership.jsonl",
                                       "phase-a-candidates.json", "phase-a/plan.json", "phase-a/progress.json")}
        original_files.update(inventory["source_files"])
        amended = json.loads(json.dumps(old_plan["protocol"]))
        amended.update(
            phase_a={"candidates": len(candidates_a()), "folds": 3, "seed": 42,
                     "designs": [0, 1, 3, 4], "C": [.005, .05, .5], "norm": ["l1", "l2"],
                     "class_weight": [None, "balanced"], "families": ["word", "char", "char_wb", "word_char_wb"]},
            phase_b={"candidate_cap": 25, "representations": 2, "folds": 5, "seed": 143,
                     "design": "Three local C values, both class weights, hinge check, union branch weights"},
            compute={"jobs": config["jobs"], "numerical_threads_per_fit": 1,
                     "fit_subprocess_wall_seconds": config["fit_timeout_seconds"], "max_iter": 20000,
                     "shared_training_and_validation_sparse_features": True,
                     "timeout_policy": "Stop only the over-budget fit; mark its candidate ineligible, never invent a prediction"},
            amendment={"reason": "User requested lower computational cost after unnormalized folds took 643-991 seconds and frequently failed convergence",
                       "selection_basis": "Runtime/convergence and user budget only; no final labels or candidate scores used to choose the reduced feature designs",
                       "removed": "All unnormalized variants; designs 2/5/7; phase A C=5; phase B third representation and outermost C multipliers",
                       "caveat": "This is an amended adaptive study, not completion of the original 256-candidate grid. Exclusions do not prove statistical inferiority."})
        output.mkdir(parents=True, exist_ok=False)
        inventory = import_folds(source, output, data)
        for name in ("audit.json", "baseline.json", "training-membership.jsonl"):
            shutil.copy2(source / name, output / name)
        write_json(output / "reuse.json", inventory)
        write_json(output / "phase-a-candidates.json", candidates_a())
        plan = {"status": "prepared", "version": 2, "source_run": str(source.resolve()),
                "config": config, "config_sha256": file_digest(config_path), "runtime": runtime(),
                "implementation": implementation(), "protocol": amended,
                "input_files": data.audit["files"], "original_run_files": original_files,
                "prepared_files": {name: file_digest(output / name)
                                   for name in ("reuse.json", "audit.json", "baseline.json", "training-membership.jsonl", "phase-a-candidates.json")},
                "source_completed_folds": inventory["source_completed_folds"],
                "imported_folds": len(inventory["imported_folds"]), "test1_accessed": False, "test2_accessed": False}
        write_json(output / "plan.json", plan)
        report(output)
    return {"status": "prepared", "output": str(output), "candidates": len(candidates_a()),
            "folds_total": len(candidates_a()) * 3, "imported_folds": len(inventory["imported_folds"]),
            "new_fits": len(candidates_a()) * 3 - len(inventory["imported_folds"]), "jobs": config["jobs"],
            "fit_timeout_seconds": config["fit_timeout_seconds"]}


def validate_plan(config_path, output):
    config, plan = load_config(config_path), read_json(output / "plan.json")
    if (config != plan["config"] or file_digest(config_path) != plan["config_sha256"]
            or implementation() != plan["implementation"] or runtime() != plan["runtime"]):
        raise ValueError("Amended source, configuration or runtime changed; use a new run.")
    for name, expected in {**plan["input_files"], **plan["original_run_files"]}.items():
        if file_digest(Path(name)) != expected:
            raise ValueError(f"An immutable original input/result changed: {name}")
    for name, expected in plan["prepared_files"].items():
        if file_digest(output / name) != expected:
            raise ValueError(f"An amended prepared artifact changed: {name}")
    return config, plan


def report(output):
    plan = read_json(output / "plan.json")
    baseline = read_json(output / "baseline.json")
    lines = ["# Compute-bounded SVM investigation", "", "Status: amended preparation; experiments pending unless recorded below.",
             "", "## Original protocol and preserved work", "",
             f"Original run: `{plan['source_run']}`. Its {plan['source_completed_folds']} completed folds, original plans, source code and cache remain unchanged. This new run imports {plan['imported_folds']} exactly compatible folds; see reuse.json for hashes and exclusions.",
             "", "The user requested a compute-budget reduction after unnormalized folds took approximately 643–991 seconds, often reaching 20000 iterations without convergence. The revision uses runtime/convergence evidence only. It does not select variants using Test1, Test2 or candidate performance. Removed variants are not declared statistically inferior.",
             "", "## Data, calibration and evaluation protocol", "",
             "Original 16093 training rows, 2027 calibration rows and 1972 meta-validation rows remain group-disjoint. Labels come only from original train.xlsx. Original identifier-masked/character-cosine similarity components (.93), seeds and exact CV memberships are preserved. The old SVM, NorBERTo probabilities/temperature and ensemble default remain unchanged. No final workbook is opened by these commands.",
             "", f"The reproduced original 75/25 ensemble has meta-validation macro-F1 {baseline['A']['f1_macro']:.12f}. These are historically reused development observations, not an independent test.",
             "", "## Reduced search and resource use", "",
             "Phase A: 96 candidates / 288 folds, four feature families, designs 0/1/3/4, normalized L1/L2 only, C=.005/.05/.5 and both class weights. This retains word/character mixtures, word and character ngrams, binary/sublinear TF, min_df/max_df differences, class weighting and the exact baseline. Some original ngrams and larger feature caps are omitted under the amended budget.",
             "", "Phase B: at most 25 candidates / 125 folds, two distinct promising representations, three local C values, both class weights, a compatible hinge solver check and union branch weights. Three/five grouped folds and seeds 42/143 are unchanged. The original one-SE selection and all downstream calibration/weight/adoption rules remain unchanged in plan.json.",
             "", f"Four worker slots by default ({plan['config']['jobs']} configured), one numerical thread per fit. TF-IDF fits once per representation/training fold; the training matrix and validation transform are shared via read-only memory-mapped sparse arrays across C/class-weight/loss candidates. Validation text never participates in feature fitting. Each native solver runs in a separate subprocess with a {plan['config']['fit_timeout_seconds']}-second wall limit, including startup/loading/fit/scoring. A timeout excludes that candidate, produces no substitute label, and is terminal on resume. Only explicit new-run changes can retry it. Feature construction has no separate wall limit.",
             "", "Sparse caches belong only to this new run. They may grow through both phases; free disk was checked before preparation. Completed predictions and fold reports are checksummed. Existing imported fit/score times retain their original pipeline meaning; new fit times exclude shared feature construction, reported once in cache/phase-*/<representation>/fold-*/features.json. Do not sum duplicated cache build times over candidates."]
    for phase in ("a", "b"):
        path = output / f"phase-{phase}/search.json"
        lines += ["", f"## Phase {phase.upper()} results", ""]
        if not path.exists():
            lines.append("Pending execution. Imported phase A folds are partial development evidence only.")
            continue
        result = read_json(path)
        eligible = [t for t in result["trials"] if t["eligible"]]
        lines += [f"Complete. {len(result['trials'])} candidates, {len(eligible)} eligible; "
                  f"{sum(t['timeouts'] for t in result['trials'])} timed-out folds.", "",
                  "```json", json.dumps(result["selection"], indent=2), "```", "",
                  "| Candidate | Family | C | Weight | Loss | Mean F1 | SD F1 | Eligible | Timeouts |",
                  "|---|---|---:|---|---|---:|---:|---|---:|"]
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(["candidate_id", "family", "C", "class_weight", "loss", "mean_f1_macro", "std_f1_macro", "eligible", "timeouts"])
        for trial in sorted(result["trials"], key=lambda t: -(t["mean_f1_macro"] if t["mean_f1_macro"] is not None else -1)):
            c = trial["candidate"]
            values = [trial["candidate_id"], c["family"], c["C"], c["class_weight"], c["loss"],
                      trial["mean_f1_macro"], trial["std_f1_macro"], trial["eligible"], trial["timeouts"]]
            writer.writerow(values)
            lines.append("| " + " | ".join(str(v) for v in values) + " |")
        atomic_text(output / f"phase-{phase}/trials.csv", stream.getvalue())
    lines += ["", "## Pending calibration, weights, final evaluation and recommendation", "",
              "Final refit, nested calibration on 2027 rows, nested weight selection on 1972 rows, frozen A/B/C decisions and paired retrospective Test1 evaluation are still pending. Keep the original ensemble. The original 49.5844% Test1 reference has not been re-evaluated in this study. No improvement is claimed.",
              "", "Original methodological caveats remain: adaptive development CV and historical holdout/Test1 exposure; similarity groups do not guarantee semantic independence; bootstrap intervals condition on fitted predictions; the requested investigation does not satisfy the assignment's original Test2-before-Test1 sequence. Test2 stays outside development.", ""]
    atomic_text(output / "report.md", "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/svm-budgeted.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    stage = commands.add_parser("search")
    stage.add_argument("--phase", choices=("a", "b"), required=True)
    commands.add_parser("report")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.config, args.output_dir)
        else:
            with file_lock(args.output_dir / ".research.lock"):
                config, _ = validate_plan(args.config, args.output_dir)
                if args.command == "search":
                    data = load_data(base_config(config))
                    result = search(data, config, args.output_dir, args.phase)
                    validate_plan(args.config, args.output_dir)
                    report(args.output_dir)
                    result = {"status": result["status"], "phase": args.phase, "selection": result["selection"]}
                else:
                    report(args.output_dir)
                    result = {"status": "report_updated"}
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
