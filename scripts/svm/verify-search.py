"""Independently verify saved grouped SVM predictions without refitting models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gpt_classifier.data import LABELS, canonical, digest, file_digest, read_jsonl, write_json
from ensemble_classifier.files import read_json
from svm_budgeted.cli import base_config, validate_plan
from svm_budgeted.engine import candidates_a, candidates_b
from svm_research.data import load_data
from svm_research.search import baseline_candidate, candidate_id, grouped_folds


def independently_score(truth, predictions):
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (truth, predictions), 1)
    tp = np.diag(matrix).astype(float)
    actual, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted != 0)
    recall = np.divide(tp, actual, out=np.zeros(3), where=actual != 0)
    f1 = np.divide(2 * tp, actual + predicted, out=np.zeros(3), where=actual + predicted != 0)
    return {"accuracy": float(tp.sum() / matrix.sum()), "f1_macro": float(f1.mean()),
            "confusion_matrix": matrix.tolist(),
            "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                                   "f1-score": float(f1[i]), "support": int(actual[i])}
                          for i, label in enumerate(LABELS)}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(config_path, root, phase):
    config, original_plan = validate_plan(config_path, root)
    data = load_data(base_config(config))
    directory = root / f"phase-{phase}"
    plan, search = read_json(directory / "plan.json"), read_json(directory / "search.json")
    candidates = candidates_a() if phase == "a" else candidates_b(read_json(root / "phase-a/search.json")["trials"])
    require(plan["candidates"] == candidates and plan["candidate_count"] == len(candidates), "Candidate plan mismatch.")
    require(search["status"] == "completed" and search["plan_sha256"] == file_digest(directory / "plan.json"), "Incomplete or changed phase plan.")
    require(search["test1_accessed"] is False and search["test2_accessed"] is False, "Final-set access flag differs.")
    labels = np.asarray([LABELS.index(r.label) for r in data.training])
    rows = np.asarray([r.excel_row for r in data.training])
    seed = config["seed"] + (101 if phase == "b" else 0)
    folds = grouped_folds(np.asarray(LABELS)[labels], data.groups, config[f"phase_{phase}_folds"], seed)
    require(plan["seed"] == seed and plan["folds"] == len(folds), "Fold count or seed mismatch.")
    assignments = np.full(len(rows), -1)
    for fold, (train, valid) in enumerate(folds):
        require(not set(data.groups[train]) & set(data.groups[valid]), "Shared similarity component.")
        assignments[valid] = fold
    require(read_jsonl(directory / "folds.jsonl") == [
        {"excel_row": int(row), "group_id": str(g), "fold": int(f)}
        for row, g, f in zip(rows, data.groups, assignments)], "Saved fold membership mismatch.")
    identifiers = {candidate_id(c) for c in candidates}
    require(len(search["trials"]) == len(candidates)
            and {t["candidate_id"] for t in search["trials"]} == identifiers, "Duplicate or missing trial.")
    verified, oof_predictions, hashes = [], {}, {}
    timeout_count = warning_count = 0
    for trial in search["trials"]:
        candidate = trial["candidate"]
        identifier = candidate_id(candidate)
        require(candidate in candidates and identifier == trial["candidate_id"], "Trial configuration mismatch.")
        require(len(trial["folds"]) == len(folds), "Missing fold report.")
        scores, features = [], []
        predictions = np.full(len(rows), -1, dtype=int)
        eligible = True
        for fold, (train, valid) in enumerate(folds):
            path = directory / "trials" / identifier / f"fold-{fold}.json"
            report = read_json(path)
            require(report == trial["folds"][fold], "Aggregate differs from saved fold report.")
            require(report["identity"] == {"candidate": candidate, "seed": seed, "fold": fold,
                    "training_ids": digest(canonical(rows[train].tolist())),
                    "validation_ids": digest(canonical(rows[valid].tolist()))}, "Fold identity mismatch.")
            hashes[str(path.relative_to(root))] = file_digest(path)
            if report["status"] == "timeout":
                require(report["metrics"] is None and report["predictions_sha256"] is None, "Timeout contains fabricated metrics.")
                timeout_count += 1
                eligible = False
                continue
            require(report["status"] == "completed" and report["shared_groups"] == 0, "Failed or leaking fold.")
            require(report["training_rows"] == len(train) and report["validation_rows"] == len(valid), "Fold row count mismatch.")
            array_path = path.with_suffix(".npz")
            require(file_digest(array_path) == report["predictions_sha256"], "Prediction checksum mismatch.")
            hashes[str(array_path.relative_to(root))] = file_digest(array_path)
            with np.load(array_path, allow_pickle=False) as array:
                predicted = array["predictions"].copy()
                require(np.array_equal(array["excel_rows"], rows[valid]) and predicted.shape == (len(valid),)
                        and predicted.dtype.kind in "iu" and np.all((0 <= predicted) & (predicted <= 2)), "Invalid prediction coverage/order/classes.")
            predictions[valid] = predicted
            measured = independently_score(labels[valid], predicted)
            require(measured["confusion_matrix"] == report["metrics"]["confusion_matrix"], "Confusion matrix mismatch.")
            for name in ("accuracy", "f1_macro"):
                require(np.isclose(measured[name], report["metrics"][name], atol=1e-12, rtol=0), f"{name} mismatch.")
            for label in LABELS:
                for name, value in measured["per_class"][label].items():
                    require(np.isclose(value, report["metrics"]["classification_report"][label][name], atol=1e-12, rtol=0), "Per-class metric mismatch.")
            scores.append(measured["f1_macro"])
            features.append(report["feature_count"])
            if report["convergence_warnings"]:
                eligible = False
                warning_count += 1
        require(eligible == trial["eligible"], "Eligibility mismatch.")
        if len(scores) == len(folds):
            mean, std = float(np.mean(scores)), float(np.std(scores, ddof=1))
            require(np.isclose(mean, trial["mean_f1_macro"], atol=1e-12, rtol=0)
                    and np.isclose(std, trial["std_f1_macro"], atol=1e-12, rtol=0), "CV mean or dispersion mismatch.")
            require(np.isclose(np.mean(features), trial["mean_feature_count"], atol=1e-12, rtol=0), "Feature count mismatch.")
            require(np.all(predictions >= 0), "Incomplete OOF predictions.")
            oof_predictions[identifier] = predictions
            verified.append({"candidate_id": identifier, "candidate": candidate, "eligible": eligible,
                             "mean_f1_macro": mean, "std_f1_macro": std, "fold_f1_macro": scores,
                             "mean_feature_count": float(np.mean(features)),
                             "pooled_oof": independently_score(labels, predictions)})
    eligible = [t for t in verified if t["eligible"]]
    best = max(eligible, key=lambda t: (t["mean_f1_macro"], -t["std_f1_macro"], t["candidate_id"]))
    threshold = best["mean_f1_macro"] - best["std_f1_macro"] / np.sqrt(len(folds))
    winner = min((t for t in eligible if t["mean_f1_macro"] >= threshold),
                 key=lambda t: (t["candidate"]["C"], t["mean_feature_count"],
                                t["candidate"]["loss"] != "squared_hinge", t["candidate"]["class_weight"] is not None,
                                -t["mean_f1_macro"], t["candidate_id"]))
    require(winner["candidate_id"] == search["selection"]["candidate_id"], "One-SE selection mismatch.")
    baseline = next(t for t in verified if t["candidate_id"] == candidate_id(baseline_candidate()))
    pairs = {}
    for name, trial in (("highest_mean", best), ("one_se_selection", winner)):
        old, new = oof_predictions[baseline["candidate_id"]], oof_predictions[trial["candidate_id"]]
        pairs[name] = {"mean_f1_delta": trial["mean_f1_macro"] - baseline["mean_f1_macro"],
                       "fold_f1_delta": (np.asarray(trial["fold_f1_macro"]) - baseline["fold_f1_macro"]).tolist(),
                       "corrected": int(np.sum((new == labels) & (old != labels))),
                       "harmed": int(np.sum((new != labels) & (old == labels)))}
    preview = candidates_b(search["trials"]) if phase == "a" else None
    return {"status": "passed", "phase": phase, "search_sha256": file_digest(directory / "search.json"),
            "verifier_sha256": file_digest(Path(__file__)), "candidates": len(candidates),
            "folds": len(candidates) * len(folds), "eligible_candidates": len(eligible),
            "timeout_folds": timeout_count, "convergence_warning_folds": warning_count,
            "source_fingerprints_verified": True, "prediction_metrics_independently_reconstructed": True,
            "fold_and_group_coverage_verified": True, "one_se_selection_verified": True,
            "baseline": baseline, "highest_mean": best, "one_se_selection": winner,
            "paired_development_diagnostics": pairs, "verified_files": hashes,
            "next_phase_candidates": len(preview) if preview is not None else None,
            "phase_b_preview": preview,
            "manifest_to_result_seconds": (directory / "search.json").stat().st_mtime - (directory / "plan.json").stat().st_mtime,
            "timing_caveat": "Artifact timestamp window, including any pauses/resumes; imported folds were computed earlier.",
            "test1_accessed": False, "test2_accessed": False,
            "limitation": "Development diagnostics after adaptive search; no final-set improvement or statistical superiority claimed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/svm-budgeted.json"))
    parser.add_argument("--phase", choices=("a", "b"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError("Use a new verification output file.")
        result = verify(args.config, args.run, args.phase)
        write_json(args.output, result, replace=False)
        print(json.dumps({k: result[k] for k in ("status", "candidates", "folds", "eligible_candidates", "timeout_folds",
                                                  "convergence_warning_folds", "next_phase_candidates", "manifest_to_result_seconds")}))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
