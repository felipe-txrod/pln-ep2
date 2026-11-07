"""Independently audit saved finalization metrics, nested roles and frozen decisions."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from ensemble_classifier.files import align, read_json, read_predictions
from ensemble_classifier.soft_voting import aligned_scores, apply_calibration
from gpt_classifier.data import LABELS, canonical, digest, file_digest, group, read_xlsx, write_json
from svm_budgeted.cli import base_config
from svm_finalization.methods import apply_calibrator
from svm_finalization.pipeline import load_frozen
from svm_research.data import load_data, location
from svm_research.search import build_pipeline, grouped_folds


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(left, right, message):
    require(np.allclose(left, right, atol=1e-11, rtol=0), message)


def independent_metrics(labels, prediction, p=None):
    truth = np.asarray([LABELS.index(label) for label in labels])
    encoded = np.asarray([LABELS.index(label) for label in prediction])
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (truth, encoded), 1)
    actual, predicted_count, tp = matrix.sum(axis=1), matrix.sum(axis=0), np.diag(matrix)
    precision = np.divide(tp, predicted_count, out=np.zeros(3), where=predicted_count != 0)
    recall = np.divide(tp, actual, out=np.zeros(3), where=actual != 0)
    f1 = np.divide(2 * tp, actual + predicted_count, out=np.zeros(3), where=(actual + predicted_count) != 0)
    result = {"accuracy": float(tp.sum() / len(labels)), "f1_macro": float(f1.mean()),
              "confusion_matrix": matrix.tolist(),
              "classification_report": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                         "f1-score": float(f1[i]), "support": int(actual[i])} for i, label in enumerate(LABELS)}}
    if p is not None:
        p = np.asarray(p)
        require(p.shape == (len(labels), 3) and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), "Invalid probabilities.")
        close(p.sum(axis=1), 1, "Probability normalization mismatch.")
        epsilon = np.finfo(p.dtype).eps
        result["log_loss"] = float(-np.log(np.clip(p[np.arange(len(labels)), truth], epsilon, 1 - epsilon)).mean())
        one_hot = np.eye(3)[truth]
        result["multiclass_brier"] = float(((p - one_hot) ** 2).sum(axis=1).mean())
        confidence, correct = p.max(axis=1), truth == encoded
        bins = np.minimum((confidence * 10).astype(int), 9)
        result["confidence_ece_10_bins"] = sum(float(np.sum(bins == i)) / len(labels)
            * abs(float(correct[bins == i].mean()) - float(confidence[bins == i].mean()))
            for i in range(10) if np.any(bins == i))
    return result


def verify_metrics(labels, prediction, report, p=None):
    measured = independent_metrics(labels, prediction, p)
    require(measured["confusion_matrix"] == report["confusion_matrix"], "Confusion matrix mismatch.")
    for name, value in measured.items():
        if name not in {"classification_report", "confusion_matrix"}:
            close(value, report[name], f"{name} mismatch.")
    for label in LABELS:
        for name, value in measured["classification_report"][label].items():
            close(value, report["classification_report"][label][name], "Per-class metric mismatch.")
    return measured


def prediction(p):
    return np.asarray(LABELS)[p.argmax(axis=1)]


def blend(svm, norberto, weight):
    p = weight * svm + (1 - weight) * norberto
    return p / p.sum(axis=1, keepdims=True)


def verify_calibration(data, scores, output):
    indices = data.calibration
    labels = np.asarray([data.holdout[i].label for i in indices])
    groups = data.holdout_groups[indices]
    saved = read_json(output / "calibration.json")
    report = saved["report"]
    with np.load(output / "calibration-oof.npz", allow_pickle=False) as values:
        require(np.array_equal(values["excel_rows"], [data.holdout[i].excel_row for i in indices]), "Calibration row order mismatch.")
        oof = {name: values[name].copy() for name in ("sigmoid", "isotonic", "selection_policy")}
    assignments = np.full(len(labels), -1)
    choices = []
    for fold, (train, valid) in enumerate(grouped_folds(labels, groups, 5, 42)):
        outer = report["outer_folds"][fold]
        require(outer["training_indices"] == train.tolist() and outer["validation_indices"] == valid.tolist(), "Calibration outer rows changed.")
        require(not set(groups[train]) & set(groups[valid]), "Calibration outer group leakage.")
        assignments[valid] = fold
        inner = outer["inner_selection"]
        expected = np.full(len(train), -1)
        for inner_fold, (fit, held) in enumerate(grouped_folds(labels[train], groups[train], 3, 43 + fold)):
            require(not set(groups[train[fit]]) & set(groups[train[held]]), "Calibration inner group leakage.")
            expected[held] = inner_fold
        require(inner["fold_assignments"] == expected.tolist(), "Calibration inner roles mismatch.")
        difference = np.asarray(inner["fold_log_loss"]["sigmoid"]) - inner["fold_log_loss"]["isotonic"]
        chosen = "isotonic" if difference.mean() > difference.std(ddof=1) / np.sqrt(3) + 1e-12 else "sigmoid"
        require(chosen == inner["chosen"], "Inner calibration decision mismatch.")
        close(oof["selection_policy"][valid], oof[chosen][valid], "Calibration OOF policy mismatch.")
        verify_metrics(labels[valid], prediction(oof[chosen][valid]), outer["held_fold_metrics"], oof[chosen][valid])
        choices.append(chosen)
    require(report["outer_assignments"] == assignments.tolist(), "Calibration outer assignment mismatch.")
    require(report["outer_method_counts"] == dict(Counter(choices)), "Calibration stability count mismatch.")
    final = report["final_method_selection"]
    delta = np.asarray(final["fold_log_loss"]["sigmoid"]) - final["fold_log_loss"]["isotonic"]
    method = "isotonic" if delta.mean() > delta.std(ddof=1) / np.sqrt(3) + 1e-12 else "sigmoid"
    require(method == report["chosen_method"], "Final calibration method mismatch.")
    for name, p in oof.items():
        verify_metrics(labels, prediction(p), report["oof_metrics"][name], p)
    require(saved["calibrator"]["method"] == f"{method}_ovr", "Final calibrator type mismatch.")
    return saved["calibrator"], report


def independent_weight(svm, norberto, labels, groups, report, seed):
    trials = []
    for index, weight in enumerate((0., .25, .5, .75, 1.)):
        predicted = prediction(blend(svm, norberto, weight))
        f1 = [independent_metrics(labels[held], predicted[held])["f1_macro"] for _, held in grouped_folds(labels, groups, 3, seed)]
        saved = report["trials"][index]
        require(saved["svm_weight"] == weight, "Weight grid changed.")
        close(saved["fold_f1_macro"], f1, "Inner weight fold score mismatch.")
        trials.append((weight, float(np.mean(f1)), float(np.std(f1, ddof=1))))
    best = max(trials, key=lambda t: (t[1], -t[2], -abs(t[0] - .75), -t[0]))
    threshold = best[1] - best[2] / np.sqrt(3)
    chosen = min((t for t in trials if t[1] >= threshold), key=lambda t: (abs(t[0] - .75), -t[1], t[0]))[0]
    close(threshold, report["one_se_threshold"], "Weight threshold mismatch.")
    require(chosen == report["chosen_svm_weight"], "Weight decision mismatch.")
    return chosen


def verify_paired_counts(labels, predictions, report):
    labels = np.asarray(labels)
    for pair in report["pairs"].values():
        left, right = predictions[pair["left"]], predictions[pair["right"]]
        a, b = left == labels, right == labels
        corrected, harmed = int(np.sum(~a & b)), int(np.sum(a & ~b))
        require((corrected, harmed) == (pair["corrected"], pair["harmed"]), "Paired errors mismatch.")
        n = corrected + harmed
        p = min(1., 2 * sum(math.comb(n, i) for i in range(min(corrected, harmed) + 1)) / 2 ** n) if n else 1.
        close(p, pair["mcnemar_exact_p_descriptive"], "Independent binomial sum mismatch.")
        close(independent_metrics(labels, right)["f1_macro"] - independent_metrics(labels, left)["f1_macro"], pair["delta"]["f1_macro"], "Paired F1 delta mismatch.")
        for label in LABELS:
            mask = labels == label
            require(pair["per_class_corrected_harmed"][label] == {"corrected": int(np.sum(mask & ~a & b)), "harmed": int(np.sum(mask & a & ~b))}, "Per-class paired error mismatch.")


def verify_development(output, frozen):
    plan = read_json(output / "plan.json")
    data = load_data(base_config(plan["search_config_value"]))
    refit = read_json(output / "refit.json")
    require(refit["training_rows_identity"] == digest(canonical([(r.excel_row, digest(r.text), r.label) for r in data.training])), "Final refit training membership mismatch.")
    model = joblib.load(output / "model.joblib")
    expected = build_pipeline(frozen["candidate"]["candidate"])
    require(model.named_steps["classifier"].get_params() == expected.named_steps["classifier"].get_params(), "Final classifier recipe differs from selection.")
    require(tuple(model.classes_) == LABELS and model.named_steps["classifier"].coef_.shape[1] == refit["feature_count"], "Final classifier dimensions mismatch.")
    actual_vectors = dict(model.named_steps["features"].transformer_list)
    for name, vector in expected.named_steps["features"].transformer_list:
        require(actual_vectors[name].get_params() == vector.get_params(), "Final TF-IDF recipe mismatch.")
    require(frozen["original_baseline"] == data.baseline, "Original ensemble contract changed.")
    with np.load(output / "holdout-margins.npz", allow_pickle=False) as values:
        require(np.array_equal(values["excel_rows"], [r.excel_row for r in data.holdout]), "Reserved-score row order mismatch.")
        require(np.array_equal(values["text_sha256"], [digest(r.text) for r in data.holdout]), "Reserved-score text mismatch.")
        margins = values["margins"].copy()
    calibrator, calibration_report = verify_calibration(data, margins, output)
    require(calibrator == frozen["svm_calibrator"], "Frozen calibrator changed.")
    selected = data.selection
    labels = np.asarray([data.holdout[i].label for i in selected])
    groups = data.holdout_groups[selected]
    old = apply_calibration(data.old_margins, data.baseline["calibration"]["svm"])[selected]
    new = apply_calibrator(margins, calibrator)[selected]
    norberto = data.norberto[selected]
    dev = read_json(output / "development.json")
    weights = dev["weights"]
    policy, assignments, choices = np.empty_like(new), np.full(len(labels), -1), []
    for fold, (train, valid) in enumerate(grouped_folds(labels, groups, 5, 42)):
        outer = weights["outer_folds"][fold]
        require(outer["training_indices"] == train.tolist() and outer["validation_indices"] == valid.tolist(), "Outer weight rows changed.")
        require(not set(groups[train]) & set(groups[valid]), "Outer weight group leakage.")
        chosen = independent_weight(new[train], norberto[train], labels[train], groups[train], outer["inner_selection"], 43 + fold)
        require(chosen == outer["svm_weight"], "Outer weight selection mismatch.")
        policy[valid], assignments[valid] = blend(new[valid], norberto[valid], chosen), fold
        verify_metrics(labels[valid], prediction(policy[valid]), outer["held_fold_metrics"], policy[valid])
        choices.append(chosen)
    chosen = independent_weight(new, norberto, labels, groups, weights["final_selection"], 42)
    require(chosen == weights["final_svm_weight"] == frozen["weights"]["C"]["svm"], "Frozen C weight mismatch.")
    require(weights["final_weight_outer_matches"] == sum(w == chosen for w in choices), "Weight stability mismatch.")
    require(weights["outer_assignments"] == assignments.tolist(), "Meta-validation fold assignment mismatch.")
    matrices = {"A": blend(old, norberto, .75), "B": blend(new, norberto, .75),
                "C_fixed_diagnostic": blend(new, norberto, chosen), "C_policy_oof": policy,
                "svm_original_calibrated": old, "svm_optimized_calibrated": new, "norberto_unchanged": norberto}
    with np.load(output / "meta-validation.npz", allow_pickle=False) as arrays:
        require(np.array_equal(arrays["excel_rows"], [data.holdout[i].excel_row for i in selected]), "Meta-validation row order mismatch.")
        for name, p in matrices.items():
            close(p, arrays[name], "Stored meta-validation probabilities mismatch.")
            verify_metrics(labels, prediction(arrays[name]), dev["metrics"][name], arrays[name])
    diagnostics = {name: prediction(matrices[name]) for name in ("A", "B", "C_fixed_diagnostic", "C_policy_oof")}
    verify_paired_counts(labels, diagnostics, dev["paired_diagnostics"])
    for name, left, right in (("B_over_A_screen", "A", "B"), ("C_policy_over_B_screen", "B", "C_policy_oof")):
        screen = dev[name]
        deltas = [independent_metrics(labels[assignments == i], diagnostics[right][assignments == i])["f1_macro"]
                  - independent_metrics(labels[assignments == i], diagnostics[left][assignments == i])["f1_macro"] for i in range(5)]
        close(deltas, screen["fold_f1_deltas"], "Adoption fold deltas mismatch.")
        checks = screen["checks"]
        require(checks["positive_in_at_least_four_folds"] == (sum(d > 0 for d in deltas) >= 4), "Adoption stability mismatch.")
        require(checks["positive_bootstrap_lower_bound"] == (screen["paired_statistics"]["pairs"]["right_minus_left"]["delta_95_interval"]["f1_macro"][0] > 0), "Bootstrap adoption mismatch.")
        require(screen["passed"] == all(checks.values()), "Adoption Boolean mismatch.")
    choice = "A"
    if dev["B_over_A_screen"]["passed"]:
        choice = "C" if dev["C_policy_over_B_screen"]["passed"] and sum(w == chosen for w in choices) >= 3 else "B"
    require(choice == dev["recommended_configuration"] == frozen["recommended_configuration"], "Frozen recommendation mismatch.")
    return {"status": "passed", "training_rows": len(data.training), "calibration_rows": len(data.calibration),
            "meta_rows": len(selected), "calibration_method": calibration_report["chosen_method"],
            "final_svm_weight": chosen, "outer_weights": choices, "recommendation": choice,
            "nested_group_roles_verified": True, "probability_and_class_metrics_reconstructed": True,
            "training_membership_and_recipe_verified": True, "paired_error_counts_verified": True,
            "test1_accessed": False, "test2_accessed": False}


def verify_final(output, frozen):
    manifest = read_json(output / "test1/prediction-manifest.json")
    report = read_json(output / "test1/metrics.json")
    plan = read_json(output / "plan.json")
    require(manifest["frozen_sha256"] == file_digest(output / "frozen.json") == report["frozen_sha256"], "Final freeze identity mismatch.")
    require(report["prediction_manifest_sha256"] == file_digest(output / "test1/prediction-manifest.json"), "Prediction manifest changed.")
    gold, source = read_xlsx(location(plan["test1_gold"]), labeled=True)
    require(len(gold) == 900 and source["sha256"] == plan["test1_expected_gold_sha256"] == report["gold"]["sha256"], "Gold provenance mismatch.")
    labels = np.asarray([r.label for r in gold])
    predictions = {}
    for name, expected in manifest["files"].items():
        path = output / "test1" / name
        require(file_digest(path) == expected, "Final prediction checksum changed.")
        rows = read_predictions(path)
        align(gold, rows)
        key = Path(name).stem
        predictions[key] = np.asarray([r["label"] for r in rows])
        p = aligned_scores(rows, "probabilities") if "probabilities" in rows[0] else None
        verify_metrics(labels, predictions[key], report["metrics"][key], p)
    verify_paired_counts(labels, predictions, report["paired_statistics"])
    require(report["recommended_configuration"] == frozen["recommended_configuration"] and report["recommendation_changed_after_test1"] is False, "Recommendation changed after final labels.")
    require(report["gold_read_after_all_predictions"] is True, "Late-gold provenance flag missing.")
    latest_prediction = max((output / "test1" / name).stat().st_mtime_ns for name in manifest["files"])
    require(latest_prediction <= (output / "test1/prediction-manifest.json").stat().st_mtime_ns <= (output / "test1/metrics.json").stat().st_mtime_ns, "Persisted prediction/evaluation order mismatch.")
    return {"status": "passed", "rows": 900, "prediction_sets": len(predictions), "recommendation": frozen["recommended_configuration"],
            "metric_and_paired_counts_independently_reconstructed": True, "frozen_before_prediction_verified": True,
            "normalized_text_groups": len({group(r.text) for r in gold}), "test2_accessed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=("development", "test1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError("Preserve the previous verification; use a new output filename.")
        frozen = load_frozen(args.run)
        result = verify_development(args.run, frozen) if args.stage == "development" else verify_final(args.run, frozen)
        result.update(verifier_sha256=file_digest(Path(__file__)), frozen_sha256=file_digest(args.run / "frozen.json"))
        write_json(args.output, result, replace=False)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
