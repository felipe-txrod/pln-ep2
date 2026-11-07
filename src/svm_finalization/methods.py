"""Nested grouped calibration, bounded weight selection and paired cluster statistics."""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
from scipy.special import softmax
from scipy.stats import binomtest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import log_loss

from ensemble_classifier.soft_voting import (FrozenScores, apply_calibration, probabilities,
                                              probability_metrics, values)
from gpt_classifier.data import LABELS
from svm_research.search import classification_metrics, grouped_folds

WEIGHTS = (0., .25, .5, .75, 1.)


def predicted(p):
    return np.asarray(LABELS)[probabilities(p).argmax(axis=1)]


def describe_probabilities(labels, p):
    return {**probability_metrics(labels, p), **classification_metrics(labels, predicted(p))}


def fit_calibration(scores, labels, method):
    if method not in {"sigmoid", "isotonic"} or set(labels) != set(LABELS):
        raise ValueError("Calibration requires a supported method and every class.")
    scores = values(scores)
    estimator = CalibratedClassifierCV(FrozenEstimator(FrozenScores()), method=method)
    estimator.fit(scores, labels)
    calibrators = estimator.calibrated_classifiers_[0].calibrators
    if method == "sigmoid":
        result = {"method": "sigmoid_ovr", "a": [float(c.a_) for c in calibrators],
                  "b": [float(c.b_) for c in calibrators]}
    else:
        result = {"method": "isotonic_ovr", "x": [c.X_thresholds_.tolist() for c in calibrators],
                  "y": [c.y_thresholds_.tolist() for c in calibrators]}
    if not np.allclose(apply_calibrator(scores, result), estimator.predict_proba(scores), atol=1e-12, rtol=0):
        raise ValueError("Portable calibration differs from the fitted sklearn model.")
    return result


def apply_calibrator(scores, calibrator):
    scores = values(scores)
    if calibrator.get("method") != "isotonic_ovr":
        return apply_calibration(scores, calibrator)
    if len(calibrator["x"]) != 3 or len(calibrator["y"]) != 3:
        raise ValueError("Isotonic calibration needs exactly three class mappings.")
    result = np.empty_like(scores)
    for i, (x, y) in enumerate(zip(calibrator["x"], calibrator["y"])):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if (x.ndim != 1 or y.shape != x.shape or not len(x) or not np.isfinite(x).all()
                or not np.isfinite(y).all() or np.any(np.diff(x) <= 0) or np.any(np.diff(y) < 0)
                or np.any((y < 0) | (y > 1))):
            raise ValueError("Invalid monotonic isotonic mapping.")
        result[:, i] = np.interp(scores[:, i], x, y)
    total = result.sum(axis=1, keepdims=True)
    return probabilities(np.divide(result, total, out=np.full_like(result, 1 / 3), where=total != 0))


def choose_calibration(scores, labels, groups, *, seed=42):
    """Use paired inner-fold log-loss improvement over sigmoid's default."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    folds = grouped_folds(labels, groups, 3, seed)
    losses, assignments = {"sigmoid": [], "isotonic": []}, np.full(len(labels), -1)
    for fold, (train, valid) in enumerate(folds):
        assignments[valid] = fold
        for method in losses:
            fitted = fit_calibration(scores[train], labels[train], method)
            p = apply_calibrator(scores[valid], fitted)
            losses[method].append(float(log_loss(labels[valid], p, labels=list(LABELS))))
    delta = np.asarray(losses["sigmoid"]) - losses["isotonic"]
    mean, se = float(delta.mean()), float(delta.std(ddof=1) / np.sqrt(len(delta)))
    chosen = "isotonic" if mean > se + 1e-12 else "sigmoid"
    return chosen, {"chosen": chosen, "fold_log_loss": losses, "paired_log_loss_gain": mean,
                    "paired_gain_se": se, "fold_assignments": assignments.tolist(), "seed": seed,
                    "criterion": "Choose isotonic only when mean paired inner log-loss gain over sigmoid exceeds its SE; otherwise sigmoid"}


def nested_calibration(scores, labels, groups, *, seed=42):
    scores, labels, groups = values(scores), np.asarray(labels), np.asarray(groups)
    outer = grouped_folds(labels, groups, 5, seed)
    oof = {name: np.full_like(scores, np.nan) for name in ("sigmoid", "isotonic", "selection_policy")}
    choices, assignments = [], np.full(len(labels), -1)
    for fold, (train, valid) in enumerate(outer):
        chosen, inner = choose_calibration(scores[train], labels[train], groups[train], seed=seed + fold + 1)
        assignments[valid] = fold
        for method in ("sigmoid", "isotonic"):
            fitted = fit_calibration(scores[train], labels[train], method)
            oof[method][valid] = apply_calibrator(scores[valid], fitted)
        oof["selection_policy"][valid] = oof[chosen][valid]
        choices.append({"outer_fold": fold, "training_indices": train.tolist(), "validation_indices": valid.tolist(),
                        "shared_groups": 0, "inner_selection": inner,
                        "held_fold_metrics": describe_probabilities(labels[valid], oof[chosen][valid])})
    method, final_check = choose_calibration(scores, labels, groups, seed=seed)
    fitted = fit_calibration(scores, labels, method)
    metrics = {name: describe_probabilities(labels, probabilities(p)) for name, p in oof.items()}
    report = {"chosen_method": method, "outer_folds": choices, "outer_assignments": assignments.tolist(),
              "outer_method_counts": dict(Counter(c["inner_selection"]["chosen"] for c in choices)),
              "final_method_selection": final_check, "oof_metrics": metrics,
              "uncalibrated_softmax_diagnostic": describe_probabilities(labels, softmax(scores, axis=1)),
              "uncalibrated_note": "Softmax margins are a diagnostic surrogate, not native SVM probabilities.",
              "quality_note": "Selection-policy outer OOF measures nested method selection; per-method OOF are diagnostics. Final chosen method selection still adapts to all calibration data."}
    return fitted, report, oof


def mix(svm, norberto, weight):
    if type(weight) not in (int, float) or not 0 <= weight <= 1:
        raise ValueError("SVM weight must be a finite number between zero and one.")
    svm, norberto = probabilities(svm), probabilities(norberto)
    if svm.shape != norberto.shape:
        raise ValueError("Misaligned ensemble probabilities.")
    return probabilities(weight * svm + (1 - weight) * norberto)


def choose_weight(svm, norberto, labels, groups, *, seed=42):
    labels, groups = np.asarray(labels), np.asarray(groups)
    folds = grouped_folds(labels, groups, 3, seed)
    trials = []
    for weight in WEIGHTS:
        prediction = predicted(mix(svm, norberto, weight))
        scores = [classification_metrics(labels[valid], prediction[valid])["f1_macro"] for _, valid in folds]
        trials.append({"svm_weight": weight, "fold_f1_macro": scores, "mean_f1_macro": float(np.mean(scores)),
                       "std_f1_macro": float(np.std(scores, ddof=1))})
    best = max(trials, key=lambda t: (t["mean_f1_macro"], -t["std_f1_macro"], -abs(t["svm_weight"] - .75), -t["svm_weight"]))
    threshold = best["mean_f1_macro"] - best["std_f1_macro"] / np.sqrt(3)
    chosen = min((t for t in trials if t["mean_f1_macro"] >= threshold),
                 key=lambda t: (abs(t["svm_weight"] - .75), -t["mean_f1_macro"], t["svm_weight"]))
    return chosen["svm_weight"], {"trials": trials, "one_se_threshold": float(threshold),
                                    "chosen_svm_weight": chosen["svm_weight"], "seed": seed,
                                    "criterion": "Within one SE of best mean macro-F1, prefer weight closest to .75, then mean F1, then smaller SVM weight"}


def nested_weights(svm, norberto, labels, groups, *, seed=42):
    labels, groups = np.asarray(labels), np.asarray(groups)
    folds = grouped_folds(labels, groups, 5, seed)
    oof, assignments, choices = np.full_like(svm, np.nan), np.full(len(labels), -1), []
    for fold, (train, valid) in enumerate(folds):
        weight, inner = choose_weight(svm[train], norberto[train], labels[train], groups[train], seed=seed + fold + 1)
        oof[valid] = mix(svm[valid], norberto[valid], weight)
        assignments[valid] = fold
        choices.append({"outer_fold": fold, "training_indices": train.tolist(), "validation_indices": valid.tolist(),
                        "svm_weight": weight, "shared_groups": 0, "inner_selection": inner,
                        "held_fold_metrics": describe_probabilities(labels[valid], oof[valid])})
    final_weight, final_check = choose_weight(svm, norberto, labels, groups, seed=seed)
    stability = sum(c["svm_weight"] == final_weight for c in choices)
    return final_weight, {"final_svm_weight": final_weight, "outer_folds": choices,
                          "outer_assignments": assignments.tolist(),
                          "outer_weight_counts": dict(Counter(str(c["svm_weight"]) for c in choices)),
                          "final_weight_outer_matches": stability, "final_selection": final_check,
                          "selection_policy_oof_metrics": describe_probabilities(labels, oof),
                          "fixed_final_weight_diagnostic": describe_probabilities(labels, mix(svm, norberto, final_weight)),
                          "quality_note": "Use outer selection-policy OOF for the adoption screen; fixed final weight on all meta-validation rows is an adaptive diagnostic."}, probabilities(oof)


def _matrix_metrics(matrices):
    matrices = np.asarray(matrices)
    true = matrices.sum(axis=-1)
    predicted_count = matrices.sum(axis=-2)
    tp = np.diagonal(matrices, axis1=-2, axis2=-1)
    f1 = np.divide(2 * tp, true + predicted_count, out=np.zeros_like(tp, dtype=float), where=(true + predicted_count) != 0)
    return np.stack((tp.sum(axis=-1) / matrices.sum(axis=(-1, -2)), f1.mean(axis=-1)), axis=-1)


def paired_statistics(labels, predictions, groups, *, repeats=2000, seed=42):
    """Resample complete groups jointly for all predictions, preserving paired errors."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    predictions = {name: np.asarray(value) for name, value in predictions.items()}
    if set(labels) != set(LABELS) or len(groups) != len(labels) or not predictions:
        raise ValueError("Invalid paired-statistics inputs.")
    truth = np.asarray([LABELS.index(label) for label in labels])
    _, inverse = np.unique(groups, return_inverse=True)
    n_groups = int(inverse.max() + 1)
    if n_groups < 2:
        raise ValueError("Cluster bootstrap requires at least two groups.")
    names = list(predictions)
    encoded, group_matrices = {}, []
    for name in names:
        prediction = np.asarray(predictions[name])
        if prediction.shape != labels.shape or not set(prediction) <= set(LABELS):
            raise ValueError("Misaligned or invalid paired predictions.")
        encoded[name] = np.asarray([LABELS.index(label) for label in prediction])
        group_matrices.append(np.bincount(inverse * 9 + truth * 3 + encoded[name], minlength=n_groups * 9).reshape(n_groups, 9))
    grouped = np.stack(group_matrices)
    draws = np.empty((repeats, len(names), 2))
    generator = np.random.default_rng(seed)
    for start in range(0, repeats, 128):
        count = min(128, repeats - start)
        weights = generator.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=count)
        matrices = np.einsum("bg,mgk->bmk", weights, grouped).reshape(count, len(names), 3, 3)
        draws[start:start + count] = _matrix_metrics(matrices)
    absolute = {name: {metric: np.quantile(draws[:, i, j], [.025, .975]).tolist()
                       for j, metric in enumerate(("accuracy", "f1_macro"))} for i, name in enumerate(names)}
    pairs = {}
    for left, right in combinations(names, 2):
        a, b = predictions[left] == labels, predictions[right] == labels
        corrected, harmed = int(np.sum(~a & b)), int(np.sum(a & ~b))
        delta = draws[:, names.index(right)] - draws[:, names.index(left)]
        left_metrics, right_metrics = classification_metrics(labels, predictions[left]), classification_metrics(labels, predictions[right])
        pairs[f"{right}_minus_{left}"] = {
            "left": left, "right": right, "both_correct": int(np.sum(a & b)), "both_wrong": int(np.sum(~a & ~b)),
            "corrected": corrected, "harmed": harmed, "net_correct": corrected - harmed,
            "changed_predictions": int(np.sum(predictions[left] != predictions[right])),
            "per_class_corrected_harmed": {label: {"corrected": int(np.sum((labels == label) & ~a & b)),
                                                   "harmed": int(np.sum((labels == label) & a & ~b))} for label in LABELS},
            "delta": {metric: right_metrics[metric] - left_metrics[metric] for metric in ("accuracy", "f1_macro")},
            "delta_95_interval": {metric: np.quantile(delta[:, j], [.025, .975]).tolist()
                                   for j, metric in enumerate(("accuracy", "f1_macro"))},
            "mcnemar_exact_p_descriptive": float(binomtest(corrected, corrected + harmed, .5).pvalue) if corrected + harmed else 1.}
    return {"groups": n_groups, "repeats": repeats, "seed": seed, "absolute_95_intervals": absolute,
            "pairs": pairs, "caveat": "Paired percentile group bootstrap conditional on fitted predictions; it does not account for all training/adaptive-selection uncertainty. McNemar is descriptive and ignores within-group dependence."}


def adoption_screen(labels, left, right, groups, assignments):
    stats = paired_statistics(labels, {"left": predicted(left), "right": predicted(right)}, groups)
    pair = stats["pairs"]["right_minus_left"]
    deltas = [describe_probabilities(np.asarray(labels)[assignments == fold], right[assignments == fold])["f1_macro"]
              - describe_probabilities(np.asarray(labels)[assignments == fold], left[assignments == fold])["f1_macro"]
              for fold in range(5)]
    a, b = describe_probabilities(labels, left), describe_probabilities(labels, right)
    per_class = {label: b["classification_report"][label]["f1-score"] - a["classification_report"][label]["f1-score"]
                 for label in LABELS}
    checks = {"positive_bootstrap_lower_bound": pair["delta_95_interval"]["f1_macro"][0] > 0,
              "positive_in_at_least_four_folds": sum(d > 0 for d in deltas) >= 4,
              "no_class_f1_loss_above_002": min(per_class.values()) >= -.02}
    return {"passed": all(checks.values()), "checks": checks, "fold_f1_deltas": deltas,
            "class_f1_deltas": per_class, "paired_statistics": stats}


def complementarity(labels, left, right):
    labels, left, right = np.asarray(labels), np.asarray(left), np.asarray(right)
    a, b = left == labels, right == labels
    return {"both_correct": int(np.sum(a & b)), "left_only_correct": int(np.sum(a & ~b)),
            "right_only_correct": int(np.sum(~a & b)), "both_wrong": int(np.sum(~a & ~b)),
            "disagreements": int(np.sum(left != right)),
            "per_class": {label: {"rows": int(np.sum(labels == label)),
                                   "left_only_correct": int(np.sum((labels == label) & a & ~b)),
                                   "right_only_correct": int(np.sum((labels == label) & ~a & b))} for label in LABELS}}
