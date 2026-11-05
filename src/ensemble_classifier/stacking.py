"""A small regularized meta-classifier over disjoint, frozen probabilities."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from gpt_classifier.data import LABELS
from .soft_voting import MODELS, probabilities

FEATURES = tuple(f"{model}:{label}" for model in MODELS for label in LABELS)
CANDIDATES = (.01, .1, 1., 10.)
FOLDS = 5
SEED = 42


def feature_matrix(matrices):
    if set(matrices) != set(MODELS):
        raise ValueError("Stacking requires exactly svm, rl and norberto probabilities.")
    parts = [probabilities(matrices[model]) for model in MODELS]
    if len({p.shape for p in parts}) != 1:
        raise ValueError("Stacking member rows are not aligned.")
    return np.ascontiguousarray(np.concatenate(parts, axis=1))


def meta_probabilities(matrices, model):
    if model.get("classes") != list(LABELS) or model.get("features") != list(FEATURES):
        raise ValueError("Meta-model class/feature order differs from the frozen contract.")
    coef, intercept = np.asarray(model["coef"], dtype=float), np.asarray(model["intercept"], dtype=float)
    if (coef.shape != (3, 9) or intercept.shape != (3,)
            or not np.isfinite(coef).all() or not np.isfinite(intercept).all()):
        raise ValueError("Invalid multinomial meta-model parameters.")
    return probabilities(softmax(feature_matrix(matrices) @ coef.T + intercept, axis=1))


def select_regularization(trials):
    """Fixed one-standard-error rule, preferring stronger regularization."""
    best = max(trials, key=lambda t: (t["mean_f1_macro"], -t["C"]))
    threshold = best["mean_f1_macro"] - best["standard_error_f1_macro"]
    chosen = min((t for t in trials if t["mean_f1_macro"] >= threshold - 1e-12), key=lambda t: t["C"])
    return chosen["C"], threshold


def _fit(X, y, C):
    estimator = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=2000,
                                   tol=1e-8, random_state=SEED, class_weight=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(X, y)
    return estimator


def fit_meta(matrices, labels, groups):
    X, y, groups = feature_matrix(matrices), np.asarray(labels), np.asarray(groups)
    if (len(X) != len(y) or len(y) != len(groups) or set(y) != set(LABELS)
            or len(set(groups)) < FOLDS):
        raise ValueError("Invalid meta-training labels/groups.")
    splits = list(StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED).split(X, y, groups))
    fold_ids = np.full(len(y), -1, dtype=int)
    folds = []
    for fold, (train, valid) in enumerate(splits):
        if (set(groups[train]) & set(groups[valid]) or set(y[train]) != set(LABELS)
                or set(y[valid]) != set(LABELS) or np.any(fold_ids[valid] != -1)):
            raise ValueError("Meta-CV is not class-complete and group-disjoint.")
        fold_ids[valid] = fold
        folds.append({"fold": fold, "training_rows": len(train), "validation_rows": len(valid),
                      "shared_groups": 0, "validation_class_counts": {
                          label: int(sum(y[valid] == label)) for label in LABELS}})
    if np.any(fold_ids == -1):
        raise ValueError("Incomplete out-of-fold coverage.")
    trials, out_of_fold = [], {}
    for C in CANDIDATES:
        scores, accuracies = [], []
        oof = np.empty((len(y), 3))
        for train, valid in splits:
            estimator = _fit(X[train], y[train], C)
            order = [list(estimator.classes_).index(label) for label in LABELS]
            oof[valid] = estimator.predict_proba(X[valid])[:, order]
            predicted = np.asarray(LABELS)[oof[valid].argmax(axis=1)]
            scores.append(float(f1_score(y[valid], predicted, labels=list(LABELS), average="macro", zero_division=0)))
            accuracies.append(float(accuracy_score(y[valid], predicted)))
        out_of_fold[C] = oof
        trials.append({"C": C, "fold_f1_macro": scores, "fold_accuracy": accuracies,
                       "mean_f1_macro": float(np.mean(scores)),
                       "standard_error_f1_macro": float(np.std(scores, ddof=1) / np.sqrt(FOLDS)),
                       "mean_accuracy": float(np.mean(accuracies))})
    selected, threshold = select_regularization(trials)
    estimator = _fit(X, y, selected)
    order = [list(estimator.classes_).index(label) for label in LABELS]
    model = {"classes": list(LABELS), "features": list(FEATURES), "coef": estimator.coef_[order].tolist(),
             "intercept": estimator.intercept_[order].tolist(), "C": selected,
             "solver": "lbfgs", "penalty": "l2", "max_iter": 2000, "tol": 1e-8,
             "class_weight": None, "n_iter": estimator.n_iter_.tolist()}
    if not np.allclose(meta_probabilities(matrices, model), estimator.predict_proba(X)[:, order], atol=1e-12, rtol=0):
        raise ValueError("Exported meta probabilities differ from LogisticRegression.predict_proba.")
    search = {"candidates": list(CANDIDATES), "folds": folds, "seed": SEED, "trials": trials,
              "selection": "mean macro-F1; one-standard-error rule selects smallest eligible C",
              "one_standard_error_threshold": threshold, "selected_C": selected,
              "diagnostic_only": "CV was used for selection, not an independent performance estimate."}
    return model, search, fold_ids, out_of_fold[selected]
