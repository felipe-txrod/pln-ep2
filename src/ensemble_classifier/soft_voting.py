"""Calibration and soft voting over frozen scores; base estimators are never fitted."""
from __future__ import annotations

from itertools import product

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

from gpt_classifier.data import LABELS

MODELS = ("svm", "rl", "norberto")
QWEN_MODELS = (*MODELS, "qwen")


def model_order(members):
    """Keep the legacy three-model contract and the selected four-model extension explicit."""
    names = tuple(members)
    if len(set(names)) != len(names) or set(names) not in (set(MODELS), set(QWEN_MODELS)):
        raise ValueError("Members must be svm, rl, norberto, optionally with the selected qwen.")
    return QWEN_MODELS if "qwen" in names else MODELS


def values(matrix):
    original = np.asarray(matrix)
    if original.dtype.kind not in "fiu":
        raise ValueError("Scores must be numeric, not booleans or strings.")
    result = original.astype(float)
    if result.ndim != 2 or result.shape[1] != 3 or not len(result) or not np.isfinite(result).all():
        raise ValueError("Expected finite nonempty scores with three aligned classes.")
    return result


def probabilities(matrix):
    result = values(matrix)
    if (np.any(result < 0) or np.any(result > 1)
            or not np.allclose(result.sum(axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError("Invalid class probabilities.")
    return result / result.sum(axis=1, keepdims=True)


def aligned_scores(rows, field, class_order=LABELS):
    if len(class_order) != 3 or set(class_order) != set(LABELS):
        raise ValueError("Missing, duplicate or unknown class names.")
    output = []
    for row in rows:
        raw = row.get(field)
        if isinstance(raw, dict) and set(raw) == set(LABELS):
            output.append([raw[label] for label in LABELS])
        elif isinstance(raw, list) and len(raw) == 3:
            output.append([raw[list(class_order).index(label)] for label in LABELS])
        else:
            raise ValueError(f"Missing or incompatible {field}; hard labels are insufficient for soft voting.")
    return probabilities(output) if field == "probabilities" else values(output)


class FrozenScores(ClassifierMixin, BaseEstimator):
    """Expose saved decision margins through the public frozen calibration API."""
    def __init__(self):
        self.classes_ = np.asarray(LABELS)

    def fit(self, X, y):
        raise RuntimeError("Refitting a base classifier is prohibited.")

    def decision_function(self, X):
        return values(X)

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]

    def __sklearn_is_fitted__(self):
        return True


def apply_calibration(scores, calibration):
    scores = values(scores)
    method = calibration.get("method")
    if method == "sigmoid_ovr":
        a, b = np.asarray(calibration["a"], dtype=float), np.asarray(calibration["b"], dtype=float)
        if a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all():
            raise ValueError("Invalid sigmoid parameters.")
        mapped = expit(-(scores * a + b))
        if np.any(mapped.sum(axis=1) == 0):
            raise ValueError("Sigmoid normalization underflow.")
        return probabilities(mapped / mapped.sum(axis=1, keepdims=True))
    if method == "identity":
        return probabilities(scores)
    if method != "temperature":
        raise ValueError("Unknown calibration method.")
    temperature = calibration["temperature"]
    if type(temperature) not in (float, int) or not np.isfinite(temperature) or not .25 <= temperature <= 10:
        raise ValueError("Temperature outside the frozen bounds.")
    return softmax(np.log(np.clip(probabilities(scores), 1e-12, 1)) / temperature, axis=1)


def _temperature(p, y):
    result = minimize_scalar(lambda log_t: log_loss(y, softmax(np.log(np.clip(p, 1e-12, 1)) / np.exp(log_t), axis=1),
                                                  labels=list(LABELS)),
                             bounds=(np.log(.25), np.log(10)), method="bounded",
                             options={"xatol": 1e-8, "maxiter": 200})
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("Temperature calibration did not converge.")
    return {"method": "temperature", "temperature": float(np.exp(result.x))}


def fit_calibrations(raw, labels, groups, *, seed=42):
    """All fitting and the native-vs-temperature check stay inside calibration rows."""
    y = np.asarray(labels)
    models = model_order(raw)
    if set(y) != set(LABELS):
        raise ValueError("Calibration must contain every class.")
    svm = values(raw["svm"])
    frozen = CalibratedClassifierCV(FrozenEstimator(FrozenScores()), method="sigmoid")
    frozen.fit(svm, y)
    # Pinned sklearn 1.7 serializes to a small, runtime-independent numeric contract.
    sigmoids = frozen.calibrated_classifiers_[0].calibrators
    result = {"svm": {"method": "sigmoid_ovr", "a": [float(s.a_) for s in sigmoids],
                       "b": [float(s.b_) for s in sigmoids]}}
    if not np.allclose(apply_calibration(svm, result["svm"]), frozen.predict_proba(svm), atol=1e-12):
        raise ValueError("Portable sigmoid parameters differ from sklearn calibration.")
    for name in models[1:]:
        p = probabilities(raw[name])
        oof = np.empty_like(p)
        folds = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
        for train, valid in folds.split(p, y, groups):
            oof[valid] = apply_calibration(p[valid], _temperature(p[train], y[train]))
        native_loss = float(log_loss(y, p, labels=list(LABELS)))
        oof_loss = float(log_loss(y, oof, labels=list(LABELS)))
        chosen = _temperature(p, y) if oof_loss < native_loss - 1e-4 else {"method": "identity"}
        chosen["calibration_only_check"] = {"native_log_loss": native_loss, "temperature_oof_log_loss": oof_loss,
                                            "folds": 3, "criterion": "OOF log loss improves by >1e-4"}
        result[name] = chosen
    return result


def soft_vote(matrices, weights):
    models = model_order(matrices)
    if set(weights) != set(models):
        raise ValueError("Every selected soft-voting member requires an explicit weight.")
    if any(type(weights[m]) not in (int, float) for m in models):
        raise ValueError("Weights must be numeric, not booleans or strings.")
    w = np.asarray([weights[name] for name in models], dtype=float)
    if not np.isfinite(w).all() or np.any(w < 0) or not np.isclose(w.sum(), 1, atol=1e-10, rtol=0):
        raise ValueError("Weights must be finite, nonnegative and sum to one.")
    parts = [probabilities(matrices[name]) for name in models]
    if len({p.shape for p in parts}) != 1:
        raise ValueError("Member probability arrays must be aligned.")
    return probabilities(np.einsum("m,mnc->nc", w, np.stack(parts)))


def choose_weights(matrices, labels):
    """Fixed quarter-simplex grid plus uniform weights, without duplicate candidates."""
    models = model_order(matrices)
    candidates = [dict(zip(models, (value / 4 for value in parts)))
                  for parts in product(range(5), repeat=len(models)) if sum(parts) == 4]
    uniform = dict.fromkeys(models, 1 / len(models))
    if uniform not in candidates:
        candidates.append(uniform)
    trials = []
    for weights in candidates:
        prediction = np.asarray(LABELS)[soft_vote(matrices, weights).argmax(axis=1)]
        trials.append({"weights": weights, "f1_macro": float(f1_score(labels, prediction, labels=list(LABELS),
                                                                      average="macro", zero_division=0)),
                       "accuracy": float(accuracy_score(labels, prediction))})
    winner = max(trials, key=lambda t: (t["f1_macro"], t["accuracy"],
                                        -sum((w - 1 / len(models)) ** 2 for w in t["weights"].values())))
    return winner["weights"], trials


def probability_metrics(labels, p):
    p = probabilities(p)
    y = np.asarray(labels)
    targets = np.asarray([[label == c for c in LABELS] for label in y], dtype=float)
    confidence = p.max(axis=1)
    correct = np.asarray(LABELS)[p.argmax(axis=1)] == y
    bins = np.minimum((confidence * 10).astype(int), 9)
    reliability = []
    for index in range(10):
        selected = bins == index
        if selected.any():
            reliability.append({"bin": index, "rows": int(selected.sum()),
                                "confidence": float(confidence[selected].mean()),
                                "accuracy": float(correct[selected].mean())})
    return {"log_loss": float(log_loss(y, p, labels=list(LABELS))),
            "multiclass_brier": float(np.mean(np.sum((p - targets) ** 2, axis=1))),
            "confidence_ece_10_bins": sum(r["rows"] / len(y) * abs(r["accuracy"] - r["confidence"]) for r in reliability),
            "reliability": reliability}
