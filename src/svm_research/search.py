"""Bounded two-phase grouped searches with fold-local TF-IDF and resumable fits."""
from __future__ import annotations

import math
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Memory, Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from ensemble_classifier.files import read_json
from gpt_classifier.data import LABELS, canonical, digest, file_digest, write_json, write_jsonl
from lexical_classifier.inference import normalize_text


def candidate_id(candidate: dict[str, Any]) -> str:
    return digest(canonical(candidate))[:20]


def vector_settings(family: str, design: int) -> dict[str, Any]:
    """Fixed fractional design; it does not enumerate an unrestricted Cartesian grid."""
    word = family == "word"
    ngrams = ([(1, 2), (1, 1), (1, 3), (1, 2), (1, 3), (1, 1), (1, 2), (1, 2)] if word
              else [(3, 5), (2, 4), (4, 6), (3, 6), (3, 5), (2, 5), (4, 5), (3, 5)])
    return {"analyzer": family, "ngram_range": list(ngrams[design]),
            "min_df": ([2, 1, 3, 5, 2, 3, 1, 5] if word else [3, 2, 5, 3, 2, 1, 5, 3])[design],
            "max_df": ([.98, 1., .95, .98, 1., .95, .98, 1.] if word
                       else [1., .98, 1., .95, .98, 1., .95, 1.])[design],
            "sublinear_tf": [True, False, True, False, True, False, True, False][design],
            "binary": design in (3, 5), "norm": ["l2", "l2", "l2", "l2", "l1", "l2", None, "l2"][design],
            "max_features": ([60000, 60000, 100000, 60000, 60000, 100000, 60000, 100000] if word
                             else [50000, 50000, 80000, 50000, 50000, 80000, 50000, 80000])[design]}


def baseline_candidate() -> dict[str, Any]:
    return {"family": "word_char_wb", "vectors": {"word": vector_settings("word", 0),
                                                    "char": vector_settings("char_wb", 0)},
            "char_weight": 1., "C": .05, "class_weight": None, "loss": "squared_hinge"}


def broad_candidates() -> list[dict[str, Any]]:
    result = [baseline_candidate()]
    for family in ("word", "char", "char_wb", "word_char_wb"):
        for design in range(8):
            vectors = ({"word": vector_settings("word", design), "char": vector_settings("char_wb", design)}
                       if family == "word_char_wb" else {family: vector_settings(family, design)})
            for c in (.005, .05, .5, 5.):
                for weight in (None, "balanced"):
                    result.append({"family": family, "vectors": vectors, "char_weight": 1.,
                                   "C": c, "class_weight": weight, "loss": "squared_hinge"})
    return list({candidate_id(c): c for c in result}.values())


def representation_id(candidate: dict[str, Any]) -> str:
    return digest(canonical({"vectors": candidate["vectors"], "char_weight": candidate["char_weight"]}))


def refined_candidates(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """At most 61 candidates, refining three representations identified in phase A."""
    leaders, seen = [], set()
    for trial in sorted((t for t in trials if t["eligible"]), key=lambda t: -t["mean_f1_macro"]):
        signature = representation_id(trial["candidate"])
        if signature not in seen:
            leaders.append(trial["candidate"])
            seen.add(signature)
        if len(leaders) == 3:
            break
    if not leaders:
        raise ValueError("Phase A produced no converged candidates.")
    result = [baseline_candidate()]
    # One-factor-at-a-time refinement limits adaptive multiplicity and compute.
    for leader in leaders:
        for multiplier in (.1, 1 / math.sqrt(10), 1., math.sqrt(10), 10.):
            for weight in (None, "balanced"):
                result.append({**leader, "C": float(leader["C"] * multiplier), "class_weight": weight})
        for loss in ("hinge", "squared_hinge"):
            for char_weight in ((.5, 1., 2.) if leader["family"] == "word_char_wb" else (1.,)):
                result.append({**leader, "loss": loss, "char_weight": char_weight})
    unique = list({candidate_id(c): c for c in result}.values())
    if len(unique) > 61:
        raise ValueError("Refinement exceeded its predeclared budget.")
    return unique


def build_pipeline(candidate: dict[str, Any], *, memory: Memory | None = None, seed: int = 42) -> Pipeline:
    vectors = []
    for name, parameters in candidate["vectors"].items():
        parameters = {**parameters, "ngram_range": tuple(parameters["ngram_range"])}
        vectors.append((name, TfidfVectorizer(preprocessor=normalize_text, dtype=np.float32, **parameters)))
    weights = ({"word": 1., "char": candidate["char_weight"]}
               if candidate["family"] == "word_char_wb" and candidate["char_weight"] != 1. else None)
    return Pipeline([("features", FeatureUnion(vectors, transformer_weights=weights, n_jobs=1)),
                     ("classifier", LinearSVC(C=candidate["C"], class_weight=candidate["class_weight"],
                                              loss=candidate["loss"], penalty="l2",
                                              dual=True if candidate["loss"] == "hinge" else "auto",
                                              max_iter=20000, tol=1e-4, random_state=seed))], memory=memory)


def grouped_folds(labels, groups, n_splits: int, seed: int):
    y, g = np.asarray(labels), np.asarray(groups)
    if len(y) != len(g) or set(y) != set(LABELS):
        raise ValueError("Missing classes or misaligned grouped inputs.")
    folds = list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, g))
    for train, valid in folds:
        if (set(g[train]) & set(g[valid]) or set(y[train]) != set(LABELS) or set(y[valid]) != set(LABELS)):
            raise ValueError("Invalid grouped fold: shared groups or absent classes.")
    return folds


def classification_metrics(y, prediction) -> dict[str, Any]:
    return {"accuracy": float(accuracy_score(y, prediction)),
            "f1_macro": float(f1_score(y, prediction, labels=list(LABELS), average="macro", zero_division=0)),
            "classification_report": classification_report(y, prediction, labels=list(LABELS), zero_division=0, output_dict=True),
            "confusion_matrix": confusion_matrix(y, prediction, labels=list(LABELS)).tolist()}


def save_array(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".array-", suffix=".npz")
    os.close(handle)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fit_fold(candidate, fold, train, valid, texts, labels, rows, groups, root, cache, seed):
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / f"fold-{fold}.json"
    array_path = root / f"fold-{fold}.npz"
    identity = {"candidate": candidate, "seed": seed, "fold": fold,
                "training_ids": digest(canonical(rows[train].tolist())),
                "validation_ids": digest(canonical(rows[valid].tolist()))}
    if report_path.exists():
        saved = read_json(report_path)
        if saved["identity"] != identity or saved["predictions_sha256"] != file_digest(array_path):
            raise ValueError("Cached fold was modified or belongs to another protocol.")
        return saved
    if set(groups[train]) & set(groups[valid]):
        raise ValueError("A similarity component crosses training and validation.")
    model = build_pipeline(candidate, memory=Memory(cache, verbose=0), seed=seed)
    started = time.perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(texts[train], labels[train])
        fit_seconds = time.perf_counter() - started
        started_score = time.perf_counter()
        prediction = model.predict(texts[valid])
        score_seconds = time.perf_counter() - started_score
    if tuple(model.classes_) != LABELS:
        raise ValueError("Unexpected SVM class order.")
    convergence = [str(w.message) for w in captured if issubclass(w.category, ConvergenceWarning)]
    save_array(array_path, excel_rows=rows[valid], predictions=np.asarray([LABELS.index(p) for p in prediction], dtype=np.uint8))
    result = {"status": "completed", "identity": identity, "training_rows": len(train), "validation_rows": len(valid),
              "shared_groups": 0, "fit_seconds": fit_seconds, "score_seconds": score_seconds,
              "n_iter": int(model.named_steps["classifier"].n_iter_), "convergence_warnings": convergence,
              "feature_count": int(model.named_steps["classifier"].coef_.shape[1]),
              "metrics": classification_metrics(labels[valid], prediction),
              "predictions_sha256": file_digest(array_path)}
    write_json(report_path, result)
    return result


def select_one_se(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer stronger regularization/smaller feature sets within the best mean's SE."""
    eligible = [t for t in trials if t["eligible"]]
    if not eligible:
        raise ValueError("No candidate converged in every fold.")
    best = max(eligible, key=lambda t: (t["mean_f1_macro"], -t["std_f1_macro"], t["candidate_id"]))
    threshold = best["mean_f1_macro"] - best["std_f1_macro"] / math.sqrt(len(best["folds"]))
    allowed = [t for t in eligible if t["mean_f1_macro"] >= threshold]
    winner = min(allowed, key=lambda t: (t["candidate"]["C"], t["mean_feature_count"],
                                       t["candidate"]["loss"] != "squared_hinge",
                                       t["candidate"]["class_weight"] is not None,
                                       -t["mean_f1_macro"], t["candidate_id"]))
    return {"candidate_id": winner["candidate_id"], "candidate": winner["candidate"],
            "mean_f1_macro": winner["mean_f1_macro"], "std_f1_macro": winner["std_f1_macro"],
            "best_mean_f1_macro": best["mean_f1_macro"], "one_se_threshold": threshold,
            "criterion": "Within one SE of best mean macro-F1: smallest C, fewest features, squared hinge, no class weighting, mean F1, stable ID"}


def search_phase(data, config, output: Path, phase: str) -> dict[str, Any]:
    phase_root = output / f"phase-{phase}"
    phase_root.mkdir(parents=True, exist_ok=True)
    trials_a = read_json(output / "phase-a/search.json")["trials"] if phase == "b" else None
    candidates = broad_candidates() if phase == "a" else refined_candidates(trials_a)
    n_splits = config[f"phase_{phase}_folds"]
    seed = config["seed"] + (101 if phase == "b" else 0)
    texts = np.asarray([r.text for r in data.training], dtype=object)
    labels = np.asarray([r.label for r in data.training])
    rows = np.asarray([r.excel_row for r in data.training])
    folds = grouped_folds(labels, data.groups, n_splits, seed)
    phase_plan = {"phase": phase, "seed": seed, "folds": n_splits,
                  "candidates": candidates, "candidate_count": len(candidates), "fits": len(candidates) * n_splits,
                  "training_rows": len(rows), "selection_metric": "macro_f1",
                  "phase_a_search_sha256": file_digest(output / "phase-a/search.json") if phase == "b" else None}
    plan_path = phase_root / "plan.json"
    if plan_path.exists():
        if read_json(plan_path) != phase_plan:
            raise ValueError("Search plan changed; use a new research directory.")
    else:
        write_json(plan_path, phase_plan)
    assignments = np.full(len(rows), -1)
    for index, (_, valid) in enumerate(folds):
        assignments[valid] = index
    write_jsonl(phase_root / "folds.jsonl", [{"excel_row": int(row), "group_id": str(g), "fold": int(fold)}
                                          for row, g, fold in zip(rows, data.groups, assignments)])
    cache = output / "cache" / f"phase-{phase}"
    memory = Memory(cache, verbose=0)
    trials = []
    # Keep equal representations adjacent, so folds reuse their training-only feature cache.
    candidates = sorted(candidates, key=lambda c: (representation_id(c), c["C"], str(c["class_weight"]), c["loss"]))
    last_representation = None
    for index, candidate in enumerate(candidates):
        signature = representation_id(candidate)
        if signature != last_representation:
            memory.reduce_size(bytes_limit=config["cache_gib"] * 1024 ** 3)
            last_representation = signature
        identifier = candidate_id(candidate)
        print(f"Phase {phase.upper()} {index + 1}/{len(candidates)} {candidate['family']} C={candidate['C']:.5g} "
              f"weight={candidate['class_weight']} loss={candidate['loss']} [{identifier}]", flush=True)
        reports = Parallel(n_jobs=config["jobs"], backend="loky", inner_max_num_threads=1)(
            delayed(fit_fold)(candidate, fold, train, valid, texts, labels, rows, data.groups,
                              phase_root / "trials" / identifier, cache, seed)
            for fold, (train, valid) in enumerate(folds))
        scores = [r["metrics"]["f1_macro"] for r in reports]
        trial = {"candidate_id": identifier, "candidate": candidate, "folds": reports,
                 "eligible": not any(r["convergence_warnings"] for r in reports),
                 "mean_f1_macro": float(np.mean(scores)), "std_f1_macro": float(np.std(scores, ddof=1)),
                 "mean_accuracy": float(np.mean([r["metrics"]["accuracy"] for r in reports])),
                 "mean_feature_count": float(np.mean([r["feature_count"] for r in reports])),
                 "fit_seconds": sum(r["fit_seconds"] for r in reports),
                 "score_seconds": sum(r["score_seconds"] for r in reports),
                 "per_class_f1": {label: [r["metrics"]["classification_report"][label]["f1-score"] for r in reports]
                                  for label in LABELS}}
        trials.append(trial)
        write_json(phase_root / "progress.json", {"completed": len(trials), "total": len(candidates), "last": identifier})
    result = {"status": "completed", "phase": phase, "plan_sha256": file_digest(plan_path), "trials": trials,
              "selection": select_one_se(trials), "test1_accessed": False, "test2_accessed": False,
              "limitation": "Adaptive development CV diagnostics, not an unbiased nested-CV generalization estimate. Phase B reuses training rows with a new grouped partition."}
    write_json(phase_root / "search.json", result)
    memory.reduce_size(bytes_limit=config["cache_gib"] * 1024 ** 3)
    return result
