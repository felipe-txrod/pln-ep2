"""Share fold-local sparse features and enforce a wall-clock limit per SVM fit."""
from __future__ import annotations

import gc
import ctypes
import math
import multiprocessing as mp
import os
import signal
import tempfile
import time
import traceback
import warnings
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from ensemble_classifier.files import read_json
from gpt_classifier.data import LABELS, canonical, digest, file_digest, write_json, write_jsonl
from svm_research.search import (baseline_candidate, broad_candidates, build_pipeline, candidate_id,
                                 classification_metrics, grouped_folds, representation_id, save_array, select_one_se,
                                 vector_settings)


def candidates_a():
    """Four normalized designs per family, three C values and both class weights."""
    allowed = {}
    for family in ("word", "char", "char_wb", "word_char_wb"):
        for design in (0, 1, 3, 4):
            vectors = ({"word": vector_settings("word", design), "char": vector_settings("char_wb", design)}
                       if family == "word_char_wb" else {family: vector_settings(family, design)})
            for c in (.005, .05, .5):
                for weight in (None, "balanced"):
                    candidate = {"family": family, "vectors": vectors, "char_weight": 1.,
                                 "C": c, "class_weight": weight, "loss": "squared_hinge"}
                    allowed[candidate_id(candidate)] = candidate
    return list(allowed.values())


def candidates_b(trials):
    """Refine at most two representations and retain the original baseline."""
    leaders, seen = [], set()
    for trial in sorted((t for t in trials if t["eligible"]), key=lambda t: (-t["mean_f1_macro"], t["candidate_id"])):
        candidate = trial["candidate"]
        signature = representation_id(candidate)
        if signature not in seen:
            leaders.append(candidate)
            seen.add(signature)
        if len(leaders) == 2:
            break
    if not leaders:
        raise ValueError("No fully converged phase A representation is available.")
    result = [baseline_candidate()]
    for leader in leaders:
        for multiplier in (1 / math.sqrt(10), 1., math.sqrt(10)):
            for weight in (None, "balanced"):
                result.append({**leader, "C": float(leader["C"] * multiplier), "class_weight": weight})
        for weight in (None, "balanced"):
            result.append({**leader, "class_weight": weight, "loss": "hinge"})
        if leader["family"] == "word_char_wb":
            for weight in (.5, 2.):
                result.append({**leader, "char_weight": weight})
    candidates = list({candidate_id(c): c for c in result}.values())
    if len(candidates) > 25:
        raise ValueError("Refinement exceeded the amended 25-candidate cap.")
    return candidates


def identity(candidate, fold, train, valid, rows, seed):
    return {"candidate": candidate, "seed": seed, "fold": fold,
            "training_ids": digest(canonical(rows[train].tolist())),
            "validation_ids": digest(canonical(rows[valid].tolist()))}


def validate_fold(root, expected):
    path = root / f"fold-{expected['fold']}.json"
    if not path.exists():
        return None
    result = read_json(path)
    if result.get("identity") != expected:
        raise ValueError("Cached fold identity differs from the amended plan.")
    if result.get("status") == "completed":
        array = path.with_suffix(".npz")
        if file_digest(array) != result["predictions_sha256"]:
            raise ValueError("Cached prediction checksum changed.")
    elif result.get("status") != "timeout":
        raise ValueError("A failed fold requires investigation; it cannot silently become a prediction.")
    return result


def feature_cache(candidate, fold, train, valid, texts, labels, rows, groups, cache, seed):
    """Fit only training text; transform validation once for all C/loss/weight settings."""
    if set(groups[train]) & set(groups[valid]):
        raise ValueError("A similarity component crosses a feature-training fold.")
    root = cache / representation_id(candidate) / f"fold-{fold}"
    root.mkdir(parents=True, exist_ok=True)
    matrix_path, metadata_path = root / "matrices.joblib", root / "features.json"
    key = {"representation": representation_id(candidate), "seed": seed,
           "training_ids": digest(canonical(rows[train].tolist())),
           "validation_ids": digest(canonical(rows[valid].tolist())),
           "training_texts": digest(canonical(texts[train].tolist())),
           "validation_texts": digest(canonical(texts[valid].tolist()))}
    if metadata_path.exists():
        meta = read_json(metadata_path)
        if meta["identity"] != key or file_digest(matrix_path) != meta["matrices_sha256"]:
            raise ValueError("Sparse feature cache identity or checksum changed.")
        return matrix_path, meta
    started = time.perf_counter()
    features = build_pipeline(candidate, seed=seed).named_steps["features"]
    with threadpool_limits(limits=1):
        training = features.fit_transform(texts[train], labels[train]).tocsr()
        validation = features.transform(texts[valid]).tocsr()
    if training.shape[1] != validation.shape[1] or training.shape[0] != len(train) or validation.shape[0] != len(valid):
        raise ValueError("Incompatible fold-local feature matrices.")
    handle, temporary = tempfile.mkstemp(dir=root, prefix=".matrices-", suffix=".joblib")
    os.close(handle)
    try:
        joblib.dump({"training": training, "validation": validation}, temporary, compress=0)
        os.replace(temporary, matrix_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    meta = {"identity": key, "feature_seconds": time.perf_counter() - started,
            "feature_count": training.shape[1], "training_shape": list(training.shape),
            "validation_shape": list(validation.shape), "training_nnz": training.nnz,
            "validation_nnz": validation.nnz, "matrices_sha256": file_digest(matrix_path),
            "matrix_bytes": matrix_path.stat().st_size, "training_only_fit": True}
    write_json(metadata_path, meta)
    del training, validation, features
    gc.collect()
    return matrix_path, meta


def fit_child(candidate, expected, matrix_path, train_labels, valid_labels, valid_rows, root, feature_meta):
    """Isolate the native solver so its parent can stop a genuinely over-budget fit."""
    path = root / f"fold-{expected['fold']}.json"
    try:
        matrices = joblib.load(matrix_path, mmap_mode="r")
        classifier = build_pipeline(candidate, seed=expected["seed"]).named_steps["classifier"]
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            started = time.perf_counter()
            classifier.fit(matrices["training"], train_labels)
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            prediction = classifier.predict(matrices["validation"])
            score_seconds = time.perf_counter() - started
        if tuple(classifier.classes_) != LABELS:
            raise ValueError("Unexpected SVM class order.")
        array_path = path.with_suffix(".npz")
        save_array(array_path, excel_rows=valid_rows,
                   predictions=np.asarray([LABELS.index(p) for p in prediction], dtype=np.uint8))
        result = {"status": "completed", "identity": expected, "training_rows": len(train_labels),
                  "validation_rows": len(valid_labels), "shared_groups": 0,
                  "fit_seconds": fit_seconds, "score_seconds": score_seconds,
                  "feature_count": feature_meta["feature_count"], "n_iter": int(classifier.n_iter_),
                  "convergence_warnings": [str(w.message) for w in captured if issubclass(w.category, ConvergenceWarning)],
                  "metrics": classification_metrics(valid_labels, prediction),
                  "predictions_sha256": file_digest(array_path),
                  "engine": "shared_fold_local_sparse_features", "feature_cache_identity": feature_meta["identity"]}
        write_json(path, result)
    except BaseException as error:
        write_json(path, {"status": "failed", "identity": expected,
                          "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        raise


def child_entry(target, arguments, parent_pid):
    """Stop the native solver if its supervising worker is interrupted or killed."""
    if sys.platform == "linux":
        library = ctypes.CDLL(None, use_errno=True)
        if library.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Could not set the child parent-death signal.")
        if os.getppid() != parent_pid:
            raise RuntimeError("The supervising worker exited before SVM startup.")
    target(*arguments)


def bounded_fit(candidate, expected, matrix_path, train_labels, valid_labels, valid_rows, root, feature_meta,
                timeout_seconds, *, target=None):
    """Use process termination rather than Python signals inside a long C extension."""
    root.mkdir(parents=True, exist_ok=True)
    saved = validate_fold(root, expected)
    if saved is not None:
        return saved
    context = mp.get_context("spawn")
    arguments = (candidate, expected, matrix_path, train_labels, valid_labels, valid_rows, root, feature_meta)
    child = context.Process(target=child_entry, args=(target or fit_child, arguments, os.getpid()))
    started = time.perf_counter()
    # Loky installs a private default context that stdlib spawn cannot import.
    previous_method = mp.get_start_method(allow_none=True)
    try:
        mp.set_start_method("spawn", force=True)
        child.start()
    finally:
        mp.set_start_method(previous_method, force=True)
    state_path = root / f"fold-{expected['fold']}-process.json"
    write_json(state_path, {"status": "running", "pid": child.pid, "timeout_seconds": timeout_seconds})
    try:
        child.join(timeout_seconds)
        if child.is_alive():
            child.terminate()
            child.join(2)
            if child.is_alive():
                child.kill()
                child.join(2)
            # A complete atomic report can arrive just as the deadline expires.
            saved = validate_fold(root, expected)
            if saved is None:
                saved = {"status": "timeout", "identity": expected, "timeout_seconds": timeout_seconds,
                         "elapsed_seconds": time.perf_counter() - started,
                         "eligible": False, "metrics": None, "predictions_sha256": None}
                write_json(root / f"fold-{expected['fold']}.json", saved)
        else:
            if child.exitcode != 0:
                raise RuntimeError(f"SVM subprocess failed (exit {child.exitcode}); inspect {root}.")
            saved = validate_fold(root, expected)
            if saved is None:
                raise RuntimeError("SVM subprocess exited without a fold report.")
        write_json(state_path, {"status": saved["status"], "pid": child.pid,
                               "elapsed_seconds": time.perf_counter() - started})
        return saved
    finally:
        if child.is_alive():
            child.terminate()
            child.join(2)
            if child.is_alive():
                child.kill()
                child.join(2)
        child.close()


def representation_fold(candidates, fold, train, valid, texts, labels, rows, groups, phase_root, cache, seed, timeout):
    reports = {}
    missing = []
    for candidate in candidates:
        identifier = candidate_id(candidate)
        expected = identity(candidate, fold, train, valid, rows, seed)
        saved = validate_fold(phase_root / "trials" / identifier, expected)
        if saved is None:
            missing.append(candidate)
        else:
            reports[identifier] = saved
    if missing:
        matrix_path, meta = feature_cache(candidates[0], fold, train, valid, texts, labels, rows, groups, cache, seed)
        for candidate in missing:
            identifier = candidate_id(candidate)
            report = bounded_fit(candidate, identity(candidate, fold, train, valid, rows, seed), matrix_path,
                                 labels[train], labels[valid], rows[valid], phase_root / "trials" / identifier, meta, timeout)
            reports[identifier] = report
            print(f"Fold {fold} {candidate['family']} C={candidate['C']:.5g} weight={candidate['class_weight']} "
                  f"loss={candidate['loss']} status={report['status']} [{identifier}]", flush=True)
    return reports


def aggregate(candidate, reports):
    complete = all(r["status"] == "completed" for r in reports)
    metrics = [r["metrics"]["f1_macro"] for r in reports] if complete else []
    return {"candidate_id": candidate_id(candidate), "candidate": candidate, "folds": reports,
            "eligible": complete and not any(r["convergence_warnings"] for r in reports),
            "mean_f1_macro": float(np.mean(metrics)) if complete else None,
            "std_f1_macro": float(np.std(metrics, ddof=1)) if complete else None,
            "mean_feature_count": float(np.mean([r["feature_count"] for r in reports])) if complete else None,
            "mean_accuracy": float(np.mean([r["metrics"]["accuracy"] for r in reports])) if complete else None,
            "fit_seconds": sum(r.get("fit_seconds", 0) for r in reports),
            "score_seconds": sum(r.get("score_seconds", 0) for r in reports),
            "timeouts": sum(r["status"] == "timeout" for r in reports),
            "per_class_f1": {label: [r["metrics"]["classification_report"][label]["f1-score"] for r in reports]
                             for label in LABELS} if complete else None}


def search(data, config, output, phase):
    phase_root = output / f"phase-{phase}"
    phase_root.mkdir(parents=True, exist_ok=True)
    candidates = candidates_a() if phase == "a" else candidates_b(read_json(output / "phase-a/search.json")["trials"])
    if phase == "b" and len(candidates) > config["phase_b_candidate_cap"]:
        raise ValueError("Phase B exceeded its frozen cap.")
    seed = config["seed"] + (101 if phase == "b" else 0)
    labels = np.asarray([r.label for r in data.training])
    texts = np.asarray([r.text for r in data.training], dtype=object)
    rows = np.asarray([r.excel_row for r in data.training])
    folds = grouped_folds(labels, data.groups, config[f"phase_{phase}_folds"], seed)
    plan = {"phase": phase, "candidates": candidates, "candidate_count": len(candidates), "folds": len(folds),
            "seed": seed, "fit_timeout_seconds": config["fit_timeout_seconds"], "jobs": config["jobs"],
            "phase_a_search_sha256": file_digest(output / "phase-a/search.json") if phase == "b" else None}
    plan_path = phase_root / "plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Amended phase plan changed; use a new output directory.")
    if not plan_path.exists():
        write_json(plan_path, plan)
    assignments = np.full(len(rows), -1)
    for fold, (_, valid) in enumerate(folds):
        assignments[valid] = fold
    write_jsonl(phase_root / "folds.jsonl", [{"excel_row": int(row), "group_id": str(g), "fold": int(fold)}
                                          for row, g, fold in zip(rows, data.groups, assignments)])
    representations = defaultdict(list)
    for candidate in candidates:
        representations[representation_id(candidate)].append(candidate)
    # Prioritize missing small feature families; all plans are fixed before fitting.
    ordered = sorted(representations.values(), key=lambda cs: (len(cs[0]["vectors"]), representation_id(cs[0])))
    tasks = [(cs, fold, train, valid) for cs in ordered for fold, (train, valid) in enumerate(folds)]
    collected = defaultdict(dict)
    print(f"Phase {phase.upper()}: {len(candidates)} candidates, {len(candidates)*len(folds)} folds, "
          f"{config['jobs']} concurrent fits, {config['fit_timeout_seconds']}s per fit", flush=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=config["jobs"], batch_size=1, pre_dispatch=config["jobs"], return_as="generator_unordered")(
            delayed(representation_fold)(cs, fold, train, valid, texts, labels, rows, data.groups, phase_root,
                                         output / "cache" / f"phase-{phase}", seed, config["fit_timeout_seconds"])
            for cs, fold, train, valid in tasks)
        for reports in results:
            for identifier, result in reports.items():
                collected[identifier][result["identity"]["fold"]] = result
            completed = sum(len(value) == len(folds) for value in collected.values())
            terminal = sum(len(value) for value in collected.values())
            write_json(phase_root / "progress.json", {"completed": completed, "total": len(candidates),
                       "terminal_folds": terminal, "total_folds": len(candidates)*len(folds),
                       "timeout_folds": sum(r["status"] == "timeout" for value in collected.values() for r in value.values())})
            print(f"Progress: {completed}/{len(candidates)} candidates; {terminal}/{len(candidates)*len(folds)} folds", flush=True)
    trials = [aggregate(c, [collected[candidate_id(c)][i] for i in range(len(folds))]) for c in candidates]
    result = {"status": "completed", "phase": phase, "plan_sha256": file_digest(plan_path), "trials": trials,
              "selection": select_one_se(trials), "test1_accessed": False, "test2_accessed": False,
              "limitation": "Amended, computationally bounded adaptive development search. Timed-out/nonconverged candidates are ineligible; removed variants are not evidence of statistical inferiority."}
    write_json(phase_root / "search.json", result)
    return result
