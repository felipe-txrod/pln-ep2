"""Train once, calibrate on reserved rows, screen ensembles and freeze before final labels."""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from ensemble_classifier.files import align, read_json, read_predictions
from ensemble_classifier.soft_voting import aligned_scores, apply_calibration
from gpt_classifier.data import (LABELS, Text, canonical, digest, file_digest, read_xlsx,
                                 write_json, write_jsonl)
from svm_budgeted.cli import base_config, validate_plan as validate_search
from svm_budgeted.engine import child_entry
from svm_research.data import load_data, location, runtime
from svm_research.search import build_pipeline, classification_metrics, save_array
from .methods import (adoption_screen, apply_calibrator, complementarity, describe_probabilities,
                      mix, nested_calibration, nested_weights, paired_statistics, predicted)


def implementation():
    source = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(source)): file_digest(path) for path in sorted((source / "svm_finalization").glob("*.py"))}


def prepare(search_root, search_config, output):
    if output.exists() or output.resolve() == search_root.resolve():
        raise FileExistsError("Finalization requires a new directory; preserve both CV searches.")
    config, search_plan = validate_search(search_config, search_root)
    inputs = {}
    for phase in ("a", "b"):
        directory = search_root / f"phase-{phase}"
        result = read_json(directory / "search.json")
        verification = read_json(search_root / f"verification-phase-{phase}.json")
        if (result["status"] != "completed" or verification["status"] != "passed"
                or verification["phase"] != phase or verification["search_sha256"] != file_digest(directory / "search.json")
                or verification["one_se_selection"]["candidate_id"] != result["selection"]["candidate_id"]):
            raise ValueError("Both search phases require matching independent verification.")
        for relative, expected in verification["verified_files"].items():
            path = search_root / relative
            if file_digest(path) != expected:
                raise ValueError("A verified CV fold changed before finalization.")
            inputs[str(path.resolve())] = expected
        for path in (directory / "plan.json", directory / "search.json", search_root / f"verification-phase-{phase}.json"):
            inputs[str(path.resolve())] = file_digest(path)
    final_search = read_json(search_root / "phase-b/search.json")
    selected = final_search["selection"]
    historical_reference = location("docs/results/data/ensemble-soft-test1.json")
    expected_gold = read_json(historical_reference)["gold_sha256"]
    inputs[str(historical_reference.resolve())] = file_digest(historical_reference)
    data = load_data(base_config(config))
    plan = {"status": "prepared", "version": 1, "search_root": str(search_root.resolve()),
            "search_config": str(search_config.resolve()), "search_config_value": config,
            "selected_candidate": selected, "input_files": inputs,
            "search_plan_sha256": file_digest(search_root / "plan.json"),
            "protocol": search_plan["protocol"], "runtime": runtime(), "implementation": implementation(),
            "training_identity": data.audit["label_identity"], "refit_wall_limit_seconds": 180,
            "test1_input": "data/test1.xlsx", "test1_gold": "data/test1-rotulado-para-teste-estatistico.xlsx",
            "test1_expected_gold_sha256": expected_gold,
            "historical_test1_run": "results/ensemble-soft-qwen-test1-20261105",
            "tie_clarification": "Equally distant weight ties use higher mean inner F1, then smaller SVM weight; fixed before meta-validation scoring.",
            "test1_accessed": False, "test2_accessed": False}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    from .reporting import report
    report(output)
    return {"status": "prepared", "selected_candidate": selected, "output": str(output)}


def validate_plan(output):
    plan = read_json(output / "plan.json")
    if runtime() != plan["runtime"] or implementation() != plan["implementation"]:
        raise ValueError("Finalization runtime/source changed; use a new directory.")
    config, _ = validate_search(Path(plan["search_config"]), Path(plan["search_root"]))
    if config != plan["search_config_value"] or file_digest(Path(plan["search_root"]) / "plan.json") != plan["search_plan_sha256"]:
        raise ValueError("Original CV plan changed.")
    for name, expected in plan["input_files"].items():
        if file_digest(Path(name)) != expected:
            raise ValueError(f"Verified search input changed: {name}")
    return plan


def _dump_model(path, model):
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".model-", suffix=".joblib")
    os.close(handle)
    try:
        joblib.dump(model, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _refit_child(candidate, training, targets, output):
    try:
        model = build_pipeline(candidate, seed=42)
        started = time.perf_counter()
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit([r.text for r in training], [r.label for r in training])
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            margins = model.decision_function([r.text for r in targets])
            score_seconds = time.perf_counter() - started
        convergence = [str(w.message) for w in captured if issubclass(w.category, ConvergenceWarning)]
        if convergence or tuple(model.classes_) != LABELS or margins.shape != (len(targets), 3) or not np.isfinite(margins).all():
            raise ValueError(f"Final SVM failed its convergence/class/output checks: {convergence}")
        _dump_model(output / "model.joblib", model)
        save_array(output / "holdout-margins.npz", margins=margins,
                   excel_rows=np.asarray([r.excel_row for r in targets]),
                   text_sha256=np.asarray([digest(r.text) for r in targets]))
        write_json(output / "refit.json", {"status": "completed", "candidate": candidate,
                   "training_rows": len(training), "target_rows": len(targets), "fit_seconds": fit_seconds,
                   "score_seconds": score_seconds, "n_iter": int(model.named_steps["classifier"].n_iter_),
                   "feature_count": int(model.named_steps["classifier"].coef_.shape[1]),
                   "convergence_warnings": convergence, "labels": list(LABELS),
                   "training_rows_identity": digest(canonical([(r.excel_row, digest(r.text), r.label) for r in training])),
                   "model_sha256": file_digest(output / "model.joblib"),
                   "margins_sha256": file_digest(output / "holdout-margins.npz"),
                   "test1_accessed": False, "test2_accessed": False})
    except BaseException as error:
        write_json(output / "refit-error.json", {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        raise


def ensure_refit(plan, data, output):
    path = output / "refit.json"
    if not path.exists():
        leftovers = [output / name for name in ("model.joblib", "holdout-margins.npz", "refit-error.json") if (output / name).exists()]
        if leftovers:
            archive = output / f"interrupted-refit-{time.time_ns()}"
            archive.mkdir()
            for item in leftovers:
                shutil.move(item, archive / item.name)
        print("Fitting the selected SVM on the 16093 original training rows only", flush=True)
        context = mp.get_context("spawn")
        arguments = (plan["selected_candidate"]["candidate"], data.training,
                     [Text(r.excel_row, r.text) for r in data.holdout], output)
        child = context.Process(target=child_entry, args=(_refit_child, arguments, os.getpid()))
        child.start()
        try:
            child.join(plan["refit_wall_limit_seconds"])
            if child.is_alive():
                raise TimeoutError("Final refit exceeded its 180-second compute budget; no ensemble was frozen.")
            if child.exitcode != 0:
                raise RuntimeError("Final refit failed; inspect refit-error.json before retrying.")
        finally:
            if child.is_alive():
                child.terminate()
                child.join(2)
                if child.is_alive():
                    child.kill()
                    child.join(2)
            child.close()
    refit = read_json(path)
    if (refit["status"] != "completed" or refit["candidate"] != plan["selected_candidate"]["candidate"]
            or refit["training_rows"] != len(data.training)
            or refit["training_rows_identity"] != digest(canonical([(r.excel_row, digest(r.text), r.label) for r in data.training]))
            or refit["model_sha256"] != file_digest(output / "model.joblib")
            or refit["margins_sha256"] != file_digest(output / "holdout-margins.npz")):
        raise ValueError("Final SVM refit provenance changed.")
    with np.load(output / "holdout-margins.npz", allow_pickle=False) as arrays:
        if (not np.array_equal(arrays["excel_rows"], [r.excel_row for r in data.holdout])
                or not np.array_equal(arrays["text_sha256"], [digest(r.text) for r in data.holdout])):
            raise ValueError("New SVM margins are misaligned with reserved holdout rows.")
        return arrays["margins"].copy(), refit


def development_analysis(data, margins, calibrator):
    selection = data.selection
    labels = np.asarray([data.holdout[i].label for i in selection])
    groups = data.holdout_groups[selection]
    old = apply_calibration(data.old_margins, data.baseline["calibration"]["svm"])[selection]
    new, norberto = apply_calibrator(margins, calibrator)[selection], data.norberto[selection]
    a, b = mix(old, norberto, .75), mix(new, norberto, .75)
    weight, weights_report, policy = nested_weights(new, norberto, labels, groups)
    c = mix(new, norberto, weight)
    assignments = np.asarray(weights_report["outer_assignments"])
    b_screen = adoption_screen(labels, a, b, groups, assignments)
    c_screen = adoption_screen(labels, b, policy, groups, assignments)
    c_screen["weight_stable"] = weights_report["final_weight_outer_matches"] >= 3
    choice = "A"
    if b_screen["passed"]:
        choice = "C" if c_screen["passed"] and c_screen["weight_stable"] else "B"
    matrices = {"A": a, "B": b, "C_fixed_diagnostic": c, "C_policy_oof": policy,
                "svm_original_calibrated": old, "svm_optimized_calibrated": new, "norberto_unchanged": norberto}
    report = {"status": "completed", "rows": len(selection), "groups": len(set(groups)),
              "weights": weights_report, "B_over_A_screen": b_screen, "C_policy_over_B_screen": c_screen,
              "recommended_configuration": choice, "recommendation_frozen_before_test1": True,
              "metrics": {name: describe_probabilities(labels, p) for name, p in matrices.items()},
              "native_svm_metrics": {"original": classification_metrics(labels, np.asarray(LABELS)[data.old_margins[selection].argmax(axis=1)]),
                                     "optimized": classification_metrics(labels, np.asarray(LABELS)[margins[selection].argmax(axis=1)])},
              "paired_diagnostics": paired_statistics(labels, {name: predicted(matrices[name])
                                          for name in ("A", "B", "C_fixed_diagnostic", "C_policy_oof")}, groups),
              "complementarity": {"original_svm_vs_norberto": complementarity(labels, predicted(old), predicted(norberto)),
                                  "optimized_svm_vs_norberto": complementarity(labels, predicted(new), predicted(norberto))},
              "test1_accessed": False, "test2_accessed": False,
              "caveat": "Meta-validation has historical exposure. B is fixed before these labels; C policy is outer OOF for weight selection conditional on fixed upstream training/calibration. Full-data fixed C is adaptive and is not its unbiased selection-policy estimate. No end-to-end generalization guarantee is claimed."}
    return report, matrices


def develop(output):
    plan = validate_plan(output)
    if (output / "frozen_manifest.json").exists():
        frozen = load_frozen(output)
        return {"status": "already_frozen", "recommended_configuration": frozen["recommended_configuration"]}
    data = load_data(base_config(plan["search_config_value"]))
    if data.audit["label_identity"] != plan["training_identity"]:
        raise ValueError("Original source label identity changed.")
    margins, refit = ensure_refit(plan, data, output)
    cal = data.calibration
    print("Nested grouped calibration: sigmoid/isotonic on 2027 reserved rows", flush=True)
    calibrator, calibration_report, cal_oof = nested_calibration(margins[cal],
        np.asarray([data.holdout[i].label for i in cal]), data.holdout_groups[cal])
    write_json(output / "calibration.json", {"calibrator": calibrator, "report": calibration_report})
    save_array(output / "calibration-oof.npz", excel_rows=np.asarray([data.holdout[i].excel_row for i in cal]),
               **cal_oof)
    print("Nested grouped weights and conservative adoption checks on 1972 meta-validation rows", flush=True)
    development, matrices = development_analysis(data, margins, calibrator)
    write_json(output / "development.json", development)
    selection = data.selection
    save_array(output / "meta-validation.npz", excel_rows=np.asarray([data.holdout[i].excel_row for i in selection]),
               text_sha256=np.asarray([digest(data.holdout[i].text) for i in selection]), **matrices)
    validate_plan(output)
    choice = development["recommended_configuration"]
    weight = development["weights"]["final_svm_weight"]
    frozen = {"version": 1, "labels": list(LABELS), "plan_sha256": file_digest(output / "plan.json"),
              "candidate": plan["selected_candidate"], "model_sha256": refit["model_sha256"],
              "svm_calibrator": calibrator, "original_baseline": data.baseline,
              "weights": {"A": {"svm": .75, "norberto": .25}, "B": {"svm": .75, "norberto": .25},
                          "C": {"svm": weight, "norberto": 1 - weight}},
              "recommended_configuration": choice, "recommendation_frozen_before_test1": True,
              "development_sha256": file_digest(output / "development.json"),
              "files": {name: file_digest(output / name) for name in ("model.joblib", "refit.json", "holdout-margins.npz",
                         "calibration.json", "calibration-oof.npz", "development.json", "meta-validation.npz")},
              "test1_accessed": False, "test2_accessed": False, "default_promoted": False}
    if (output / "frozen.json").exists():
        if read_json(output / "frozen.json") != frozen:
            raise ValueError("An existing frozen configuration differs; preserve it and use a new directory.")
    else:
        write_json(output / "frozen.json", frozen, replace=False)
    write_json(output / "frozen_manifest.json", {"artifact_sha256": file_digest(output / "frozen.json"),
               "labels": list(LABELS), "recommended_configuration": choice,
               "test1_accessed": False, "test2_accessed": False}, replace=False)
    from .reporting import report
    report(output)
    return {"status": "frozen", "calibration": calibration_report["chosen_method"],
            "weights": frozen["weights"], "recommended_configuration": choice,
            "meta_macro_f1": {name: development["metrics"][name]["f1_macro"] for name in ("A", "B", "C_fixed_diagnostic", "C_policy_oof")}}


def load_frozen(output):
    plan = validate_plan(output)
    frozen, manifest = read_json(output / "frozen.json"), read_json(output / "frozen_manifest.json")
    if (file_digest(output / "frozen.json") != manifest["artifact_sha256"]
            or frozen["plan_sha256"] != file_digest(output / "plan.json")
            or frozen["candidate"] != plan["selected_candidate"]
            or frozen["recommended_configuration"] != manifest["recommended_configuration"]
            or frozen["test1_accessed"] is not False or frozen["test2_accessed"] is not False):
        raise ValueError("Frozen finalization contract changed.")
    for name, expected in frozen["files"].items():
        if file_digest(output / name) != expected:
            raise ValueError(f"A frozen artifact changed: {name}")
    return frozen


def _cached_member(directory, name, records, source, frozen):
    root = directory / "members" / name
    run, path = read_json(root / "run.json"), root / "predictions.jsonl"
    if (run["status"] != "completed" or run["gold_accessed"] is not False
            or run["training_performed"] is not False or run["input"]["sha256"] != source["sha256"]
            or run["artifact_sha256"] != frozen["original_baseline"]["model_identities"][name]["artifact_sha256"]
            or run["predictions_sha256"] != file_digest(path) or run["labels"] != list(LABELS)):
        raise ValueError("Historical Test1 scores have incompatible source/model provenance.")
    rows = read_predictions(path)
    align(records, rows)
    scores = aligned_scores(rows, "decision_scores" if name == "svm" else "probabilities")
    if [row["label"] for row in rows] != np.asarray(LABELS)[scores.argmax(axis=1)].tolist():
        raise ValueError("Historical labels disagree with native score argmax.")
    return scores, {"run_sha256": file_digest(root / "run.json"), "predictions_sha256": file_digest(path)}


def predict_test1(output):
    frozen = load_frozen(output)  # Must complete before any final-set input access.
    plan = read_json(output / "plan.json")
    target = output / "test1"
    if (target / "prediction-manifest.json").exists():
        manifest = read_json(target / "prediction-manifest.json")
        if (manifest.get("status") != "completed" or manifest.get("rows") != 900
                or manifest.get("gold_accessed") is not False
                or manifest["frozen_sha256"] != file_digest(output / "frozen.json")):
            raise ValueError("Test1 predictions belong to a different frozen configuration.")
        for name, expected in manifest["files"].items():
            if file_digest(target / name) != expected:
                raise ValueError("Saved Test1 predictions changed.")
        return manifest
    records, source = read_xlsx(location(plan["test1_input"]))
    if len(records) != 900 or any(type(r) is not Text for r in records):
        raise ValueError("Test1 inference requires exactly 900 label-free text records.")
    historical_root = location(plan["historical_test1_run"])
    historical_run = read_json(historical_root / "run.json")
    baseline_root = location(plan["search_config_value"]["baseline"])
    if (historical_run["artifact_sha256"] != file_digest(baseline_root / "model.json")
            or historical_run["input"]["sha256"] != source["sha256"]
            or historical_run["gold_accessed"] is not False
            or historical_run["predictions_sha256"] != file_digest(historical_root / "predictions.jsonl")):
        raise ValueError("Historical 75/25 Test1 ensemble identity changed.")
    old_margins, old_source = _cached_member(historical_root, "svm", records, source, frozen)
    native_n, n_source = _cached_member(historical_root, "norberto", records, source, frozen)
    old = apply_calibration(old_margins, frozen["original_baseline"]["calibration"]["svm"])
    norberto = apply_calibration(native_n, frozen["original_baseline"]["calibration"]["norberto"])
    with threadpool_limits(limits=1):
        model = joblib.load(output / "model.joblib")
        if tuple(model.classes_) != LABELS:
            raise ValueError("Final SVM class order changed.")
        new_margins = model.decision_function([r.text for r in records])
    new = apply_calibrator(new_margins, frozen["svm_calibrator"])
    matrices = {name: mix(old if name == "A" else new, norberto, weights["svm"])
                for name, weights in frozen["weights"].items()}
    historical = read_predictions(historical_root / "predictions.jsonl")
    align(records, historical)
    if (not np.allclose(matrices["A"], aligned_scores(historical, "probabilities"), atol=1e-12, rtol=0)
            or [r["label"] for r in historical] != predicted(matrices["A"]).tolist()):
        raise ValueError("Configuration A does not reproduce exact historical predictions.")
    matrices.update(svm_original_calibrated=old, svm_optimized_calibrated=new, norberto_unchanged=norberto)
    target.mkdir(parents=True, exist_ok=True)
    for name, p in matrices.items():
        write_jsonl(target / f"{name}.jsonl", [{"excel_row": r.excel_row, "text_sha256": digest(r.text),
                    "label": str(label), "probabilities": dict(zip(LABELS, values.tolist()))}
                    for r, label, values in zip(records, predicted(p), p)])
    for name, margins in (("svm_original_native", old_margins), ("svm_optimized_native", new_margins)):
        write_jsonl(target / f"{name}.jsonl", [{"excel_row": r.excel_row, "text_sha256": digest(r.text),
                    "label": LABELS[int(scores.argmax())], "decision_scores": dict(zip(LABELS, scores.tolist()))}
                    for r, scores in zip(records, margins)])
    names = [*matrices, "svm_original_native", "svm_optimized_native"]
    manifest = {"status": "completed", "rows": 900, "input": source,
                "frozen_sha256": file_digest(output / "frozen.json"), "gold_accessed": False,
                "parameters_updated": False, "historical_A_matches_exactly": True,
                "recommended_configuration": frozen["recommended_configuration"],
                "original_svm_source": old_source, "norberto_source": n_source,
                "files": {f"{name}.jsonl": file_digest(target / f"{name}.jsonl") for name in names}}
    load_frozen(output)
    write_json(target / "prediction-manifest.json", manifest)
    return manifest


def evaluate_test1(output):
    frozen = load_frozen(output)
    manifest = predict_test1(output)
    target, path = output / "test1", output / "test1/metrics.json"
    if path.exists():
        result = read_json(path)
        if result["frozen_sha256"] != manifest["frozen_sha256"] or result["prediction_manifest_sha256"] != file_digest(target / "prediction-manifest.json"):
            raise ValueError("An existing final evaluation has incompatible provenance.")
        return result
    # Persisted, checksummed A/B/C and individual predictions exist before this read.
    plan = read_json(output / "plan.json")
    gold, gold_source = read_xlsx(location(plan["test1_gold"]), labeled=True)
    if len(gold) != 900 or gold_source["sha256"] != plan["test1_expected_gold_sha256"]:
        raise ValueError("Final comparison requires all 900 prescribed labels.")
    labels = np.asarray([r.label for r in gold])
    from gpt_classifier.data import group
    groups = np.asarray([group(r.text) for r in gold])
    predictions, metrics = {}, {}
    for name in manifest["files"]:
        rows = read_predictions(target / name)
        align(gold, rows)
        key = Path(name).stem
        predictions[key] = np.asarray([r["label"] for r in rows])
        metrics[key] = classification_metrics(labels, predictions[key])
        if "probabilities" in rows[0]:
            metrics[key] = describe_probabilities(labels, aligned_scores(rows, "probabilities"))
    if not np.isclose(metrics["A"]["f1_macro"], .49584421663197825, atol=1e-12, rtol=0):
        raise ValueError("Historical A macro-F1 does not match the verified 900-row reference.")
    stats = paired_statistics(labels, predictions, groups)
    result = {"status": "completed", "rows": 900, "gold": gold_source,
              "frozen_sha256": file_digest(output / "frozen.json"),
              "prediction_manifest_sha256": file_digest(target / "prediction-manifest.json"),
              "recommended_configuration": frozen["recommended_configuration"],
              "recommendation_changed_after_test1": False, "gold_read_after_all_predictions": True,
              "historical_A_matches_exactly": True, "metrics": metrics, "paired_statistics": stats,
              "complementarity": {"original_svm_vs_norberto": complementarity(labels, predictions["svm_original_calibrated"], predictions["norberto_unchanged"]),
                                  "optimized_svm_vs_norberto": complementarity(labels, predictions["svm_optimized_calibrated"], predictions["norberto_unchanged"])},
              "test2_accessed": False, "default_promoted": False,
              "limitations": ["Retrospective Test1 evaluation with historical exposure and known overlap with original training texts.",
                               "All 900 rows retained. Normalized-text bootstrap groups account for known exact/normalized duplicates, not all semantic dependence.",
                               "Multiple descriptive comparisons and adaptive research history preclude a fresh confirmatory superiority claim.",
                               "The development recommendation was frozen and cannot change based on Test1."]}
    load_frozen(output)
    if file_digest(location(plan["test1_gold"])) != gold_source["sha256"]:
        raise ValueError("Final gold workbook changed during evaluation.")
    for name, expected in manifest["files"].items():
        if file_digest(target / name) != expected:
            raise ValueError("Final predictions changed during evaluation.")
    write_json(path, result, replace=False)
    from .reporting import report
    report(output)
    return result
