"""Two disjoint holdout roles, frozen soft ensemble, then separate final evaluation."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.model_selection import StratifiedGroupKFold

from gpt_classifier.backend import file_lock
from gpt_classifier.data import LABELS, Text, canonical, digest, file_digest, read_xlsx, write_json, write_jsonl
from gpt_classifier.evaluation import metrics
from .files import align, read_json, read_predictions
from .soft_sources import (DATASET_SHA256, SPLIT_SHA256, collect_scores, holdout_records,
                           load_soft_config, location, model_identities, verify_lr_training)
from .soft_voting import MODELS, apply_calibration, choose_weights, fit_calibrations, model_order, probability_metrics, soft_vote
from .soft_compatibility import compatible_fit_code


def implementation(models=MODELS):
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "ensemble_classifier").glob("soft_*.py"))
    for package in ("lexical_classifier", "norberto_classifier"):
        paths.extend((root / package).glob("*.py"))
    if "qwen" in models:
        paths.extend((root / "qwen_classifier").glob("*.py"))
    paths.extend([root / "ensemble_classifier/files.py", root / "gpt_classifier/data.py"])
    return {str(p.relative_to(root)): file_digest(p) for p in sorted(paths)}


def prediction_rows(records, p):
    return [{"excel_row": r.excel_row, "text_sha256": digest(r.text),
             "label": LABELS[int(scores.argmax())], "probabilities": dict(zip(LABELS, scores.tolist()))}
            for r, scores in zip(records, p)]


def separate_holdout(labels, groups, seed):
    labels, groups = np.asarray(labels), np.asarray(groups)
    if len(labels) != len(groups) or len(set(groups)) < 6:
        raise ValueError("Insufficient or misaligned calibration/weight groups.")
    calibration, weights = next(StratifiedGroupKFold(2, shuffle=True, random_state=seed)
                                .split(np.zeros(len(labels)), labels, groups))
    if (set(groups[calibration]) & set(groups[weights])
            or set(calibration) & set(weights)
            or set(labels[calibration]) != set(LABELS) or set(labels[weights]) != set(LABELS)):
        raise ValueError("Calibration and weight-selection data are not isolated/stratified.")
    return calibration, weights


def fit_soft(config_path: Path, output: Path, *, device="cpu"):
    config = load_soft_config(config_path)
    models = model_order(config.get("models", MODELS))
    records, groups, source, all_records, development = holdout_records(config)
    identities = model_identities(models)
    verify_lr_training(all_records, development)
    identity = {"config": config, "models": identities, "source_sha256": source["sha256"],
                "implementation": implementation(models), "runtime": {p: importlib.metadata.version(p)
                    for p in ("numpy", "scipy", "scikit-learn", "joblib")}}
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".lock"):
        plan_path = output / "plan.json"
        if plan_path.exists() and read_json(plan_path) != identity:
            raise ValueError("Output belongs to another soft-voting experiment; choose a new directory.")
        write_json(plan_path, identity)
        if (output / "run.json").exists() and read_json(output / "run.json").get("status") == "completed":
            load_soft_artifact(output / "model.json")
            return read_json(output / "run.json")
        labels = np.asarray([r.label for r in records])
        calibration, weights = separate_holdout(labels, groups, config["seed"])
        roles = {int(i): "calibration" for i in calibration}
        roles.update({int(i): "weight_selection" for i in weights})
        write_jsonl(output / "partition.jsonl", [{"excel_row": r.excel_row, "text_sha256": digest(r.text),
                    "group_id": str(groups[i]), "role": roles[i]} for i, r in enumerate(records)])
        report = {"status": "preparing", "rows": len(records), "labels": list(LABELS),
                  "base_training_performed": False, "calibration_fitted": False,
                  "base_checkpoint_selection_used_holdout": any(
                      member.get("holdout_used_for_checkpoint_selection", False) for member in identities.values()),
                  "test1_accessed": False, "test2_accessed": False, "models": identities,
                  "partition": {"development_rows": 16093, "holdout_rows": len(records),
                                "calibration_rows": len(calibration), "weight_selection_rows": len(weights),
                                "calibration_groups": len(set(groups[calibration])),
                                "weight_selection_groups": len(set(groups[weights])),
                                "shared_rows": 0, "shared_groups": 0,
                                "class_counts": {role: {label: int(sum(labels[indices] == label)) for label in LABELS}
                                                 for role, indices in (("calibration", calibration), ("weight_selection", weights))}},
                  "caveat": "The holdout has historical exposure and selected the Qwen checkpoint when Qwen is included. Calibration and weight-selection scores are development diagnostics, not independent final performance. No test1 labels are used."}
        write_json(output / "run.json", report)
        try:
            raw, provenance = collect_scores(config, [Text(r.excel_row, r.text) for r in records], source,
                                              identities, output / "scores", phase="holdout", device=device)
            calibrators = fit_calibrations({m: raw[m][calibration] for m in models},
                                          labels[calibration], groups[calibration], seed=config["seed"])
            report["calibration_fitted"] = True
            calibrated = {m: apply_calibration(raw[m], calibrators[m]) for m in models}
            selected, trials = choose_weights({m: calibrated[m][weights] for m in models}, labels[weights])
            write_json(output / "weight-search.json", {"criterion": "F1-macro; tie: accuracy, then closest to equal weights",
                       "grid": f"quarter simplex plus uniform weights; {len(trials)} fixed candidates", "trials": trials})
            diagnostic = {}
            selection_records = [records[int(i)] for i in weights]
            for m in models:
                native = softmax(raw[m], axis=1) if m == "svm" else raw[m]
                diagnostic[m] = {"before": probability_metrics(labels[weights], native[weights]),
                                 "after": probability_metrics(labels[weights], calibrated[m][weights]),
                                 "before_note": "SVM softmax margins are an uncalibrated diagnostic surrogate, not predict_proba" if m == "svm" else "native probabilities",
                                 "metrics_after": metrics(selection_records, prediction_rows(selection_records, calibrated[m][weights]))}
            mixed = soft_vote(calibrated, selected)
            report.update(status="completed", weights=selected, calibration=calibrators,
                          fixed_candidate_count=len(trials), probability_diagnostics_on_weight_partition=diagnostic,
                          selection_metrics=metrics(selection_records, prediction_rows(selection_records, mixed[weights])),
                          partition_sha256=file_digest(output / "partition.jsonl"), score_provenance=provenance)
            artifact = {"version": 2, "labels": list(LABELS), "models": list(models), "weights": selected,
                        "calibration": calibrators, "model_identities": identities, "score_configuration": config,
                        "source_sha256": DATASET_SHA256, "split_sha256": SPLIT_SHA256,
                        "partition_sha256": report["partition_sha256"], "fit_code": implementation(models),
                        "selection_criterion": "macro_f1_on_disjoint_weight_partition", "candidate_count": len(trials)}
            if model_identities(models) != identities or file_digest(location(config["train"])) != source["sha256"]:
                raise ValueError("Base models or training source changed during fitting.")
            write_json(output / "model.json", artifact)
            write_json(output / "frozen_manifest.json", {"artifact_sha256": file_digest(output / "model.json"),
                       "labels": list(LABELS), "model_type": "weighted_soft_voting"})
            report["artifact_sha256"] = file_digest(output / "model.json")
            write_json(output / "run.json", report)
            return report
        except BaseException:
            report["status"] = "failed"
            write_json(output / "run.json", report)
            raise


def load_soft_artifact(path):
    artifact = read_json(path)
    contract = read_json(path.with_name("frozen_manifest.json"))
    models = model_order(artifact.get("models", ()))
    if contract.get("artifact_sha256") != file_digest(path):
        raise ValueError("Frozen soft-voting artifact changed.")
    if (artifact.get("version") not in (1, 2) or artifact.get("labels") != list(LABELS)
            or artifact.get("models") != list(models)
            or not compatible_fit_code(path, artifact, implementation(models), "soft")
            or artifact.get("model_identities") != model_identities(models)
            or artifact.get("source_sha256") != DATASET_SHA256 or artifact.get("split_sha256") != SPLIT_SHA256):
        raise ValueError("Soft-voting model/code/class provenance is incompatible.")
    sample = {m: np.asarray([[.2, .3, .5]]) for m in models}
    calibrated = {m: apply_calibration(sample[m], artifact["calibration"][m]) for m in models}
    soft_vote(calibrated, artifact["weights"])
    return artifact


def predict_soft(artifact_path: Path, source_path: Path, output: Path, *, device="cpu", live_norberto=False,
                 live_qwen=False):
    artifact = load_soft_artifact(artifact_path)
    models = model_order(artifact["models"])
    records, source = read_xlsx(source_path)  # Even a labeled source yields Text objects only.
    identity = {"artifact_sha256": file_digest(artifact_path), "input": source,
                "targets_sha256": digest(canonical([(r.excel_row, digest(r.text)) for r in records]))}
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".lock"):
        if (output / "plan.json").exists() and read_json(output / "plan.json") != identity:
            raise ValueError("Prediction output belongs to another source or ensemble.")
        write_json(output / "plan.json", identity)
        config = dict(artifact["score_configuration"])
        if live_norberto:
            config["recorded_norberto"] = {}
        if live_qwen:
            config["recorded_qwen"] = {}
        phase = "test1" if source["sha256"] == "e626f030d4bf2be889fc41dd1fe9b81fb31f1f50f3ace50530d658683260449f" else "inference"
        raw, provenance = collect_scores(config, records, source, artifact["model_identities"],
                                          output / "members", phase=phase, device=device)
        calibrated = {m: apply_calibration(raw[m], artifact["calibration"][m]) for m in models}
        # Persist all individual and ensemble decisions before evaluation can read gold.
        for m in models:
            directory = output / "calibrated" / m
            directory.mkdir(parents=True, exist_ok=True)
            write_jsonl(directory / "predictions.jsonl", prediction_rows(records, calibrated[m]))
            calibrated_identity = digest(canonical({"base_model": artifact["model_identities"][m],
                                                    "calibration": artifact["calibration"][m]}))
            write_json(directory / "run.json", {"status": "completed", "input": source, "labels": list(LABELS),
                       "artifact_sha256": calibrated_identity,
                       "base_artifact_sha256": artifact["model_identities"][m]["artifact_sha256"],
                       "calibration_artifact_sha256": identity["artifact_sha256"], "training_performed": False,
                       "predictions_sha256": file_digest(directory / "predictions.jsonl")})
            write_json(directory / "frozen_manifest.json", {"artifact_sha256": calibrated_identity,
                       "labels": list(LABELS), "model_type": "calibrated_frozen_member"})
        mixed = soft_vote(calibrated, artifact["weights"])
        write_jsonl(output / "predictions.jsonl", prediction_rows(records, mixed))
        if file_digest(source_path) != source["sha256"] or file_digest(artifact_path) != identity["artifact_sha256"]:
            raise ValueError("Input or frozen ensemble changed during inference.")
        report = {"status": "completed", **identity, "rows": len(records), "labels": list(LABELS),
                  "artifact_path": str(artifact_path.resolve()),
                  "ensemble_identity": identity["artifact_sha256"], "weights": artifact["weights"],
                  "base_training_performed": False, "calibration_fitted": False, "gold_accessed": False,
                  "members": provenance, "predictions_sha256": file_digest(output / "predictions.jsonl")}
        write_json(output / "run.json", report)
        return report


def evaluate_soft(run: Path, gold_path: Path, output: Path):
    """Read the 900 labels only after every prediction file is committed to disk."""
    manifest = read_json(run / "run.json")
    if manifest.get("status") != "completed" or manifest.get("rows") != 900 or manifest.get("gold_accessed") is not False:
        raise ValueError("Final comparison requires a completed, gold-free 900-row prediction run.")
    if file_digest(run / "predictions.jsonl") != manifest.get("predictions_sha256"):
        raise ValueError("Ensemble predictions changed before evaluation.")
    artifact_path = Path(manifest["artifact_path"])
    artifact = load_soft_artifact(artifact_path)
    models = model_order(artifact["models"])
    if file_digest(artifact_path) != manifest["artifact_sha256"] or artifact["weights"] != manifest["weights"]:
        raise ValueError("The ensemble changed after prediction.")
    all_predictions = {"ensemble": read_predictions(run / "predictions.jsonl")}
    for m in models:
        all_predictions[m] = read_predictions(run / "members" / m / "predictions.jsonl")
        all_predictions[m + "_calibrated"] = read_predictions(run / "calibrated" / m / "predictions.jsonl")
        member = read_json(run / "members" / m / "run.json")
        calibrated = read_json(run / "calibrated" / m / "run.json")
        if (file_digest(run / "members" / m / "predictions.jsonl") != member["predictions_sha256"]
                or file_digest(run / "calibrated" / m / "predictions.jsonl") != calibrated["predictions_sha256"]):
            raise ValueError("Member predictions changed before evaluation.")
    if output.exists():
        raise FileExistsError("Use a new final evaluation directory; no retuning on test1.")
    gold, info = read_xlsx(gold_path, labeled=True)
    if len(gold) != 900:
        raise ValueError("Final evaluation requires all 900 professor-labeled records.")
    records = [Text(r.excel_row, r.text) for r in gold]
    for predictions in all_predictions.values():
        align(records, predictions)
    report = {"status": "completed", "rows": 900, "labels": list(LABELS),
              "ensemble_artifact_sha256": manifest["artifact_sha256"], "gold_sha256": info["sha256"],
              "gold_read_after_all_predictions": True, "parameters_updated": False,
              "weights": manifest["weights"], "metrics": {name: metrics(gold, predictions)
                                                            for name, predictions in all_predictions.items()},
              "caveat": "All parameters were frozen before this evaluation. Test1 has historical exposure elsewhere in the project; do not retune or claim it was historically untouched. Raw and calibrated individuals are reported separately."}
    if file_digest(gold_path) != info["sha256"]:
        raise ValueError("Gold workbook changed during evaluation.")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "metrics.json", report)
    return report
