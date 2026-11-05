"""Fit only a meta-LR on the old weight partition; keep the original soft run intact."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np

from gpt_classifier.data import (LABELS, Example, Text, canonical, digest, file_digest,
                                 read_jsonl, read_xlsx, write_json, write_jsonl)
from gpt_classifier.evaluation import align as align_labels, compare, metrics
from .files import align, read_json, read_predictions
from .soft_pipeline import implementation as soft_implementation
from .soft_pipeline import load_soft_artifact, prediction_rows, predict_soft, separate_holdout
from .soft_sources import collect_scores, holdout_records, location, model_identities, verify_lr_training
from .soft_voting import MODELS, aligned_scores, apply_calibration, model_order, soft_vote
from .soft_compatibility import compatible_fit_code
from .stacking import feature_names, fit_meta, meta_probabilities


def implementation(models=MODELS):
    root = Path(__file__).resolve().parents[1]
    return {**soft_implementation(models), **{str(p.relative_to(root)): file_digest(p)
            for p in sorted((root / "ensemble_classifier").glob("stacking*.py"))}}


def reference(path):
    root = Path(__file__).resolve().parents[2]
    path = Path(path).resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def cached_scores(config, records, source, identities, directory, phase):
    models = model_order(identities or MODELS)
    # Missing cache is an explicit error: fitting/comparing must not trigger neural inference.
    if any(not (directory / m / name).is_file() for m in models for name in ("run.json", "predictions.jsonl")):
        raise FileNotFoundError("Saved member scores are incomplete; restore them or run predict-soft/fit-soft separately.")
    return collect_scores(config, records, source, identities, directory, phase=phase)


def fit_stacking(config_path: Path, output: Path):
    if output.exists():
        raise FileExistsError("Use a new stacking fit directory; previous experiments are immutable.")
    config = read_json(config_path)
    if set(config) != {"soft_model", "soft_partition", "holdout_scores"}:
        raise ValueError("Stacking configuration requires soft_model, soft_partition and holdout_scores only.")
    soft_path, partition_path = location(config["soft_model"]), location(config["soft_partition"])
    soft = load_soft_artifact(soft_path)
    models = model_order(soft["model_identities"])
    soft_sha = file_digest(soft_path)
    records, groups, source, all_records, development = holdout_records(soft["score_configuration"])
    verify_lr_training(all_records, development)
    if file_digest(partition_path) != soft["partition_sha256"]:
        raise ValueError("Stacking must reuse the exact frozen calibration/weight partition.")
    partition = read_jsonl(partition_path)
    y = np.asarray([r.label for r in records])
    calibration, meta = separate_holdout(y, groups, soft["score_configuration"]["seed"])
    calibration_indices = set(calibration.tolist())
    expected = [{"excel_row": r.excel_row, "text_sha256": digest(r.text), "group_id": str(groups[i]),
                 "role": "calibration" if i in calibration_indices else "weight_selection"}
                for i, r in enumerate(records)]
    if partition != expected:
        raise ValueError("Saved holdout rows/groups/roles do not match the original source.")
    raw, provenance = cached_scores(soft["score_configuration"], [Text(r.excel_row, r.text) for r in records],
                                    source, soft["model_identities"], location(config["holdout_scores"]), "holdout")
    # Calibration rows are never fed to the meta-model, even during cross-validation.
    calibrated = {m: apply_calibration(raw[m][meta], soft["calibration"][m]) for m in models}
    model, search, fold_ids, oof = fit_meta(calibrated, y[meta], groups[meta])
    meta_records = [records[int(i)] for i in meta]
    assignments = {int(i): int(fold) for i, fold in zip(meta, fold_ids)}
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "partition.jsonl", [dict(row, stacking_role="calibration_only" if row["role"] == "calibration"
                   else "meta_training", meta_validation_fold=assignments.get(i)) for i, row in enumerate(partition)])
    write_json(output / "cv-search.json", search)
    write_jsonl(output / "meta-oof-predictions.jsonl", prediction_rows(meta_records, oof))
    artifact = {"version": 2, "model_type": "frozen_probability_stacking", "labels": list(LABELS),
                "models": list(models), "features": list(feature_names(models)), "meta_model": model,
                "soft_model": reference(soft_path), "soft_artifact_sha256": soft_sha,
                "partition_sha256": file_digest(output / "partition.jsonl"),
                "source_partition_sha256": soft["partition_sha256"], "fit_code": implementation(models),
                "runtime": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn")}}
    if (file_digest(soft_path) != soft_sha or file_digest(partition_path) != soft["partition_sha256"]
            or model_identities(models) != soft["model_identities"]):
        raise ValueError("Frozen inputs changed during meta training.")
    write_json(output / "model.json", artifact)
    write_json(output / "frozen_manifest.json", {"model_type": artifact["model_type"], "labels": list(LABELS),
               "artifact_sha256": file_digest(output / "model.json"),
               "files": {name: file_digest(output / name) for name in ("partition.jsonl", "cv-search.json", "meta-oof-predictions.jsonl")}})
    report = {"status": "completed", "base_training_performed": False, "calibration_fitted": False,
              "meta_training_performed": True, "test1_accessed": False, "test2_accessed": False,
              "artifact_sha256": file_digest(output / "model.json"), "soft_artifact_sha256": soft_sha,
              "selected_C": model["C"], "search": search, "score_provenance": provenance,
              "base_checkpoint_selection_used_holdout": any(
                  member.get("holdout_used_for_checkpoint_selection", False) for member in soft["model_identities"].values()),
              "partition": {"development_rows": 16093, "holdout_rows": len(records),
                            "calibration_rows": len(calibration), "meta_training_rows": len(meta),
                            "calibration_groups": len(set(groups[calibration])), "meta_training_groups": len(set(groups[meta])),
                            "shared_rows": 0, "shared_groups": 0, "meta_cv_folds": len(search["folds"]),
                            "class_counts": {role: {c: int(sum(y[idx] == c)) for c in LABELS}
                                             for role, idx in (("calibration", calibration), ("meta_training", meta))}},
              "meta_selection_oof_metrics": metrics(meta_records, prediction_rows(meta_records, oof)),
              "caveat": "OOF scores selected C; not an independent test. Holdout/test1 have historical exposure. No test1 labels enter training."}
    write_json(output / "run.json", report)
    return report


def load_stacking_artifact(path):
    artifact, contract = read_json(path), read_json(path.with_name("frozen_manifest.json"))
    models = model_order(artifact.get("models", ()))
    if (contract.get("artifact_sha256") != file_digest(path)
            or contract.get("model_type") != "frozen_probability_stacking"
            or artifact.get("version") not in (1, 2) or artifact.get("labels") != list(LABELS)
            or artifact.get("models") != list(models) or artifact.get("features") != list(feature_names(models))
            or not compatible_fit_code(path, artifact, implementation(models), "stacking")):
        raise ValueError("Frozen stacking artifact/code/class contract changed.")
    for name, sha in contract["files"].items():
        if file_digest(path.with_name(name)) != sha:
            raise ValueError("Stacking training provenance changed.")
    soft_path = location(artifact["soft_model"])
    soft = load_soft_artifact(soft_path)
    models = model_order(soft["model_identities"])
    if (file_digest(soft_path) != artifact["soft_artifact_sha256"]
            or soft["partition_sha256"] != artifact["source_partition_sha256"]
            or artifact["partition_sha256"] != file_digest(path.with_name("partition.jsonl"))):
        raise ValueError("Reference soft model/calibration/partition changed.")
    meta_probabilities({m: np.asarray([[.2, .3, .5]]) for m in models}, artifact["meta_model"])
    return artifact, soft


def saved_soft_inputs(soft_path, source_path, run):
    """Validate and reconstruct a saved soft run without accessing any gold labels."""
    soft = load_soft_artifact(soft_path)
    models = model_order(soft["model_identities"])
    records, source = read_xlsx(source_path)
    manifest = read_json(run / "run.json")
    if (manifest.get("status") != "completed" or manifest.get("rows") != len(records)
            or manifest.get("labels") != list(LABELS) or manifest.get("gold_accessed") is not False
            or manifest.get("base_training_performed") is not False or manifest.get("calibration_fitted") is not False
            or manifest.get("artifact_sha256") != file_digest(soft_path)
            or manifest.get("input", {}).get("sha256") != source["sha256"]
            or manifest.get("weights") != soft["weights"]
            or manifest.get("targets_sha256") != digest(canonical([(r.excel_row, digest(r.text)) for r in records]))):
        raise ValueError("Soft run differs from the frozen stacking reference or input.")
    raw, _ = cached_scores(soft["score_configuration"], records, source, soft["model_identities"], run / "members", "inference")
    calibrated = {m: apply_calibration(raw[m], soft["calibration"][m]) for m in models}
    paths = {"run.json": file_digest(run / "run.json")}
    predictions = {}
    for m in (*models, *(m + "_calibrated" for m in models), "soft_voting"):
        directory = run if m == "soft_voting" else (run / "calibrated" / m.removesuffix("_calibrated")
                     if m.endswith("_calibrated") else run / "members" / m)
        meta_path, pred_path = directory / "run.json", directory / "predictions.jsonl"
        metadata = read_json(meta_path)
        if metadata.get("predictions_sha256") != file_digest(pred_path):
            raise ValueError("Soft/member predictions changed after inference.")
        rows = read_predictions(pred_path)
        align(records, rows)
        align_labels(records, rows)
        if m == "soft_voting" or m.endswith("_calibrated"):
            expected = soft_vote(calibrated, soft["weights"]) if m == "soft_voting" else calibrated[m.removesuffix("_calibrated")]
            if (not np.allclose(aligned_scores(rows, "probabilities"), expected, atol=1e-12, rtol=0)
                    or [r["label"] for r in rows] != [r["label"] for r in prediction_rows(records, expected)]):
                raise ValueError("Saved probabilities/labels differ from frozen calibration/soft voting.")
        predictions[m] = rows
        paths[str(meta_path.relative_to(run))] = file_digest(meta_path)
        paths[str(pred_path.relative_to(run))] = file_digest(pred_path)
    return records, source, calibrated, predictions, paths


def predict_stacking(model_path: Path, source_path: Path, output: Path, *, soft_run=None, device="cpu",
                     live_norberto=False, live_qwen=False):
    if output.exists():
        raise FileExistsError("Use a new stacking prediction directory.")
    artifact, _ = load_stacking_artifact(model_path)
    soft_path = location(artifact["soft_model"])
    if soft_run is None:
        soft_run = output / "soft"
        predict_soft(soft_path, source_path, soft_run, device=device, live_norberto=live_norberto,
                     live_qwen=live_qwen)
    records, source, calibrated, _, snapshots = saved_soft_inputs(soft_path, source_path, soft_run)
    predictions = prediction_rows(records, meta_probabilities(calibrated, artifact["meta_model"]))
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "predictions.jsonl", predictions)
    report = {"status": "completed", "rows": len(records), "labels": list(LABELS), "input": source,
              "input_path": reference(source_path), "artifact_path": reference(model_path),
              "artifact_sha256": file_digest(model_path), "soft_run": reference(soft_run), "soft_files": snapshots,
              "soft_artifact_sha256": artifact["soft_artifact_sha256"],
              "base_training_performed": False, "meta_training_performed": False, "calibration_fitted": False,
              "gold_accessed": False, "predictions_sha256": file_digest(output / "predictions.jsonl")}
    if file_digest(source_path) != source["sha256"]:
        raise ValueError("Source changed during stacking prediction.")
    write_json(output / "run.json", report)
    return report


def evaluate_stacking(run: Path, gold_path: Path, output: Path):
    if output.exists():
        raise FileExistsError("Use a new final stacking evaluation directory.")
    manifest = read_json(run / "run.json")
    if (manifest.get("status") != "completed" or manifest.get("rows") != 900
            or manifest.get("gold_accessed") is not False
            or manifest.get("meta_training_performed") is not False
            or file_digest(run / "predictions.jsonl") != manifest.get("predictions_sha256")):
        raise ValueError("A completed immutable 900-row stacking prediction run is required.")
    model_path = location(manifest["artifact_path"])
    artifact, soft = load_stacking_artifact(model_path)
    models = model_order(artifact["models"])
    if file_digest(model_path) != manifest["artifact_sha256"]:
        raise ValueError("Stacking changed after prediction.")
    records, source, calibrated, predictions, snapshots = saved_soft_inputs(
        location(artifact["soft_model"]), location(manifest["input_path"]), location(manifest["soft_run"]))
    if source["sha256"] != manifest["input"]["sha256"] or snapshots != manifest["soft_files"]:
        raise ValueError("Reference predictions changed after stacking inference.")
    predictions["stacking"] = read_predictions(run / "predictions.jsonl")
    expected = meta_probabilities(calibrated, artifact["meta_model"])
    if predictions["stacking"] != prediction_rows(records, expected):
        raise ValueError("Stacking predictions differ from the frozen meta-model.")
    # The first labeled read is deliberately after validation of every prediction.
    gold, info = read_xlsx(gold_path, labeled=True)
    for rows in predictions.values():
        align_labels(gold, rows)
    train, _ = read_xlsx(location(soft["score_configuration"]["train"]), labeled=True)
    paired = {}
    for name in ("soft_voting", *models):
        baseline = [Example(r.excel_row, r.text, p["label"]) for r, p in zip(records, predictions[name])]
        comparison = compare(gold, predictions["stacking"], baseline, train)
        paired[name] = {"accuracy_delta": comparison["accuracy_delta"], **comparison["mcnemar"]}
    cases = []
    for truth, soft_row, stack_row in zip(gold, predictions["soft_voting"], predictions["stacking"]):
        if soft_row["label"] != stack_row["label"]:
            category = ("corrected" if stack_row["label"] == truth.label else
                        "harmed" if soft_row["label"] == truth.label else "both_wrong_changed_label")
            cases.append({"excel_row": truth.excel_row, "text_sha256": digest(truth.text), "resp_text": truth.text,
                          "gold": truth.label, "soft_voting": soft_row["label"], "stacking": stack_row["label"],
                          "category": category})
    scores = {name: metrics(gold, rows) for name, rows in predictions.items()}
    report = {"status": "completed", "rows": len(gold), "labels": list(LABELS), "metrics": scores,
              "paired_stacking_vs": paired, "changes_vs_soft": {name: sum(c["category"] == name for c in cases)
                  for name in ("corrected", "harmed", "both_wrong_changed_label")},
              "artifact_sha256": manifest["artifact_sha256"], "soft_artifact_sha256": artifact["soft_artifact_sha256"],
              "predictions_sha256": manifest["predictions_sha256"], "gold_sha256": info["sha256"],
              "gold_read_after_all_predictions": True, "parameters_updated": False,
              "protocol": "Isolated exploratory test1 comparison; historical exposure and repeated texts limit inference. No test2-before-test1 claim; do not tune on these results."}
    if file_digest(gold_path) != info["sha256"]:
        raise ValueError("Gold changed during evaluation.")
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "changes-vs-soft.jsonl", cases)
    write_json(output / "metrics.json", report)
    return report
