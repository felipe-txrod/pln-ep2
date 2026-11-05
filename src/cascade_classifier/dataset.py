"""Audited saved scores and nested grouped OOF predictions for routing development."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from ensemble_classifier.soft_sources import holdout_records, location, verify_lr_training
from ensemble_classifier.soft_voting import QWEN_MODELS, apply_calibration
from ensemble_classifier.stacking import fit_meta, meta_probabilities
from ensemble_classifier.stacking_pipeline import cached_scores, load_stacking_artifact
from ensemble_classifier.files import read_json
from gpt_classifier.data import LABELS, Text, digest, file_digest, read_jsonl


def validation_inputs(config):
    model_path = location(config.stacking_model)
    artifact, soft = load_stacking_artifact(model_path)
    if artifact["models"] != list(QWEN_MODELS):
        raise ValueError("The cascade requires the selected four-member stacking.")
    records, groups, source, all_records, development = holdout_records(soft["score_configuration"])
    verify_lr_training(all_records, development)
    partition_path = model_path.with_name("partition.jsonl")
    partition = read_jsonl(partition_path)
    if ([row["excel_row"] for row in partition] != [record.excel_row for record in records]
            or any(row["text_sha256"] != digest(record.text) or row["group_id"] != str(group)
                   for row, record, group in zip(partition, records, groups))):
        raise ValueError("Stacking partition/source alignment failed.")
    meta = np.asarray([i for i, row in enumerate(partition) if row["stacking_role"] == "meta_training"])
    calibration = np.asarray([i for i, row in enumerate(partition) if row["stacking_role"] == "calibration_only"])
    if (len(meta) != 1972 or len(calibration) != 2027
            or set(groups[meta]) & set(groups[calibration])):
        raise ValueError("The saved calibration/meta roles are not isolated.")
    raw, _ = cached_scores(soft["score_configuration"], [Text(r.excel_row, r.text) for r in records],
                          source, soft["model_identities"], location(config.holdout_scores), "holdout")
    calibrated = {name: apply_calibration(raw[name][meta], soft["calibration"][name]) for name in QWEN_MODELS}
    native_labels = {name: np.asarray(LABELS)[raw[name][meta].argmax(axis=1)].tolist() for name in QWEN_MODELS}
    selected = [records[int(i)] for i in meta]
    evidence = {"source_sha256": source["sha256"], "stacking_artifact_sha256": file_digest(model_path),
                "soft_artifact_sha256": artifact["soft_artifact_sha256"], "partition_sha256": file_digest(partition_path),
                "base_models_retrained": False, "development_rows": 16093, "calibration_rows": 2027,
                "routing_validation_rows": 1972, "shared_calibration_meta_groups": 0,
                "inherited_exposure": "Qwen checkpoint selection and historical research used the holdout; it is not untouched full-process validation."}
    return selected, groups[meta], calibrated, native_labels, evidence


def nested_predictions(records, groups, matrices):
    """Select C using inner folds; each outer target is excluded from fitting and C selection."""
    labels = np.asarray([record.label for record in records])
    groups = np.asarray(groups)
    oof = np.full((len(records), 3), np.nan)
    assignments = np.full(len(records), -1, dtype=int)
    audits = []
    for fold, (train, valid) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42)
                                         .split(np.zeros(len(records)), labels, groups)):
        if set(groups[train]) & set(groups[valid]) or np.any(assignments[valid] != -1):
            raise ValueError("Outer routing folds are not group-disjoint.")
        model, search, _, _ = fit_meta({name: part[train] for name, part in matrices.items()},
                                      labels[train], groups[train])
        oof[valid] = meta_probabilities({name: part[valid] for name, part in matrices.items()}, model)
        assignments[valid] = fold
        audits.append({"fold": fold, "model": model, "inner_search": search,
                       "training_ids": [records[int(i)].excel_row for i in train],
                       "validation_ids": [records[int(i)].excel_row for i in valid], "shared_groups": 0})
    if not np.isfinite(oof).all() or np.any(assignments < 0):
        raise ValueError("Incomplete nested OOF coverage.")
    return oof, assignments, audits
