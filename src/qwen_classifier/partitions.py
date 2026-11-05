"""Reuse the immutable EP1 development/holdout partition without resampling."""
from __future__ import annotations

import csv
from pathlib import Path

from gpt_classifier.data import LABELS, Example, file_digest, group

EP1_DATASET_SHA256 = "0e9219233664ac675fc47982bbc822acd3bb15d67fb7bfc26b67bec047e46818"
EP1_SPLIT_SHA256 = "0f78e7a809c7b7584db05b33cf03a73ff586d5de02ba824c16562d5070d41494"


def validate_ep1_sources(source: Path, split_path: Path) -> None:
    """Reject changed data or split files before running GPU preflight or training."""
    if file_digest(source) != EP1_DATASET_SHA256:
        raise ValueError("The training workbook differs from the frozen EP1 dataset.")
    if file_digest(split_path) != EP1_SPLIT_SHA256:
        raise ValueError("The partition differs from the frozen EP1 split.")


def split_ep1(records: list[Example], source: Path, split_path: Path
              ) -> tuple[list[Example], list[Example], dict]:
    """Verify complete ordered membership, class coverage, and group isolation."""
    validate_ep1_sources(source, split_path)
    with split_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if (reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames)
                or not {"excel_row", "group_id", "role"} <= set(reader.fieldnames)):
            raise ValueError("Frozen split requires unique excel_row, group_id, and role columns.")
        rows = list(reader)
    try:
        row_ids = [int(row["excel_row"]) for row in rows]
    except (TypeError, ValueError) as error:
        raise ValueError("Frozen split contains an invalid Excel row ID.") from error
    if len(rows) != len(records) or row_ids != [record.excel_row for record in records]:
        raise ValueError("Frozen split row IDs must cover the workbook exactly in original order.")
    if any(row["role"] not in {"development", "holdout"}
           or not isinstance(row["group_id"], str) or not row["group_id"].strip() for row in rows):
        raise ValueError("Frozen split has invalid roles or empty group IDs.")
    train = [record for record, row in zip(records, rows) if row["role"] == "development"]
    validation = [record for record, row in zip(records, rows) if row["role"] == "holdout"]
    train_groups = {row["group_id"] for row in rows if row["role"] == "development"}
    validation_groups = {row["group_id"] for row in rows if row["role"] == "holdout"}
    if train_groups & validation_groups:
        raise ValueError("Frozen similarity groups overlap between training and holdout.")
    if {group(record.text) for record in train} & {group(record.text) for record in validation}:
        raise ValueError("Normalized response texts overlap between training and holdout.")
    if any({record.label for record in partition} != set(LABELS) for partition in (train, validation)):
        raise ValueError("Every frozen partition must contain all predefined classes.")
    return train, validation, {
        "method": "frozen_ep1_development_holdout", "source_sha256": EP1_DATASET_SHA256,
        "split_path": str(split_path), "split_sha256": EP1_SPLIT_SHA256,
        "train_role": "development", "validation_role": "holdout",
        "train_groups": len(train_groups), "validation_groups": len(validation_groups),
        "row_overlap": 0, "group_overlap": 0, "normalized_text_overlap": 0,
        "holdout_used_for_fit": False,
        "holdout_used_for_checkpoint_selection": True,
    }
