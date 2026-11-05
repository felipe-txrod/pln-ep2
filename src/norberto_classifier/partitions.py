"""Select the immutable EP1 development partition without model dependencies."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Sequence

from .artifacts import LABELS


def development_partition(records: Sequence[Any], split_path: Path, expected_sha256: str, *,
                          development_rows: int, holdout_rows: int) -> tuple[list[Any], dict[str, Any]]:
    with split_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256:
            raise ValueError("The split differs from the frozen EP1 SHA-256.")
    with split_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if (not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames))
                or not {"excel_row", "group_id", "role"} <= set(reader.fieldnames)):
            raise ValueError("The split needs unique row, group, and role columns.")
        rows = list(reader)
    if len(rows) != len(records) or [int(row["excel_row"]) for row in rows] != [r.excel_row for r in records]:
        raise ValueError("Split rows do not match the original workbook IDs and order.")
    if len({r.excel_row for r in records}) != len(records):
        raise ValueError("Duplicate workbook row IDs are not allowed.")
    if any(row["role"] not in {"development", "holdout"} or not row["group_id"] for row in rows):
        raise ValueError("Split roles or group IDs are invalid.")
    selected = [r for r, row in zip(records, rows) if row["role"] == "development"]
    held_out = [r for r, row in zip(records, rows) if row["role"] == "holdout"]
    if len(selected) != development_rows or len(held_out) != holdout_rows:
        raise ValueError("Development/holdout counts differ from the frozen EP1 contract.")
    groups = {role: {row["group_id"] for row in rows if row["role"] == role}
              for role in ("development", "holdout")}
    if groups["development"] & groups["holdout"]:
        raise ValueError("Text groups overlap between development and holdout.")
    if any({r.label for r in partition} != set(LABELS) for partition in (selected, held_out)):
        raise ValueError("Both partitions must preserve all original labels.")
    return selected, {"split_sha256": expected_sha256, "source_rows": len(records),
                      "training_rows": len(selected), "reserved_holdout_rows": len(held_out),
                      "training_groups": len(groups["development"]), "holdout_groups": len(groups["holdout"]),
                      "holdout_row_overlap": 0, "holdout_group_overlap": 0}
