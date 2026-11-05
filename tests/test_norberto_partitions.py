from __future__ import annotations

import csv
import hashlib
from types import SimpleNamespace

import pytest

from norberto_classifier.partitions import development_partition


def make_split(tmp_path, *, change=None):
    path = tmp_path / "split.csv"
    rows = [{"excel_row": i + 2, "group_id": str(i), "role": "development" if i < 3 else "holdout"}
            for i in range(6)]
    if change:
        change(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("excel_row", "group_id", "role"))
        writer.writeheader()
        writer.writerows(rows)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def records():
    return [SimpleNamespace(excel_row=i + 2, text=f"response {i}", label=("c1", "c234", "c5")[i % 3])
            for i in range(6)]


def test_frozen_selection_preserves_order_and_excludes_holdout(tmp_path):
    path, digest = make_split(tmp_path)
    source = records()
    selected, report = development_partition(source, path, digest, development_rows=3, holdout_rows=3)
    assert selected == source[:3]
    assert [r.excel_row for r in selected] == [2, 3, 4]
    assert not {r.excel_row for r in selected} & {r.excel_row for r in source[3:]}
    assert report["reserved_holdout_rows"] == 3
    assert report["holdout_group_overlap"] == report["holdout_row_overlap"] == 0


def test_modified_split_is_rejected_before_selection(tmp_path):
    path, digest = make_split(tmp_path)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256"):
        development_partition(records(), path, digest, development_rows=3, holdout_rows=3)


@pytest.mark.parametrize("change,message", [
    (lambda rows: rows[0].update(excel_row=9), "IDs and order"),
    (lambda rows: rows[3].update(group_id="0"), "groups overlap"),
    (lambda rows: rows[3].update(role="test"), "roles"),
    (lambda rows: rows[3].update(role="development"), "counts"),
    (lambda rows: rows[0].update(group_id=""), "group IDs"),
    (lambda rows: rows.pop(), "IDs and order"),
])
def test_invalid_partition_is_rejected(tmp_path, change, message):
    path, digest = make_split(tmp_path, change=change)
    with pytest.raises(ValueError, match=message):
        development_partition(records(), path, digest, development_rows=3, holdout_rows=3)


def test_missing_class_in_development_is_rejected(tmp_path):
    path, digest = make_split(tmp_path)
    source = records()
    source[0].label = "c5"
    with pytest.raises(ValueError, match="labels"):
        development_partition(source, path, digest, development_rows=3, holdout_rows=3)
