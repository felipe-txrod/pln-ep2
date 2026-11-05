"""Check overlap row counting and label ambiguity without real datasets."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from gpt_classifier.data import Example, Text

source = Path(__file__).resolve().parents[1] / "scripts/diagnostics/audit-dataset-overlap.py"
spec = importlib.util.spec_from_file_location("dataset_overlap", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def dataset(name, records, labeled=False):
    return module.Dataset(name, records, {"sha256": name, "rows": len(records)}, labeled)


def test_matching_rows_count_each_occurrence_once():
    left = dataset("train", [Text(2, "same"), Text(3, "same"), Text(4, "unique")])
    right = dataset("test", [Text(2, "same"), Text(3, "same"), Text(4, "same")])
    report = module.compare(left, right)
    assert report["exact"] == {
        "shared_unique_texts": 1, "left_matching_rows": 2, "right_matching_rows": 3}
    assert report["same_text_multiset"] is False


def test_normalization_and_ambiguous_training_labels():
    left = dataset("train", [Example(2, "A &amp; B", "c1"), Example(3, "a & b", "c5"),
                              Example(4, "other", "c234")], True)
    right = dataset("gold", [Example(2, " A\n& B ", "c1"), Example(3, "other", "c234"),
                              Example(4, "other", "c5"), Example(5, "unseen", "c1")], True)
    report = module.compare(left, right)
    assert report["exact"]["right_matching_rows"] == 2
    assert report["normalized"]["right_matching_rows"] == 3
    assert report["normalized"]["right_rows_vs_left_labels"] == {
        "only_same_label": 1, "same_and_other_labels": 1, "only_other_labels": 1}
    summary = module.describe(left)
    assert summary["normalized"]["conflicting_label_groups"] == 1
    assert summary["normalized"]["rows_in_conflicting_label_groups"] == 2
    assert module.normalize_text("AÇÃO!") != module.normalize_text("acao")


def test_content_identity_checks_order_and_labels():
    left = dataset("left", [Example(2, "a", "c1"), Example(3, "b", "c5")], True)
    reordered = dataset("reordered", [Example(2, "b", "c5"), Example(3, "a", "c1")], True)
    changed = dataset("changed", [Example(2, "a", "c5"), Example(3, "b", "c5")], True)
    assert module.compare(left, reordered)["same_text_multiset"] is True
    assert module.compare(left, reordered)["same_texts_in_order"] is False
    assert module.compare(left, changed)["same_texts_in_order"] is True
    assert module.compare(left, changed)["same_texts_and_labels_in_order"] is False
