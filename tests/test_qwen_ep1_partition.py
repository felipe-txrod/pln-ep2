"""Verify frozen EP1 membership, leakage rejection, and training CLI propagation."""
from __future__ import annotations

import csv
from dataclasses import replace
import json

import pytest

from gpt_classifier.data import LABELS, file_digest, read_jsonl, read_xlsx
from qwen_classifier.cli import main
from qwen_classifier.config import Config
from qwen_classifier.dataset import prepare
from qwen_classifier import partitions
from qwen_classifier.training import train


def write_split(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("excel_row", "group_id", "role"))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def frozen_partition(make_xlsx, tmp_path, monkeypatch):
    source = make_xlsx("source.xlsx", [(f"independent response {index}", LABELS[index % 3])
                                       for index in range(12)])
    rows = [{"excel_row": index + 2, "group_id": str(index),
             "role": "development" if index < 9 else "holdout"} for index in range(12)]
    split_path = tmp_path / "split.csv"
    write_split(split_path, rows)
    monkeypatch.setattr(partitions, "EP1_DATASET_SHA256", file_digest(source))
    monkeypatch.setattr(partitions, "EP1_SPLIT_SHA256", file_digest(split_path))
    return source, split_path, rows


def test_frozen_preparation_preserves_ids_snapshot_and_checkpoint_policy(frozen_partition, tmp_path):
    source, split_path, _ = frozen_partition
    output = tmp_path / "prepared"
    manifest = prepare(source, output, Config(), ep1_split=split_path)
    assert manifest["train_rows"] == list(range(2, 11))
    assert manifest["validation_rows"] == [11, 12, 13]
    assert manifest["split_method"] == "frozen_ep1_development_holdout"
    assert manifest["partition"]["group_overlap"] == 0
    assert manifest["partition"]["holdout_used_for_fit"] is False
    assert manifest["partition"]["holdout_used_for_checkpoint_selection"] is True
    assert (output / "frozen-split.csv").read_bytes() == split_path.read_bytes()
    assert [row["excel_row"] for row in read_jsonl(output / "train.jsonl")] == list(range(2, 11))
    assert [row["excel_row"] for row in read_jsonl(output / "validation.jsonl")] == [11, 12, 13]
    assert prepare(source, tmp_path / "different-seed", replace(Config(), seed=7),
                   ep1_split=split_path)["train_rows"] == manifest["train_rows"]
    with pytest.raises(FileExistsError):
        prepare(source, output, Config(), ep1_split=split_path)


@pytest.mark.parametrize("corruption,message", [
    ("reordered", "original order"), ("duplicate", "original order"),
    ("missing", "original order"), ("role", "invalid roles"),
    ("empty_group", "empty group"), ("shared_group", "similarity groups overlap"),
    ("row_id", "invalid Excel row ID"),
])
def test_frozen_membership_rejects_structural_corruption(frozen_partition, tmp_path, monkeypatch, corruption, message):
    source, split_path, rows = frozen_partition
    if corruption == "reordered":
        rows.reverse()
    elif corruption == "duplicate":
        rows[-1]["excel_row"] = rows[-2]["excel_row"]
    elif corruption == "missing":
        rows.pop()
    elif corruption == "role":
        rows[-1]["role"] = "test"
    elif corruption == "empty_group":
        rows[-1]["group_id"] = ""
    elif corruption == "shared_group":
        rows[-1]["group_id"] = rows[0]["group_id"]
    else:
        rows[-1]["excel_row"] = "invalid"
    write_split(split_path, rows)
    monkeypatch.setattr(partitions, "EP1_SPLIT_SHA256", file_digest(split_path))
    with pytest.raises(ValueError, match=message):
        prepare(source, tmp_path / "rejected", Config(), ep1_split=split_path)
    assert not (tmp_path / "rejected").exists()


def test_normalized_text_leakage_and_missing_classes_are_rejected(frozen_partition):
    source, split_path, _ = frozen_partition
    records, _ = read_xlsx(source, labeled=True)
    corrupted = [*records[:-1], replace(records[-1], text=records[0].text.upper())]
    with pytest.raises(ValueError, match="Normalized response texts overlap"):
        partitions.split_ep1(corrupted, source, split_path)
    corrupted = [replace(record, label="c1") for record in records]
    with pytest.raises(ValueError, match="all predefined classes"):
        partitions.split_ep1(corrupted, source, split_path)


def test_changed_split_is_rejected_before_gpu_preflight(frozen_partition, tmp_path, monkeypatch):
    import qwen_classifier.training as training
    source, split_path, _ = frozen_partition
    split_path.write_text(split_path.read_text() + "\n")
    monkeypatch.setattr(training, "smoke_test", lambda *args: pytest.fail("GPU preflight must not run"))
    with pytest.raises(ValueError, match="frozen EP1 split"):
        train(Config(), source, tmp_path / "training", tmp_path, ep1_split=split_path)
    assert not (tmp_path / "training").exists()


def test_changed_workbook_is_rejected_before_gpu_preflight(frozen_partition, tmp_path, monkeypatch):
    import qwen_classifier.training as training
    source, split_path, _ = frozen_partition
    monkeypatch.setattr(partitions, "EP1_DATASET_SHA256", "0" * 64)
    monkeypatch.setattr(training, "smoke_test", lambda *args: pytest.fail("GPU preflight must not run"))
    with pytest.raises(ValueError, match="frozen EP1 dataset"):
        train(Config(), source, tmp_path / "training", tmp_path, ep1_split=split_path)


def test_training_passes_frozen_split_to_fit(frozen_partition, tmp_path, monkeypatch):
    import qwen_classifier.training as training
    source, split_path, _ = frozen_partition
    observed = {}
    monkeypatch.setattr(training, "smoke_test", lambda *args: {"status": "passed", "selected_mode": "qlora"})

    def fit(config, output, cache_dir, **kwargs):
        observed.update(kwargs)
        output.mkdir()
        return {"status": "completed"}

    monkeypatch.setattr(training, "fit", fit)
    result = train(Config(), source, tmp_path / "training", tmp_path, ep1_split=split_path)
    assert result["status"] == "completed"
    assert observed["source"] == source and observed["ep1_split"] == split_path


def test_prepare_cli_accepts_frozen_partition(frozen_partition, tmp_path, capsys):
    source, split_path, _ = frozen_partition
    output = tmp_path / "cli"
    assert main(["prepare", "--input", str(source), "--ep1-split", str(split_path),
                 "--output-dir", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["train_rows"] == 9 and result["validation_rows"] == 3
    assert result["partition"]["holdout_used_for_fit"] is False
