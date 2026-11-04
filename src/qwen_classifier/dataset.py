"""Leakage-safe grouped splits and text-only tokenization."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from gpt_classifier.data import LABELS, Example, file_digest, group, read_xlsx, write_json, write_jsonl
from .config import Config, LABEL2ID


def split_records(records: list[Example], config: Config) -> tuple[list[Example], list[Example]]:
    labels = [record.label for record in records]
    groups = [group(record.text) for record in records]
    for label in LABELS:
        if len({g for g, y in zip(groups, labels) if y == label}) < config.folds:
            raise ValueError(f"Not enough independent groups for {label} and {config.folds} folds.")
    splitter = StratifiedGroupKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
    train_indices, validation_indices = list(splitter.split(np.zeros(len(records)), labels, groups))[config.fold]
    train = [records[int(i)] for i in train_indices]
    validation = [records[int(i)] for i in validation_indices]
    if {group(r.text) for r in train} & {group(r.text) for r in validation}:
        raise RuntimeError("Normalized text groups leaked across the split.")
    for partition in (train, validation):
        if {r.label for r in partition} != set(LABELS):
            raise ValueError("Every partition must contain all predefined labels.")
    return train, validation


def prepare(source: Path, output: Path, config: Config) -> dict:
    root = Path(__file__).resolve().parents[2]
    for name in ("test1.xlsx", "test2.xlsx", "test1-rotulado-para-teste-estatistico.xlsx"):
        protected = root / "data" / name
        if protected.exists() and file_digest(source) == file_digest(protected):
            raise ValueError("Official test workbooks cannot be used for development.")
    records, source_info = read_xlsx(source, labeled=True)
    train, validation = split_records(records, config)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source": source_info, "config": config.to_dict(), "labels": list(LABELS),
        "split_method": "StratifiedGroupKFold over HTML-unescaped, whitespace-normalized lowercase text",
        "train_rows": [r.excel_row for r in train], "validation_rows": [r.excel_row for r in validation],
        "train_counts": dict(Counter(r.label for r in train)),
        "validation_counts": dict(Counter(r.label for r in validation)),
        "duplicates_and_conflicting_labels": "Retained in the same partition; no evaluation rows removed.",
    }
    write_json(output / "split.json", manifest)
    write_jsonl(output / "train.jsonl", [asdict(r) for r in train])
    write_jsonl(output / "validation.jsonl", [asdict(r) for r in validation])
    return manifest


class TokenizedTexts:
    """Keep Excel metadata outside tensors; labels never enter the input text."""

    def __init__(self, records, tokenizer, max_length: int, *, labeled: bool = True):
        self.items = []
        self.truncated = 0
        for record in records:
            ids = tokenizer.encode(record.text, add_special_tokens=False)
            self.truncated += int(len(ids) + 1 > max_length)
            # A distinct EOS guarantees a valid final token even for empty responses.
            ids = ids[:max_length - 1] + [tokenizer.eos_token_id]
            item = {"input_ids": ids, "attention_mask": [1] * len(ids)}
            if labeled:
                item["labels"] = LABEL2ID[record.label]
            self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]
