"""Audit EP1/EP2 workbook identities and exact/normalized text overlap on CPU."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import html
from itertools import combinations
import json
from pathlib import Path
import re
from typing import Any

from gpt_classifier.data import Example, Text, read_xlsx, write_json

ROOT = Path(__file__).resolve().parents[2]


def normalize_text(text: str) -> str:
    """Use the same HTML, whitespace, and case normalization as EP1 grouping."""
    return re.sub(r"\s+", " ", html.unescape(text)).strip().lower()


@dataclass
class Dataset:
    name: str
    records: list[Text] | list[Example]
    info: dict[str, Any]
    labeled: bool


def text_index(dataset: Dataset, normalized: bool) -> dict[str, list[Text | Example]]:
    index: dict[str, list[Text | Example]] = defaultdict(list)
    for row in dataset.records:
        index[normalize_text(row.text) if normalized else row.text].append(row)
    return dict(index)


def describe(dataset: Dataset) -> dict[str, Any]:
    report = {**dataset.info, "labeled": dataset.labeled}
    for normalized in (False, True):
        groups = text_index(dataset, normalized)
        sizes = [len(rows) for rows in groups.values()]
        summary = {"unique_texts": len(groups),
                   "duplicate_extra_rows": len(dataset.records) - len(groups),
                   "duplicate_groups": sum(size > 1 for size in sizes),
                   "rows_in_duplicate_groups": sum(size for size in sizes if size > 1),
                   "empty_text_rows": len(groups.get("", []))}
        if dataset.labeled:
            conflicting = [rows for rows in groups.values() if len({r.label for r in rows}) > 1]
            summary.update(conflicting_label_groups=len(conflicting),
                           rows_in_conflicting_label_groups=sum(map(len, conflicting)))
        report["normalized" if normalized else "exact"] = summary
    if dataset.labeled:
        report["label_counts"] = dict(Counter(r.label for r in dataset.records))
    return report


def compare(left: Dataset, right: Dataset) -> dict[str, Any]:
    report = {"left": left.name, "right": right.name,
              "same_file_sha256": left.info["sha256"] == right.info["sha256"],
              "same_texts_in_order": [r.text for r in left.records] == [r.text for r in right.records],
              "same_text_multiset": Counter(r.text for r in left.records) == Counter(r.text for r in right.records)}
    if left.labeled and right.labeled:
        report["same_texts_and_labels_in_order"] = (
            [(r.text, r.label) for r in left.records] == [(r.text, r.label) for r in right.records])
    for normalized in (False, True):
        a, b = text_index(left, normalized), text_index(right, normalized)
        shared = a.keys() & b.keys()
        summary = {"shared_unique_texts": len(shared),
                   "left_matching_rows": sum(len(a[key]) for key in shared),
                   "right_matching_rows": sum(len(b[key]) for key in shared)}
        if left.labeled and right.labeled:
            agreement = Counter()
            for key in shared:
                labels = {r.label for r in a[key]}
                for row in b[key]:
                    category = ("only_same_label" if labels == {row.label} else
                                "same_and_other_labels" if row.label in labels else "only_other_labels")
                    agreement[category] += 1
            summary["right_rows_vs_left_labels"] = {
                category: agreement[category] for category in
                ("only_same_label", "same_and_other_labels", "only_other_labels")}
        report["normalized" if normalized else "exact"] = summary
    return report


def audit(ep1_root: Path, ep2_root: Path) -> dict[str, Any]:
    sources = [("ep1_train", ep1_root / "data/train.xlsx", True),
               ("ep1_test1", ep1_root / "data/test1.xlsx", False),
               ("ep2_train", ep2_root / "data/train.xlsx", True),
               ("ep2_test1", ep2_root / "data/test1.xlsx", False),
               ("ep2_test1_gold", ep2_root / "data/test1-rotulado-para-teste-estatistico.xlsx", True),
               ("ep2_test2", ep2_root / "data/test2.xlsx", False)]
    datasets = []
    for name, path, labeled in sources:
        records, info = read_xlsx(path, labeled=labeled)
        datasets.append(Dataset(name, records, info, labeled))
    return {"schema_version": 1, "scope": "Provided workbooks; no model inference or training.",
            "normalization": "HTML unescape, collapse whitespace, strip, lowercase; retain punctuation and accents.",
            "limitations": "Text equality does not establish equality of original requests, authors, or annotation context. No fuzzy or semantic matching.",
            "datasets": {d.name: describe(d) for d in datasets},
            "pairs": [compare(a, b) for a, b in combinations(datasets, 2)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep1-root", type=Path, default=ROOT.parent / "EP1")
    parser.add_argument("--ep2-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("The output must be new; existing audit results are preserved.")
    report = audit(args.ep1_root, args.ep2_root)
    write_json(args.output, report, replace=False)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
