"""Audit frozen row membership, label provenance, similarity groups and scores."""
from __future__ import annotations

import csv
import importlib.metadata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ensemble_classifier.files import align, read_json
from ensemble_classifier.soft_sources import DATASET_SHA256, SPLIT_SHA256, holdout_records, location
from ensemble_classifier.soft_voting import aligned_scores, apply_calibration
from gpt_classifier.data import LABELS, Example, canonical, digest, file_digest, group, read_jsonl
from lexical_classifier.inference import FrozenLexicalClassifier


@dataclass
class DevelopmentData:
    training: list[Example]
    groups: np.ndarray
    holdout: list[Example]
    holdout_groups: np.ndarray
    calibration: np.ndarray
    selection: np.ndarray
    old_margins: np.ndarray
    norberto: np.ndarray
    baseline: dict[str, Any]
    audit: dict[str, Any]


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    expected = {"seed", "jobs", "cache_gib", "phase_a_folds", "phase_b_folds", "train", "split",
                "group_manifest", "group_audit", "original_split", "baseline", "baseline_scores"}
    if set(config) != expected:
        raise ValueError("Unknown or missing SVM research configuration fields.")
    for field in ("seed", "jobs", "cache_gib", "phase_a_folds", "phase_b_folds"):
        if type(config[field]) is not int or config[field] < (0 if field == "seed" else 1):
            raise ValueError(f"Invalid integer setting: {field}")
    if config["jobs"] > 4 or config["phase_a_folds"] != 3 or config["phase_b_folds"] != 5:
        raise ValueError("Use at most four workers and the predeclared 3/5 grouped folds.")
    return config


def partition_indices(records, groups, partition):
    """Require complete, ordered membership and indivisible similarity components."""
    if len(partition) != len(records) or len({r["excel_row"] for r in partition}) != len(records):
        raise ValueError("Duplicate or missing holdout membership.")
    roles, normalized_roles, component_roles = {}, defaultdict(set), defaultdict(set)
    for record, component, row in zip(records, groups, partition):
        if (row["excel_row"] != record.excel_row or row["text_sha256"] != digest(record.text)
                or row["group_id"] != str(component) or row["role"] not in {"calibration", "weight_selection"}):
            raise ValueError("Holdout row/text/group provenance mismatch.")
        roles[record.excel_row] = row["role"]
        normalized_roles[group(record.text)].add(row["role"])
        component_roles[str(component)].add(row["role"])
    if any(len(value) != 1 for value in [*normalized_roles.values(), *component_roles.values()]):
        raise ValueError("A duplicate or similarity component crosses holdout roles.")
    return tuple(np.asarray([i for i, r in enumerate(records) if roles[r.excel_row] == role])
                 for role in ("calibration", "weight_selection"))


def verified_scores(root: Path, records, name: str, identity: dict[str, Any]) -> np.ndarray:
    run, path = read_json(root / name / "run.json"), root / name / "predictions.jsonl"
    if (run.get("status") != "completed" or run.get("artifact_sha256") != identity["artifact_sha256"]
            or run.get("input", {}).get("sha256") != DATASET_SHA256
            or run.get("training_performed") is not False or run.get("gold_accessed") is not False
            or run.get("predictions_sha256") != file_digest(path)):
        raise ValueError(f"Incompatible saved {name} scores.")
    rows = read_jsonl(path)
    align(records, rows)
    scores = aligned_scores(rows, "decision_scores" if name == "svm" else "probabilities")
    if [r["label"] for r in rows] != np.asarray(LABELS)[scores.argmax(axis=1)].tolist():
        raise ValueError("Saved labels differ from score argmax.")
    return scores


def load_data(config: dict[str, Any], *, reproduce: bool = False) -> DevelopmentData:
    holdout, hg, source, records, membership = holdout_records(config)
    training = [r for r in records if r.excel_row in membership]
    groups = np.asarray([membership[r.excel_row] for r in training])
    manifest = read_json(location(config["group_manifest"]))
    historical_audit = read_json(location(config["group_audit"]))
    if (file_digest(location(config["original_split"])) != SPLIT_SHA256
            or manifest.get("dataset_sha256") != DATASET_SHA256
            or manifest.get("grouping", {}).get("threshold") != .93
            or historical_audit.get("status") != "passed" or historical_audit.get("group_overlap") != 0
            or historical_audit.get("dataset_sha256") != DATASET_SHA256):
        raise ValueError("Missing original similarity-group provenance.")
    with location(config["split"]).open(newline="", encoding="utf-8") as stream:
        split = list(csv.DictReader(stream))
    normalized_components = defaultdict(set)
    for record, row in zip(records, split):
        normalized_components[group(record.text)].add(row["group_id"])
    if any(len(value) != 1 for value in normalized_components.values()):
        raise ValueError("Identical normalized texts belong to multiple components.")
    baseline_root = location(config["baseline"])
    baseline = read_json(baseline_root / "model.json")
    if (file_digest(baseline_root / "model.json") != read_json(baseline_root / "frozen_manifest.json")["artifact_sha256"]
            or baseline["source_sha256"] != DATASET_SHA256 or baseline["split_sha256"] != SPLIT_SHA256
            or baseline["weights"] != {"svm": .75, "rl": 0., "norberto": .25, "qwen": 0.}
            or baseline["partition_sha256"] != file_digest(baseline_root / "partition.jsonl")):
        raise ValueError("The historical 75/25 ensemble or its partition changed.")
    cal, selection = partition_indices(holdout, hg, read_jsonl(baseline_root / "partition.jsonl"))
    if (len(cal), len(selection), len(set(groups)), len(set(hg))) != (2027, 1972, 12956, 3281):
        raise ValueError("Unexpected original partition counts.")
    identities = baseline["model_identities"]
    svm_root = location(identities["svm"]["bundle"])
    classifier = FrozenLexicalClassifier(svm_root / "frozen_manifest.json", bundle_dir=svm_root).load()
    norberto_root = location(identities["norberto"]["bundle"])
    contract = read_json(norberto_root / "frozen_manifest.json")
    training_manifest = read_json(norberto_root / "run.json")
    if (file_digest(norberto_root / "run.json") != identities["norberto"]["training_manifest_sha256"]
            or file_digest(norberto_root / "final_model/adapter_model.safetensors") != identities["norberto"]["artifact_sha256"]
            or training_manifest["training_rows"] != 16093 or training_manifest["reserved_holdout_rows"] != 3999
            or training_manifest["dataset_sha256"] != DATASET_SHA256
            or training_manifest["split_sha256"] != SPLIT_SHA256):
        raise ValueError("NorBERTo artifact or training provenance differs from the baseline.")
    for relative, expected in contract["files"].items():
        if relative != "base_model/model.safetensors" and file_digest(norberto_root / relative) != expected:
            raise ValueError(f"NorBERTo contract changed: {relative}")
    old = verified_scores(location(config["baseline_scores"]), holdout, "svm", identities["svm"])
    raw_n = verified_scores(location(config["baseline_scores"]), holdout, "norberto", identities["norberto"])
    n = apply_calibration(raw_n, baseline["calibration"]["norberto"])
    max_difference = None
    if reproduce:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            fresh = classifier._model.decision_function([r.text for r in holdout])
        max_difference = float(np.abs(fresh - old).max())
        if not np.allclose(fresh, old, atol=1e-10, rtol=1e-10):
            raise ValueError("The original SVM does not reproduce saved holdout margins.")
    # Enumerate only inputs used in development; never open either final workbook.
    protected = [location(config[k]) for k in ("train", "split", "group_manifest", "group_audit", "original_split")]
    protected += [baseline_root / name for name in ("model.json", "partition.jsonl", "run.json", "frozen_manifest.json")]
    protected += [svm_root / name for name in ("model.joblib", "metadata.json", "config.json", "frozen_manifest.json")]
    protected += [norberto_root / name for name in contract["files"] if name != "base_model/model.safetensors"]
    protected += [norberto_root / "run.json", norberto_root / "frozen_manifest.json"]
    protected += [location(config["baseline_scores"]) / member / name
                  for member in ("svm", "norberto") for name in ("run.json", "predictions.jsonl")]
    counts = {"training": training, "calibration": [holdout[i] for i in cal],
              "weight_selection": [holdout[i] for i in selection]}
    conflicts = defaultdict(set)
    for r, g in zip(training, groups):
        conflicts[g].add(r.label)
    audit = {"status": "passed", "source": source, "labels_from": "Original train.xlsx clarity column only",
             "label_identity": digest(canonical([(r.excel_row, digest(r.text), r.label) for r in records])),
             "rows": {name: len(rows) for name, rows in counts.items()},
             "class_counts": {name: dict(Counter(r.label for r in rows)) for name, rows in counts.items()},
             "groups": {"training": len(set(groups)), "calibration": len(set(hg[cal])),
                        "weight_selection": len(set(hg[selection]))},
             "shared_groups": 0, "normalized_duplicates_crossing_roles": 0,
             "training_conflicting_similarity_groups": sum(len(v) > 1 for v in conflicts.values()),
             "historical_grouping": manifest["grouping"], "baseline_reproduction_max_abs_difference": max_difference,
             "original_svm": classifier.manifest,
             "files": {str(p.resolve()): file_digest(p) for p in protected},
             "test1_accessed": False, "test2_accessed": False,
             "limitations": ["Similarity groups were constructed label-free on the full original corpus before this study.",
                             "Existing components are reused; semantic paraphrases may remain across folds.",
                             "The 3999 rows are development data with historical exposure, not an independent test.",
                             "Final test1 has historical exposure and known source-text overlap; retain all 900 rows."]}
    return DevelopmentData(training, groups, holdout, hg, cal, selection, old, n, baseline, audit)


def runtime() -> dict[str, str]:
    return {package: importlib.metadata.version(package)
            for package in ("numpy", "scipy", "scikit-learn", "joblib", "openpyxl")}
