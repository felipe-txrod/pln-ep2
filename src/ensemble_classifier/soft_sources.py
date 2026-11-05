"""Reuse verified EP1 model packages and their recorded probabilities."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from gpt_classifier.data import LABELS, Text, canonical, digest, file_digest, group, read_xlsx, write_json, write_jsonl
from .files import align, read_json, read_predictions, validate_scores
from .soft_voting import MODELS, aligned_scores, probabilities

ROOT = Path(__file__).resolve().parents[2]
DATASET_SHA256 = "0e9219233664ac675fc47982bbc822acd3bb15d67fb7bfc26b67bec047e46818"
SPLIT_SHA256 = "0f78e7a809c7b7584db05b33cf03a73ff586d5de02ba824c16562d5070d41494"


def load_soft_config(path):
    config = read_json(path)
    if set(config) != {"seed", "train", "split", "recorded_norberto"} or type(config["seed"]) is not int:
        raise ValueError("Soft configuration must contain seed, train, split and recorded_norberto only.")
    return config


def location(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def holdout_records(config):
    train, split_path = location(config["train"]), location(config["split"])
    if file_digest(train) != DATASET_SHA256 or file_digest(split_path) != SPLIT_SHA256:
        raise ValueError("The training workbook or frozen EP1 partition changed.")
    records, source = read_xlsx(train, labeled=True)
    with split_path.open(encoding="utf-8", newline="") as stream:
        split = list(csv.DictReader(stream))
    if (len(records) != 20092 or len(split) != len(records)
            or [int(r["excel_row"]) for r in split] != [r.excel_row for r in records]):
        raise ValueError("EP1 split/source row alignment failed.")
    development, holdout = {}, {}
    for record, row in zip(records, split):
        if row["role"] not in {"development", "holdout"} or not row["group_id"]:
            raise ValueError("Invalid frozen partition role/group.")
        (development if row["role"] == "development" else holdout)[record.excel_row] = row["group_id"]
    if len(development) != 16093 or len(holdout) != 3999 or set(development.values()) & set(holdout.values()):
        raise ValueError("The original 16,093/3,999 partition is not isolated.")
    dev_texts = {group(r.text) for r in records if r.excel_row in development}
    if dev_texts & {group(r.text) for r in records if r.excel_row in holdout}:
        raise ValueError("Normalized texts overlap between development and holdout.")
    selected = [r for r in records if r.excel_row in holdout]
    return selected, np.asarray([holdout[r.excel_row] for r in selected]), source, records, development


def model_identities():
    result = {}
    for name, directory in (("svm", "models/svm"), ("rl", "models/rl-development")):
        root = ROOT / directory
        contract = read_json(root / "frozen_manifest.json")
        if contract["model_type"] != name or contract["labels"] != list(LABELS):
            raise ValueError("Unexpected member class/type contract.")
        if file_digest(root / "model.joblib") != contract["artifact_sha256"]:
            raise ValueError("Frozen lexical weights changed.")
        for path, expected in contract["files"].items():
            if file_digest(root / path) != expected:
                raise ValueError("Frozen lexical metadata changed.")
        meta = read_json(root / "metadata.json")
        partition = meta["fixed_validation"] if name == "svm" else meta
        reserved = partition.get("reserved_holdout_rows", partition.get("holdout_rows"))
        training = partition.get("training_rows", partition.get("development_rows"))
        if (meta["dataset_sha256"] != DATASET_SHA256 or partition["split_sha256"] != SPLIT_SHA256
                or training != 16093 or reserved != 3999 or partition.get("holdout_used_for_fit", False)):
            raise ValueError("A full-corpus or incompatible lexical model cannot use this holdout.")
        result[name] = {"artifact_sha256": contract["artifact_sha256"],
                        "manifest_sha256": file_digest(root / "frozen_manifest.json"), "bundle": directory}
    from norberto_classifier.artifacts import ADAPTER_SHA256
    contract = read_json(ROOT / "src/norberto_classifier/frozen_manifest.json")
    root = ROOT / "models/norberto"
    # Recorded scores need no base weights, but every available inference contract is verified.
    for path, expected in contract["files"].items():
        if path == "base_model/model.safetensors" and not (root / path).exists():
            continue
        if file_digest(root / path) != expected:
            raise ValueError("The original NorBERTo package changed.")
    meta = read_json(root / "run.json")
    if (meta["training_rows"] != 16093 or meta["reserved_holdout_rows"] != 3999
            or meta["split_sha256"] != SPLIT_SHA256 or meta["dataset_sha256"] != DATASET_SHA256
            or contract["files"]["final_model/adapter_model.safetensors"] != ADAPTER_SHA256):
        raise ValueError("NorBERTo did not reserve the original holdout.")
    result["norberto"] = {"artifact_sha256": ADAPTER_SHA256,
                           "manifest_sha256": file_digest(ROOT / "src/norberto_classifier/frozen_manifest.json"),
                           "training_manifest_sha256": file_digest(root / "run.json"), "bundle": "models/norberto"}
    return result


def verify_lr_training(records, development):
    rows = read_predictions_training(ROOT / "models/rl-development/training-rows.jsonl")
    expected = [{"excel_row": r.excel_row, "text_sha256": digest(r.text), "label": r.label}
                for r in records if r.excel_row in development]
    meta = read_json(ROOT / "models/rl-development/metadata.json")
    if rows != expected or digest(canonical(rows)) != meta["training_rows_identity_sha256"]:
        raise ValueError("LR training membership differs from frozen development.")


def read_predictions_training(path):
    # Same strict JSON parser, but these identities have no model prediction label semantics.
    from .files import decode_json
    with path.open(encoding="utf-8") as stream:
        return [decode_json(line, path) for line in stream]


def recorded_norberto(config, records, source, identities, phase):
    specification = config["recorded_norberto"].get(phase)
    if specification is None:
        return None, None
    path, metadata_path = location(specification["csv"]), location(specification["metadata"])
    if (file_digest(path) != specification["csv_sha256"]
            or file_digest(metadata_path) != specification["metadata_sha256"]):
        raise ValueError("Recorded NorBERTo predictions or provenance changed.")
    meta = read_json(metadata_path)
    if (meta["status"] != "complete"
            or meta["source_final_metadata_sha256"] != identities["norberto"]["training_manifest_sha256"]
            or meta["adapter_sha256"]["adapter_model.safetensors"] != identities["norberto"]["artifact_sha256"]
            or meta.get("test_sha256", meta.get("dataset_sha256")) != source["sha256"]
            or (phase == "holdout" and meta.get("split_sha256") != SPLIT_SHA256)):
        raise ValueError("Recorded NorBERTo scores belong to another model/dataset.")
    if "predictions_sha256" in meta and meta["predictions_sha256"] != specification["csv_sha256"]:
        raise ValueError("NorBERTo prediction checksum differs from its original report.")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["excel_row"]) for row in rows] != [r.excel_row for r in records]:
        raise ValueError("Recorded NorBERTo row alignment failed.")
    p = probabilities([[float(row[f"prob_norberto_lora_{label}"]) for label in LABELS] for row in rows])
    expected = np.asarray(LABELS)[p.argmax(axis=1)]
    if any(row.get("predicted_label", row.get("pred_norberto_lora")) != label for row, label in zip(rows, expected)):
        raise ValueError("NorBERTo labels and recorded probabilities disagree.")
    return p, {"csv_sha256": file_digest(path), "metadata_sha256": file_digest(metadata_path)}


def collect_scores(config, records, source, identities, directory, *, phase, device="cpu"):
    """Label-free inference with per-member cache, compatible prediction JSONL records."""
    if any(type(r) is not Text for r in records):
        raise TypeError("Scoring accepts Text targets only, without gold labels.")
    matrices, provenance = {}, {}
    for name in MODELS:
        root = directory / name
        field = "decision_scores" if name == "svm" else "probabilities"
        key = {"member": name, "model_identity": identities[name], "source_sha256": source["sha256"],
               "targets_sha256": digest(canonical([(r.excel_row, digest(r.text)) for r in records]))}
        if (root / "run.json").exists():
            manifest = read_json(root / "run.json")
            if (manifest.get("status") != "completed" or manifest.get("score_identity") != key
                    or manifest.get("predictions_sha256") != file_digest(root / "predictions.jsonl")):
                raise ValueError("Cached member scores have incompatible provenance.")
            rows = read_predictions(root / "predictions.jsonl")
            align(records, rows)
            for row in rows:
                validate_scores(row, list(LABELS))
            matrices[name] = aligned_scores(rows, field)
            provenance[name] = manifest
            continue
        print(f"Collecting frozen {name} scores: {len(records)} rows", flush=True)
        extra = None
        native_predict_proba = None
        if name == "norberto":
            p, extra = recorded_norberto(config, records, source, identities, phase)
            if p is None:
                from norberto_classifier import NorbertoClassifier
                predicted = NorbertoClassifier(device=device).predict([r.text for r in records])
                p = aligned_scores([{"probabilities": v.probabilities} for v in predicted], "probabilities")
            labels = np.asarray(LABELS)[p.argmax(axis=1)]
        else:
            from lexical_classifier import FrozenLexicalClassifier
            bundle = location(identities[name]["bundle"])
            classifier = FrozenLexicalClassifier(bundle / "frozen_manifest.json", bundle_dir=bundle)
            predicted = classifier.predict([r.text for r in records])
            native_predict_proba = hasattr(classifier._model, "predict_proba")
            if name == "svm" and native_predict_proba:
                raise ValueError("The selected SVM contract unexpectedly changed its probability interface.")
            p = aligned_scores([{field: getattr(v, field)} for v in predicted], field)
            labels = [v.label for v in predicted]
        rows = [{"excel_row": record.excel_row, "text_sha256": digest(record.text), "label": str(label),
                 field: dict(zip(LABELS, scores.tolist()))}
                for record, label, scores in zip(records, labels, p)]
        root.mkdir(parents=True, exist_ok=True)
        write_jsonl(root / "predictions.jsonl", rows)
        manifest = {"status": "completed", "labels": list(LABELS), "input": source,
                    "artifact_sha256": identities[name]["artifact_sha256"], "score_identity": key,
                    "recorded_score_provenance": extra, "training_performed": False,
                    "native_predict_proba": native_predict_proba,
                    "gold_accessed": False, "predictions_sha256": file_digest(root / "predictions.jsonl")}
        write_json(root / "run.json", manifest)
        matrices[name], provenance[name] = p, manifest
    if model_identities() != identities:
        raise ValueError("Base models changed during scoring.")
    return matrices, provenance
