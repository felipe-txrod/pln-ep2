"""Validate persisted member outputs without importing classifier runtimes."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from gpt_classifier.data import LABELS, Text, digest, file_digest


def decode_json(content: str, path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Nonfinite JSON constant in {path}: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field in {path}: {key}")
            result[key] = value
        return result

    return json.loads(content, parse_constant=reject_constant,
                      object_pairs_hook=unique_object)


def read_json(path: Path) -> dict[str, Any]:
    value = decode_json(path.read_text(encoding="utf-8"), path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def read_predictions(path: Path) -> list[dict[str, Any]]:
    result = []
    # Split only on LF; response text can contain other Unicode separators.
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                raise ValueError(f"Empty prediction line in {path}.")
            row = decode_json(line, path)
            if (not isinstance(row, dict) or type(row.get("excel_row")) is not int
                    or row["excel_row"] < 2 or row.get("label") not in LABELS
                    or not isinstance(row.get("text_sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", row["text_sha256"])):
                raise ValueError(f"Invalid prediction record in {path}.")
            result.append(row)
    if len({row["excel_row"] for row in result}) != len(result):
        raise ValueError(f"Duplicate prediction rows in {path}.")
    return result


def align(records: list[Text], rows: list[dict[str, Any]]) -> None:
    if len(records) != len(rows):
        raise ValueError("Member coverage differs from the source rows.")
    for record, row in zip(records, rows):
        if row["excel_row"] != record.excel_row or row["text_sha256"] != digest(record.text):
            raise ValueError(f"Member row/hash alignment failed at Excel row {record.excel_row}.")


def validate_scores(row: dict[str, Any], labels: list[str]) -> None:
    for field in ("probabilities", "logits", "decision_scores"):
        values = row.get(field)
        if values is None:
            continue
        if isinstance(values, dict):
            if set(values) != set(LABELS):
                raise ValueError(f"Unknown classes in {field}.")
            scores = [values[label] for label in LABELS]
        elif isinstance(values, list) and len(values) == len(LABELS) and set(labels) == set(LABELS):
            scores = [values[labels.index(label)] for label in LABELS]
        else:
            raise ValueError(f"An explicit class order is required for {field}.")
        if any(type(value) not in (float, int) or not math.isfinite(value) for value in scores):
            raise ValueError(f"Nonfinite or invalid {field}.")
        if field == "probabilities" and (
                any(value < 0 or value > 1 for value in scores)
                or not math.isclose(sum(scores), 1, rel_tol=0, abs_tol=1e-6)):
            raise ValueError("Invalid probability distribution.")


def member_run(specification: dict[str, str], records: list[Text], source_sha256: str
               ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(specification["run"])
    metadata_path, predictions_path = root / "run.json", root / "predictions.jsonl"
    metadata_digest, predictions_digest = file_digest(metadata_path), file_digest(predictions_path)
    metadata = read_json(metadata_path)
    source = metadata.get("input")
    if (metadata.get("status") != "completed" or not isinstance(source, dict)
            or source.get("sha256") != source_sha256):
        raise ValueError(f"Member run is incomplete or uses another source: {root}")
    labels = metadata.get("labels")
    identity_field = specification.get("identity_field")
    if identity_field is None:
        identity_field = next((key for key in ("artifact_sha256", "adapter_sha256", "artifact_manifest_sha256")
                               if isinstance(metadata.get(key), str)), None)
    artifact = metadata.get(identity_field) if identity_field else None
    if not isinstance(artifact, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact):
        raise ValueError(f"Member run lacks a usable immutable model identity: {root}")
    selected_manifest_digest = None
    if "manifest" in specification:
        selected_manifest = Path(specification["manifest"])
        selected_manifest_digest = file_digest(selected_manifest)
        contract = read_json(selected_manifest)
        if identity_field == "artifact_manifest_sha256":
            expected = selected_manifest_digest
        elif identity_field == "artifact_sha256":
            expected = contract.get("artifact_sha256")
        else:
            raise ValueError("A selected manifest requires an artifact or artifact-manifest identity.")
        if artifact != expected:
            raise ValueError(f"Member predictions differ from the selected model manifest: {root}")
    rows = read_predictions(predictions_path)
    if labels is None and any(isinstance(row.get(field), list) for row in rows
                              for field in ("probabilities", "logits")):
        # Existing Qwen runs pin their label order through adapter_identity.
        if identity_field != "adapter_sha256" or "adapter" not in specification:
            raise ValueError(f"List-valued scores require labels or a pinned Qwen adapter: {root}")
        from gpt_classifier.data import canonical
        adapter = Path(specification["adapter"])
        names = ("adapter_config.json", "adapter_model.safetensors", "experiment.json",
                 "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        if not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Missing Qwen adapter weights: {adapter}")
        observed = digest(canonical({name: file_digest(adapter / name) for name in names
                                     if (adapter / name).exists()}))
        if observed != artifact:
            raise ValueError("The Qwen adapter differs from the member inference run.")
        labels = read_json(adapter / "experiment.json").get("labels")
    if labels is not None and (not isinstance(labels, list) or len(labels) != len(LABELS)
                               or any(not isinstance(label, str) for label in labels)
                               or set(labels) != set(LABELS)):
        raise ValueError(f"Member class order is incompatible: {root}")
    align(records, rows)
    for row in rows:
        validate_scores(row, labels or list(LABELS))
    if (file_digest(metadata_path) != metadata_digest or file_digest(predictions_path) != predictions_digest
            or (selected_manifest_digest is not None
                and file_digest(Path(specification["manifest"])) != selected_manifest_digest)):
        raise ValueError(f"Member files changed during import: {root}")
    provenance = {"identity": {"field": identity_field, "sha256": artifact,
                               "config": metadata.get("config"),
                               "artifact_manifest_sha256": metadata.get("artifact_manifest_sha256"),
                               "implementation_sha256": metadata.get("implementation_sha256")},
                  "run_sha256": metadata_digest,
                  "predictions_sha256": predictions_digest}
    if selected_manifest_digest is not None:
        provenance["identity"]["selected_manifest_sha256"] = selected_manifest_digest
    return rows, provenance
