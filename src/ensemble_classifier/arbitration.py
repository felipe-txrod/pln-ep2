"""Import arbitrator predictions without executing or loading any model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from gpt_classifier.data import Text, digest, file_digest

from .config import EnsembleConfig
from .files import member_run, read_json, read_predictions


def local_tiebreaker(config: EnsembleConfig, records: list[Text], source_sha256: str,
                     members: dict[str, list[dict[str, Any]]], provenance: dict[str, dict[str, Any]],
                     run: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    name = config.tiebreaker
    if name == "gpt":
        raise ValueError("GPT arbitration must use the GPT import contract.")
    if name in members and run is None:
        return members[name], provenance[name]
    specification = dict(config.members[name])
    if run is not None:
        specification["run"] = str(run)
    rows, details = member_run(specification, records, source_sha256)
    if name in provenance and details["identity"] != provenance[name]["identity"]:
        raise ValueError("A member used as arbitrator must retain the same model identity.")
    return rows, details


def gpt_tiebreaker(run: Path, records: list[Text], routed: list[Text],
                   configuration: dict[str, Any]
                   ) -> tuple[dict[int, str], dict[str, Any], dict[str, str]]:
    metadata_digest = file_digest(run / "run.json")
    predictions_digest = file_digest(run / "predictions.jsonl")
    metadata = read_json(run / "run.json")
    if (metadata.get("status") != "completed" or metadata.get("dry_run")
            or metadata.get("config") != configuration):
        raise ValueError("GPT arbitration must be completed with the frozen GPT configuration.")
    raw = read_predictions(run / "predictions.jsonl")
    available = {row["excel_row"]: row for row in raw}
    expected = {record.excel_row: record for record in records}
    if set(available) - set(expected):
        raise ValueError("GPT predictions contain unknown source rows.")
    for row in raw:
        if row["text_sha256"] != digest(expected[row["excel_row"]].text):
            raise ValueError("GPT prediction source text changed.")
    if {record.excel_row for record in routed} - set(available):
        raise ValueError("GPT arbitration does not cover every routed row.")
    identity = {key: metadata.get(key) for key in ("config", "bank_sha256", "backend", "endpoint", "environment")}
    if not isinstance(identity["bank_sha256"], str) or not isinstance(identity["backend"], dict):
        raise ValueError("GPT arbitration lacks bank/runtime provenance.")
    if (file_digest(run / "run.json") != metadata_digest
            or file_digest(run / "predictions.jsonl") != predictions_digest):
        raise ValueError("GPT files changed during import.")
    decisions = {record.excel_row: available[record.excel_row]["label"] for record in routed}
    details = {"run_sha256": metadata_digest, "predictions_sha256": predictions_digest}
    return decisions, identity, details
