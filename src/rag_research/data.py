"""Read verified historical validation artifacts without changing their fingerprints."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cascade_classifier.experiment import verify_files
from gpt_classifier.data import Example, Text, digest, file_digest, read_jsonl
from ensemble_classifier.files import read_json
from rag_cascade.data import load_study


@dataclass
class ResearchData:
    bank: list[Example]
    bank_groups: dict[int, str]
    targets: list[Text]
    routing: dict[int, dict]
    gold: dict[int, str]
    groups: dict[int, str]
    folds: dict[int, int]
    study: dict
    old_config: object
    logged: dict[int, dict]
    identities: dict


def load_data(config) -> ResearchData:
    directory = Path(config.study)
    study, old_config, bank, bank_groups = load_study(directory)
    from ensemble_classifier.soft_sources import location
    partition = read_jsonl(location(old_config.transport.stacking_model).with_name("partition.jsonl"))
    groups = {r["excel_row"]: r["group_id"] for r in partition if r["stacking_role"] == "meta_training"}
    targets = [Text(**r) for r in read_jsonl(directory / "targets.jsonl")]
    routing = {r["excel_row"]: r for r in read_jsonl(directory / "oof-routing.jsonl")}
    if set(groups) != set(routing) or set(groups) != {r.excel_row for r in targets}:
        raise ValueError("Historical validation IDs are not aligned.")
    folds = {i: row["outer_fold"] for i, row in routing.items()}
    for group in set(groups.values()):
        if len({folds[i] for i in groups if groups[i] == group}) != 1:
            raise ValueError("Historical outer folds split a similarity group.")
    logged, identity = {}, {"study_sha256": file_digest(directory / "study.json")}
    for stage in ("pilot", "confirmation"):
        root = directory / stage
        plan, completed = read_json(root / "plan.json"), read_json(root / "run.json")
        verify_files(root, plan["files"])
        verify_files(root, completed["files"])
        if completed["status"] != "completed" or completed["plan_sha256"] != file_digest(root / "plan.json"):
            raise ValueError("Historical context stage is not a completed immutable run.")
        for recipe in plan["recipes"]:
            arm = root / recipe["name"]
            # Inspect the original traces; do not synthesize replies for new contexts.
            if recipe["name"] not in {"text", "local", "hybrid-3-rag", "hybrid-3-local"}:
                continue
            decisions = read_json(arm / "gpt/decisions.json")
            for trace in read_jsonl(arm / "context-trace.jsonl"):
                identifier = trace["excel_row"]
                value = decisions[str(identifier)]
                if value.get("label") not in {"c1", "c234", "c5"} or value.get("status") != "completed":
                    raise ValueError("A historical GPT answer is missing or invalid.")
                if identifier in logged and logged[identifier]["stage"] != stage:
                    raise ValueError("Pilot and confirmation queries overlap.")
                logged.setdefault(identifier, {"stage": stage, "arms": {}})["arms"][recipe["name"]] = {
                    "label": value["label"], "retrieved": trace["retrieved"], "receipt": value}
        identity[stage + "_plan_sha256"] = file_digest(root / "plan.json")
        identity[stage + "_run_sha256"] = file_digest(root / "run.json")
    gold = {r["excel_row"]: r["label"] for r in read_jsonl(directory / "validation-gold.jsonl")}
    if set(gold) != set(groups) or any(routing[r.excel_row]["text_sha256"] != digest(r.text) for r in targets):
        raise ValueError("Validation text/label/ID coverage differs from the audited source.")
    if set(bank_groups.values()) & set(groups.values()):
        raise ValueError("Retrieval bank and validation similarity groups overlap.")
    if any(set(value["arms"]) != {"text", "local", "hybrid-3-rag", "hybrid-3-local"} for value in logged.values()):
        raise ValueError("Historical paired context coverage is incomplete.")
    return ResearchData(bank, bank_groups, targets, routing, gold, groups, folds, study, old_config, logged, identity)
