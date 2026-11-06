"""Verified development-only example bank and group-disjoint context-study cohorts."""
from __future__ import annotations

from pathlib import Path

from cascade_classifier.experiment import implementation as cascade_implementation
from cascade_classifier.experiment import load_cascade, prepared_inputs, verify_files
from ensemble_classifier.files import read_json
from ensemble_classifier.soft_sources import SPLIT_SHA256, holdout_records, location
from ensemble_classifier.stacking_pipeline import load_stacking_artifact
from gpt_classifier.data import LABELS, canonical, digest, file_digest, read_jsonl, write_json, write_jsonl
from cascade_classifier.routing import select_targets
from .config import RagConfig


def implementation():
    root = Path(__file__).resolve().parents[1]
    paths = [*Path(__file__).parent.glob("*.py"), root / "gpt_classifier/selection.py", root / "gpt_classifier/embeddings.py",
             root / "ensemble_classifier/stacking_pipeline.py", root / "gpt_classifier/delivery.py"]
    return {**cascade_implementation(), **{str(path.relative_to(root)): file_digest(path) for path in sorted(paths)}}


def training_bank(config):
    cascade, transport = load_cascade(location(config.cascade_model))
    _, soft = load_stacking_artifact(location(transport.stacking_model))
    holdout, groups, source, records, development = holdout_records(soft["score_configuration"])
    bank = [record for record in records if record.excel_row in development]
    if len(bank) != 16093 or len(holdout) != 3999 or set(r.label for r in bank) != set(LABELS):
        raise ValueError("RAG requires the original 16,093-row development bank.")
    identity = {"source_sha256": source["sha256"], "bank_rows": len(bank), "reserved_holdout_rows": len(holdout),
                "bank_sha256": digest(canonical([(r.excel_row, digest(r.text), r.label, development[r.excel_row]) for r in bank])),
                "cascade_sha256": file_digest(location(config.cascade_model)),
                "stacking_sha256": file_digest(location(transport.stacking_model)),
                "split_sha256": SPLIT_SHA256, "stacking_partition_sha256": cascade["evidence"]["partition_sha256"]}
    if transport.stacking_model != config.transport.stacking_model:
        raise ValueError("Context study must retain the validated stacking model.")
    return bank, development, identity


def split_cohorts(rows, groups, *, pilot_size, budget):
    selected = select_targets(rows, budget)
    ordered = sorted(selected, key=lambda row: (digest("rag-pilot-v1:" + row["text_sha256"]), row["excel_row"]))
    pilot, used = [], set()
    for row in ordered:
        component = groups[row["excel_row"]]
        if component not in used:
            pilot.append(row["excel_row"])
            used.add(component)
        if len(pilot) == pilot_size:
            break
    if len(pilot) != pilot_size:
        raise ValueError("Insufficient independent difficult groups for the requested pilot.")
    confirmation = [row["excel_row"] for row in rows if groups[row["excel_row"]] not in used]
    if len(confirmation) < 100 or set(pilot) & set(confirmation):
        raise ValueError("Insufficient group-disjoint confirmation data.")
    return {"pilot_ids": sorted(pilot), "confirmation_ids": confirmation,
            "excluded_pilot_group_ids": sorted(used), "shared_groups": 0,
            "pilot_selection": "Label-free SHA-256 order among the original difficult targets; one target per similarity group"}


def prepare(config: RagConfig, output: Path):
    if output.exists():
        raise FileExistsError("Use a new RAG study directory; previous experiments are immutable.")
    _, _, bank = training_bank(config)
    old, targets, rows = prepared_inputs(location(config.oof_directory))
    cascade, _ = load_cascade(location(config.cascade_model))
    if (file_digest(location(config.oof_directory) / "preparation.json") != cascade["preparation_sha256"]
            or old["routing_rule"] != cascade["routing_rule"]
            or old["evidence"]["stacking_artifact_sha256"] != bank["stacking_sha256"]):
        raise ValueError("OOF inputs differ from the audited routing/stacking model.")
    partition = read_jsonl(location(config.transport.stacking_model).with_name("partition.jsonl"))
    groups = {row["excel_row"]: row["group_id"] for row in partition if row["stacking_role"] == "meta_training"}
    if set(groups) != {target.excel_row for target in targets}:
        raise ValueError("OOF inputs do not cover the original isolated meta-training rows.")
    split = split_cohorts(rows, groups, pilot_size=config.pilot_size, budget=config.transport.initial_budget)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("targets.jsonl", "oof-routing.jsonl", "validation-gold.jsonl"):
        (output / name).write_bytes((location(config.oof_directory) / name).read_bytes())
    write_json(output / "cohorts.json", split)
    report = {"status": "prepared", "config": config.to_dict(), "bank": bank, "routing_rule": cascade["routing_rule"],
              "oof_source_sha256": file_digest(location(config.oof_directory) / "preparation.json"),
              "implementation": implementation(), "cohorts": split,
              "files": {name: file_digest(output / name) for name in
                        ("targets.jsonl", "oof-routing.jsonl", "validation-gold.jsonl", "cohorts.json")},
              "classification_executed": False, "test1_accessed": False, "test2_accessed": False,
              "limitations": "Nested OOF excludes outer targets from meta fitting/C selection; historical holdout and Qwen checkpoint exposure remain."}
    write_json(output / "study.json", report)
    return report


def load_study(directory):
    study = read_json(directory / "study.json")
    verify_files(directory, study["files"])
    if study["implementation"] != implementation():
        raise ValueError("Study implementation changed; preserve it and prepare a reviewed new study.")
    config = RagConfig.from_dict(study["config"])
    bank, components, identity = training_bank(config)
    if identity != study["bank"]:
        raise ValueError("Training bank or base models changed after preparation.")
    return study, config, bank, components
