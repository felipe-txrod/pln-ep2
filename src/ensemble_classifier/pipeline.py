"""Offline ensembles with independently configured members and arbitrator."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from gpt_classifier.data import LABELS, canonical, digest, file_digest, read_xlsx, write_json, write_jsonl

from .config import EnsembleConfig
from .arbitration import gpt_tiebreaker, local_tiebreaker
from .files import align, member_run, read_json, read_predictions
from .strategies import local_decision


def combine(config: EnsembleConfig, source: Path, output: Path, *, gpt_run: Path | None = None,
            tiebreaker_run: Path | None = None,
            phase: str = "inference", test2_run: Path | None = None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Use a new ensemble output directory: {output}")
    if phase not in {"inference", "statistical"}:
        raise ValueError("phase must be inference or statistical.")
    if gpt_run is not None and tiebreaker_run is not None:
        raise ValueError("Use only one of --tiebreaker-run and --gpt-run.")
    if gpt_run is not None and config.tiebreaker != "gpt":
        raise ValueError("--gpt-run is valid only when the selected tiebreaker is gpt.")
    selected_run = tiebreaker_run if tiebreaker_run is not None else gpt_run
    records, source_info = read_xlsx(source)
    gpt_config = None
    if config.tiebreaker == "gpt":
        from gpt_classifier.prompt import Config as GPTConfig
        if config.gpt_config is None:
            raise ValueError("GPT arbitration requires gpt_config.")
        gpt_config = GPTConfig.load(config.gpt_config).to_dict()
    members, provenance = {}, {}
    for name in config.models:
        members[name], provenance[name] = member_run(config.members[name], records, source_info["sha256"])
    arbitrator_rows, arbitrator_provenance, arbitrator_identity = None, None, None
    if config.tiebreaker != "gpt":
        arbitrator_rows, arbitrator_provenance = local_tiebreaker(
            config, records, source_info["sha256"], members, provenance, selected_run)
        arbitrator_identity = arbitrator_provenance["identity"]
    implementation = {path.name: file_digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    frozen = {"models": list(config.models), "strategy": config.strategy,
              "members": {name: provenance[name]["identity"] for name in config.models},
              "tiebreaker": {"model": config.tiebreaker, "identity": arbitrator_identity,
                             "gpt_config": gpt_config},
              "implementation_sha256": implementation}
    identity = digest(canonical(frozen))
    root = Path(__file__).resolve().parents[2]
    official = {name: file_digest(root / "data" / name)
                for name in ("test1.xlsx", "test1-rotulado-para-teste-estatistico.xlsx", "test2.xlsx")
                if (root / "data" / name).is_file()}
    is_test1 = source_info["sha256"] in {official.get("test1.xlsx"), official.get("test1-rotulado-para-teste-estatistico.xlsx")}
    earlier = None
    if is_test1 and phase != "statistical":
        raise ValueError("Official test1 requires phase=statistical and a completed ensemble test2 run.")
    if phase == "statistical":
        if not is_test1 or test2_run is None:
            raise ValueError("Statistical inference requires official test1 and a test2 run.")
        earlier = read_json(test2_run / "run.json")
        previous_input = earlier.get("input")
        if (earlier.get("status") != "completed" or earlier.get("ensemble_identity") != identity
                or not isinstance(previous_input, dict)
                or previous_input.get("sha256") != official.get("test2.xlsx")):
            raise ValueError("Test2 must be completed with exactly the same ensemble and model artifacts.")
        previous_predictions = read_predictions(test2_run / "predictions.jsonl")
        previous_records, _ = read_xlsx(root / "data/test2.xlsx")
        align(previous_records, previous_predictions)
        if file_digest(test2_run / "predictions.jsonl") != earlier.get("predictions_sha256"):
            raise ValueError("The completed ensemble test2 predictions changed.")
    local = [local_decision([members[name][index]["label"] for name in config.models], config.strategy)
             for index in range(len(records))]
    routed = [record for record, label in zip(records, local) if label is None]
    arbitration = {}
    if routed and arbitrator_rows is not None:
        available = {row["excel_row"]: row["label"] for row in arbitrator_rows}
        arbitration = {record.excel_row: available[record.excel_row] for record in routed}
    elif routed and selected_run is not None:
        if gpt_config is None:
            raise ValueError("Missing configuration for GPT arbitration.")
        arbitration, arbitrator_identity, arbitrator_provenance = gpt_tiebreaker(
            selected_run, records, routed, gpt_config)
    if (earlier is not None and arbitrator_identity is not None
            and earlier.get("tiebreaker_identity") not in (None, arbitrator_identity)):
        raise ValueError("Arbitrator model, bank or runtime differs from the completed test2 run.")
    report = {"status": "pending_tiebreaks" if routed and not arbitration else "completed",
              "input": source_info, "rows": len(records), "labels": list(LABELS),
              "phase": phase, "ensemble_identity": identity, "configuration": frozen,
              "members": provenance, "tiebreaker_model": config.tiebreaker,
              "tiebreaker_identity": arbitrator_identity, "tiebreaker_run": arbitrator_provenance,
              "routed_rows": len(routed), "routed_fraction": len(routed) / len(records),
              "training_performed": False, "remote_requests_performed": False}
    if file_digest(source) != source_info["sha256"]:
        raise ValueError("The source workbook changed during combination.")
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "routing.jsonl", [{"excel_row": record.excel_row,
                "text_sha256": digest(record.text)} for record in routed])
    if report["status"] == "pending_tiebreaks":
        # Targets contain no labels or local model decisions.
        write_jsonl(output / "pending-targets.jsonl", [
            {"excel_row": record.excel_row, "text": record.text} for record in routed])
    else:
        predictions = []
        for index, (record, label) in enumerate(zip(records, local)):
            predictions.append({"excel_row": record.excel_row, "text_sha256": digest(record.text),
                                "label": arbitration[record.excel_row] if label is None else label,
                                "decision_source": config.tiebreaker if label is None else config.strategy,
                                "arbitrated": label is None,
                                "member_labels": {name: members[name][index]["label"] for name in config.models}})
        write_jsonl(output / "predictions.jsonl", predictions)
        report["predictions_sha256"] = file_digest(output / "predictions.jsonl")
    write_json(output / "run.json", report)
    return report
