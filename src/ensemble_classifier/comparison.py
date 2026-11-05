"""Compare fixed hard-voting configurations as an explicitly exploratory study."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any

from gpt_classifier.data import Text, canonical, digest, file_digest, read_xlsx, write_json, write_jsonl

from .config import EnsembleConfig
from .files import align, member_run, read_predictions
from .strategies import local_decision


def majority_predictions(records: list[Text], members: dict[str, list[dict[str, Any]]],
                         models: tuple[str, ...], tiebreaker: str) -> list[dict[str, Any]]:
    """Use one vote per member and consult the fixed arbitrator only without a majority."""
    if not models or len(set(models)) != len(models):
        raise ValueError("Voting members must be nonempty and distinct.")
    for name in set(models) | {tiebreaker}:
        if name not in members:
            raise ValueError(f"Missing predictions for {name}.")
        align(records, members[name])
    predictions = []
    for index, record in enumerate(records):
        votes = {name: members[name][index]["label"] for name in models}
        decision = local_decision(list(votes.values()), "majority")
        tied = decision is None
        predictions.append({
            "excel_row": record.excel_row, "text_sha256": digest(record.text),
            "label": members[tiebreaker][index]["label"] if tied else decision,
            "decision_source": tiebreaker if tied else "majority",
            "arbitrated": tied, "member_labels": votes,
        })
    return predictions


def compare_majority(config: EnsembleConfig, source: Path, gold: Path, output: Path) -> dict[str, Any]:
    """Persist all candidate predictions before loading gold; never promote a winner."""
    if output.exists():
        raise FileExistsError(f"Use a new comparison output directory: {output}")
    if config.strategy != "majority" or not 2 <= len(config.models) <= 8:
        raise ValueError("Comparison requires majority voting and between two and eight local models.")
    started = perf_counter()
    records, source_info = read_xlsx(source)
    members, provenance = {}, {}
    for name in config.models:
        members[name], provenance[name] = member_run(config.members[name], records, source_info["sha256"])
    implementation = {name: file_digest(Path(__file__).parent / name)
                      for name in ("comparison.py", "strategies.py", "files.py", "config.py")}
    report = {
        "schema_version": 1, "status": "generating_predictions", "phase": "exploratory",
        "input": source_info, "members": provenance,
        "implementation_sha256": implementation,
        "training_performed": False, "model_inference_performed": False,
        "remote_requests_performed": False, "default_configuration_changed": False,
        "policy": "Equal hard votes; strict majority; each local model tested as fixed tiebreaker.",
        "ranking": "Accuracy, then macro-F1, then configuration identifier for deterministic ordering.",
        "limitation": "Configurations compared using gold labels are exploratory, not an independent final test or a delivery selection.",
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.json", report)
    candidates = []
    for size in range(1, len(config.models) + 1):
        for selected in combinations(config.models, size):
            # A singleton cannot tie; the other arbitrators would duplicate it.
            arbitrators = selected if size == 1 else config.models
            for tiebreaker in arbitrators:
                name = "-".join(selected) + "__tie-" + tiebreaker
                directory = output / name
                directory.mkdir()
                predictions = majority_predictions(records, members, selected, tiebreaker)
                configuration = {"models": list(selected), "strategy": "majority", "tiebreaker": tiebreaker,
                                 "members": {model: config.members[model]
                                             for model in sorted(set(selected) | {tiebreaker})}}
                frozen = {**configuration, "members": {model: provenance[model]["identity"]
                          for model in configuration["members"]}, "implementation_sha256": implementation}
                write_json(directory / "config.json", configuration)
                write_jsonl(directory / "predictions.jsonl", predictions)
                candidates.append({"id": name, "models": list(selected), "tiebreaker": tiebreaker,
                                   "ensemble_identity": digest(canonical(frozen)),
                                   "predictions_sha256": file_digest(directory / "predictions.jsonl"),
                                   "label_sequence_sha256": digest(canonical([row["label"] for row in predictions])),
                                   "routed_rows": sum(row["arbitrated"] for row in predictions)})
    if file_digest(source) != source_info["sha256"]:
        raise ValueError("The source workbook changed during combination.")
    report.update(status="predictions_completed", candidates=candidates)
    write_json(output / "run.json", report)

    # Gold is read only after every candidate's text-only prediction has been persisted.
    from gpt_classifier.evaluation import metrics

    truths, gold_info = read_xlsx(gold, labeled=True)
    if len(truths) != len(records) or any(
            truth.excel_row != record.excel_row or truth.text != record.text
            for truth, record in zip(truths, records)):
        raise ValueError("Gold row IDs, texts, and order must exactly match the prediction source.")
    for candidate in candidates:
        directory = output / candidate["id"]
        if file_digest(directory / "predictions.jsonl") != candidate["predictions_sha256"]:
            raise ValueError("Candidate predictions changed before evaluation.")
        predictions = read_predictions(directory / "predictions.jsonl")
        scores = metrics(truths, predictions)
        scores["correct"] = sum(row["label"] == truth.label for row, truth in zip(predictions, truths))
        write_json(directory / "metrics.json", scores)
        candidate.update(scores)
    if file_digest(gold) != gold_info["sha256"]:
        raise ValueError("The gold workbook changed during evaluation.")
    ranked = sorted(candidates, key=lambda row: (-row["correct"], -row["f1_macro"], row["id"]))
    full = [row for row in ranked if len(row["models"]) == len(config.models)]
    agreement = [{"models": [left, right],
                  "agreement_rows": sum(a["label"] == b["label"]
                                        for a, b in zip(members[left], members[right]))}
                 for left, right in combinations(config.models, 2)]
    report.update(status="completed", gold=gold_info, rows=len(records), candidates=ranked,
                  best_by_accuracy=[row["id"] for row in ranked if row["correct"] == ranked[0]["correct"]],
                  best_full_ensemble_by_accuracy=[row["id"] for row in full if row["correct"] == full[0]["correct"]],
                  unique_label_sequences=len({row["label_sequence_sha256"] for row in candidates}),
                  pairwise_agreement=agreement, elapsed_seconds=perf_counter() - started)
    write_json(output / "run.json", report)
    write_json(output / "summary.json", report)
    return report
