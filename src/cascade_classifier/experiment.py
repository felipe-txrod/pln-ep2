"""Validation-only selection, immutable publication, label-free prediction and final scoring."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from ensemble_classifier.files import align, read_json, read_predictions
from ensemble_classifier.soft_sources import location
from ensemble_classifier.soft_voting import QWEN_MODELS, aligned_scores
from ensemble_classifier.stacking import meta_probabilities
from ensemble_classifier.stacking_pipeline import load_stacking_artifact, saved_soft_inputs
from gpt_classifier.backend import file_lock
from gpt_classifier.data import Example, Text, canonical, digest, file_digest, read_jsonl, read_xlsx, write_json, write_jsonl
from gpt_classifier.evaluation import metrics
from .config import CascadeConfig
from .dataset import nested_predictions, validation_inputs
from .gpt import SelectiveGPT, estimated_cost
from .routing import RoutingRule, choose_rule, rank_targets, select_targets


def implementation():
    root = Path(__file__).resolve().parents[1]
    paths = list(Path(__file__).parent.glob("*.py"))
    paths.extend(root / "gpt_classifier" / name for name in ("backend.py", "codex_backend.py", "batching.py", "prompt.py"))
    return {str(path.relative_to(root)): file_digest(path) for path in sorted(paths)}


def verify_files(root, expected):
    for name, expected_sha in expected.items():
        if file_digest(root / name) != expected_sha:
            raise ValueError(f"Frozen experiment input changed: {name}")


def prepare_validation(config: CascadeConfig, output: Path):
    if output.exists():
        raise FileExistsError("Use a new validation preparation directory.")
    records, groups, members, native_labels, evidence = validation_inputs(config)
    oof, folds, audits = nested_predictions(records, groups, members)
    targets = [Text(record.excel_row, record.text) for record in records]
    rule, search = choose_rule(targets, oof, members, [record.label for record in records], config.initial_budget,
                               native_labels=native_labels)
    rows = rank_targets(targets, oof, members, rule, native_labels=native_labels)
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "targets.jsonl", [{"excel_row": record.excel_row, "text": record.text} for record in targets])
    write_jsonl(output / "validation-gold.jsonl", records)
    write_jsonl(output / "oof-routing.jsonl", [dict(row, outer_fold=int(fold)) for row, fold in zip(rows, folds)])
    write_json(output / "nested-cv.json", audits)
    write_json(output / "routing-search.json", search)
    files = {name: file_digest(output / name) for name in ("targets.jsonl", "validation-gold.jsonl", "oof-routing.jsonl",
                                                          "nested-cv.json", "routing-search.json")}
    report = {"status": "completed", "config": config.to_dict(), "routing_rule": asdict(rule), "files": files,
              "evidence": evidence, "probability_source": "Nested 5x5 grouped OOF meta-model predictions",
              "outer_target_excluded_from_C_selection": True, "test1_accessed": False, "test2_accessed": False,
              "oof_baseline_metrics": metrics(records, rows)}
    write_json(output / "preparation.json", report)
    return report


def prepared_inputs(directory):
    report = read_json(directory / "preparation.json")
    verify_files(directory, report["files"])
    targets = [Text(**row) for row in read_jsonl(directory / "targets.jsonl")]
    rows = read_jsonl(directory / "oof-routing.jsonl")
    align(targets, rows)
    return report, targets, rows


def apply_decisions(rows, selected, decisions):
    selected_ids = {row["excel_row"] for row in selected}
    result = []
    for row in rows:
        decision = decisions.get(row["excel_row"], {}) if row["excel_row"] in selected_ids else {}
        label = decision.get("label")
        if label is not None and label not in ("c1", "c234", "c5"):
            raise ValueError("Invalid GPT class; refusing to substitute it.")
        result.append({"excel_row": row["excel_row"], "text_sha256": row["text_sha256"],
                       "label": label if label is not None else row["label"], "stacking_label": row["label"],
                       "routed": row["excel_row"] in selected_ids,
                       "decision_source": "gpt" if label is not None else "fallback" if row["excel_row"] in selected_ids else "stacking"})
    return result


def comparison(gold, baseline, predictions, selected, decisions, config):
    truth = {row.excel_row: row.label for row in gold}
    changed = [{"excel_row": old["excel_row"], "text_sha256": old["text_sha256"], "gold": truth[old["excel_row"]],
                "stacking": old["label"], "cascade": new["label"],
                "category": "corrected" if new["label"] == truth[old["excel_row"]] else
                            "harmed" if old["label"] == truth[old["excel_row"]] else "both_wrong_changed_label"}
               for old, new in zip(baseline, predictions) if old["label"] != new["label"]]
    receipts = {}
    for row in selected:
        value = decisions.get(row["excel_row"], {})
        if value.get("label") is not None:
            receipts[value.get("item_request_key", (value["request_key"], row["text_sha256"]))] = value
    usage = {key: sum(value.get("usage_allocation", {}).get(key, 0) for value in receipts.values())
             for key in ("input_tokens", "output_tokens", "cached_input_tokens")}
    result = {"metrics": metrics(gold, predictions), "routed_instances": len(selected),
              "successful_gpt_instances": sum(row["decision_source"] == "gpt" for row in predictions),
              "fallback_instances": sum(row["decision_source"] == "fallback" for row in predictions),
              "changes": {category: sum(row["category"] == category for row in changed)
                          for category in ("corrected", "harmed", "both_wrong_changed_label")},
              "amortized_usage": usage, "estimated_api_equivalent_usd": estimated_cost(usage, config),
              "allocated_request_latency_seconds": sum(value.get("latency_allocation_seconds", 0) for value in receipts.values()),
              "distinct_batch_receipts": len({value["request_key"] for value in receipts.values()}),
              "accounting_note": "Per-instance shares of shared batches; actual experiment attempts/cost/latency are recorded separately."}
    return result, changed


def validate_initial(directory: Path, cache_path: Path, *, service=None):
    report, targets, rows = prepared_inputs(directory)
    config = CascadeConfig(**{**report["config"], "budgets": tuple(report["config"]["budgets"])})
    destination = directory / "initial"
    service = service or SelectiveGPT(config, cache_path)
    with file_lock(directory / ".validation.lock"):
        if (destination / "selection.json").exists():
            return read_json(destination / "selection.json")
        selected = select_targets(rows, config.initial_budget)
        outputs, decisions_by_mode, calls = {}, {}, {}
        for mode in ("text_only", "with_local"):
            decisions, run = service.execute(targets, selected, destination / mode, include_local=mode == "with_local")
            predictions = apply_decisions(rows, selected, decisions)
            write_jsonl(destination / mode / "predictions.jsonl", predictions)
            outputs[mode], decisions_by_mode[mode], calls[mode] = predictions, decisions, run
        # Validation gold is read only after both prompt variants have persisted predictions.
        gold = [Example(**row) for row in read_jsonl(directory / "validation-gold.jsonl")]
        scores = {mode: comparison(gold, rows, outputs[mode], selected, decisions_by_mode[mode], config)[0]
                  for mode in outputs}
        mode = max(scores, key=lambda name: (scores[name]["metrics"]["f1_macro"],
                                             scores[name]["metrics"]["accuracy"], name == "text_only"))
        result = {"status": "completed", "budget": config.initial_budget, "scores": scores, "calls": calls,
                  "selected_prompt": mode, "test1_accessed": False, "selection_data": "Nested OOF validation only",
                  "files": {f"{mode}/{name}": file_digest(destination / mode / name)
                            for mode in outputs for name in ("predictions.jsonl", "decisions.json", "run.json")},
                  "preparation_sha256": file_digest(directory / "preparation.json")}
        write_json(destination / "selection.json", result)
        return result


def validate_extended(directory: Path, cache_path: Path, *, service=None):
    report, targets, rows = prepared_inputs(directory)
    initial = read_json(directory / "initial/selection.json")
    verify_files(directory / "initial", initial["files"])
    if initial["preparation_sha256"] != file_digest(directory / "preparation.json"):
        raise ValueError("Prompt selection belongs to another preparation.")
    config = CascadeConfig(**{**report["config"], "budgets": tuple(report["config"]["budgets"])})
    destination = directory / "extended"
    with file_lock(directory / ".validation.lock"):
        if (destination / "comparison.json").exists():
            return read_json(destination / "comparison.json")
        selected = select_targets(rows, max(config.budgets))
        service = service or SelectiveGPT(config, cache_path)
        decisions, calls = service.execute(targets, selected, destination, include_local=initial["selected_prompt"] == "with_local")
        outputs = {str(budget): apply_decisions(rows, select_targets(rows, budget), decisions) for budget in (0., *config.budgets)}
        for budget, predictions in outputs.items():
            write_jsonl(destination / f"predictions-{budget}.jsonl", predictions)
        gold = [Example(**row) for row in read_jsonl(directory / "validation-gold.jsonl")]
        scores = {}
        for budget in (0., *config.budgets):
            score, changes = comparison(gold, rows, outputs[str(budget)], select_targets(rows, budget), decisions, config)
            scores[str(budget)] = score
            write_jsonl(destination / f"changes-{budget}.jsonl", changes)
        best = max((0., *config.budgets), key=lambda budget: (scores[str(budget)]["metrics"]["f1_macro"],
                                                             -budget, scores[str(budget)]["metrics"]["accuracy"]))
        result = {"status": "completed", "scores": scores, "calls": calls, "selected_budget": best,
                  "selected_prompt": initial["selected_prompt"], "test1_accessed": False,
                  "files": {name: file_digest(destination / name) for name in
                            ("decisions.json", "run.json", *(f"predictions-{budget}.jsonl" for budget in (0., *config.budgets)))},
                  "initial_selection_sha256": file_digest(directory / "initial/selection.json"),
                  "preparation_sha256": file_digest(directory / "preparation.json")}
        write_json(destination / "comparison.json", result)
        return result


def freeze(directory: Path, output: Path):
    if output.exists():
        raise FileExistsError("Use a new frozen cascade package.")
    preparation, _, _ = prepared_inputs(directory)
    initial_path, comparison_path = directory / "initial/selection.json", directory / "extended/comparison.json"
    selection = read_json(comparison_path)
    verify_files(directory / "extended", selection["files"])
    verify_files(directory / "initial", read_json(initial_path)["files"])
    if (selection["preparation_sha256"] != file_digest(directory / "preparation.json")
            or selection["initial_selection_sha256"] != file_digest(initial_path)):
        raise ValueError("Validation selections changed before freezing.")
    config = preparation["config"]
    load_stacking_artifact(location(config["stacking_model"]))
    if file_digest(location(config["stacking_model"])) != preparation["evidence"]["stacking_artifact_sha256"]:
        raise ValueError("Stacking changed after routing validation.")
    artifact = {"version": 1, "model_type": "stacking_selective_gpt", "config": config,
                "routing_rule": preparation["routing_rule"], "prompt_mode": selection["selected_prompt"],
                "selected_budget": selection["selected_budget"], "comparison_budgets": [0., *config["budgets"]],
                "evidence": preparation["evidence"], "implementation": implementation(), "test1_accessed": False,
                "validation_sha256": file_digest(comparison_path), "prompt_selection_sha256": file_digest(initial_path),
                "preparation_sha256": file_digest(directory / "preparation.json"),
                "selection_rule": "Validation macro-F1; ties prefer lower routing budget. Zero budget is eligible."}
    artifact["codex_runtime"] = read_json(directory / "extended/run.json")["runtime"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "model.json", artifact)
    write_json(output / "validation.json", selection)
    write_json(output / "prompt-ablation.json", read_json(initial_path))
    write_json(output / "preparation.json", preparation)
    write_json(output / "manifest.json", {name: file_digest(output / name) for name in
                                         ("model.json", "validation.json", "prompt-ablation.json", "preparation.json")})
    return artifact


def load_cascade(path):
    artifact = read_json(path)
    verify_files(path.parent, read_json(path.with_name("manifest.json")))
    if (artifact.get("version") != 1 or artifact.get("model_type") != "stacking_selective_gpt"
            or artifact.get("implementation") != implementation()):
        raise ValueError("Cascade implementation changed; preserve this package and freeze a reviewed new version.")
    config = CascadeConfig(**{**artifact["config"], "budgets": tuple(artifact["config"]["budgets"])})
    if file_digest(location(config.stacking_model)) != artifact["evidence"]["stacking_artifact_sha256"]:
        raise ValueError("The selected stacking identity changed.")
    RoutingRule(**artifact["routing_rule"])
    if (artifact["selected_budget"] not in artifact["comparison_budgets"]
            or artifact["comparison_budgets"] != [0., *config.budgets]
            or artifact["prompt_mode"] not in {"text_only", "with_local"}):
        raise ValueError("Invalid frozen cascade selection.")
    return artifact, config


def predict_cascade(model_path, source_path, stacking_run, output, cache_path, *, service=None, compare_budgets=False):
    artifact, config = load_cascade(model_path)
    stack_path = location(config.stacking_model)
    stack, _ = load_stacking_artifact(stack_path)
    saved = read_json(stacking_run / "run.json")
    if saved.get("artifact_sha256") != file_digest(stack_path) or saved.get("status") != "completed":
        raise ValueError("A completed prediction run from the frozen stacking is required.")
    targets, source, calibrated, member_predictions, snapshots = saved_soft_inputs(
        location(stack["soft_model"]), source_path, location(saved["soft_run"]))
    predictions = read_predictions(stacking_run / "predictions.jsonl")
    expected = meta_probabilities(calibrated, stack["meta_model"])
    if (file_digest(stacking_run / "predictions.jsonl") != saved["predictions_sha256"]
            or saved["soft_files"] != snapshots or saved["input"]["sha256"] != source["sha256"]
            or not np.allclose(aligned_scores(predictions, "probabilities"), expected, atol=1e-12, rtol=0)):
        raise ValueError("Stacking probabilities or input provenance changed.")
    align(targets, predictions)
    native_labels = {name: [row["label"] for row in member_predictions[name]] for name in QWEN_MODELS}
    rows = rank_targets(targets, expected, calibrated, RoutingRule(**artifact["routing_rule"]), native_labels=native_labels)
    budgets = artifact["comparison_budgets"] if compare_budgets else sorted({0., artifact["selected_budget"]})
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".prediction.lock"):
        identity = {"cascade_sha256": file_digest(model_path), "source": source, "budgets": budgets,
                    "stacking_run_sha256": file_digest(stacking_run / "run.json")}
        plan_path = output / "plan.json"
        if plan_path.exists() and read_json(plan_path) != identity:
            raise ValueError("Prediction output belongs to another source or frozen cascade.")
        write_json(plan_path, identity)
        selected = select_targets(rows, max(budgets))
        # The plan depends exclusively on model outputs; it is persisted before any GPT request.
        write_jsonl(output / "routing.jsonl", rows)
        service = service or SelectiveGPT(config, cache_path)
        if selected:
            if service.backend_factory().metadata() != artifact["codex_runtime"]:
                raise ValueError("Codex runtime changed after validation; do not silently change the final experiment.")
            decisions, calls = service.execute(targets, selected, output / "gpt", include_local=artifact["prompt_mode"] == "with_local")
        else:
            decisions, calls = {}, {"attempts": 0, "targets": 0, "wall_seconds": 0, "estimated_api_equivalent_usd": 0}
            write_json(output / "gpt/decisions.json", {})
            write_json(output / "gpt/run.json", calls)
        files = {name: file_digest(output / name) for name in ("routing.jsonl", "gpt/decisions.json", "gpt/run.json")}
        for budget in budgets:
            name = f"predictions-{budget}.jsonl"
            write_jsonl(output / name, apply_decisions(rows, select_targets(rows, budget), decisions))
            files[name] = file_digest(output / name)
        if file_digest(source_path) != source["sha256"]:
            raise ValueError("Source changed during cascade inference.")
        report = {"status": "completed", **identity, "files": files, "rows": len(targets),
                  "selected_budget": artifact["selected_budget"], "calls": calls, "gold_accessed": False,
                  "model_path": str(model_path.resolve()), "parameters_updated": False,
                  "comparison_note": "All budgets were frozen on validation; the comparison union queries at most the largest budget."}
        write_json(output / "run.json", report)
        return report


def evaluate_cascade(run, gold_path, output):
    if output.exists():
        raise FileExistsError("Use a new final evaluation directory.")
    manifest = read_json(run / "run.json")
    artifact, config = load_cascade(Path(manifest["model_path"]))
    if (manifest.get("status") != "completed" or manifest.get("rows") != 900 or manifest.get("gold_accessed") is not False
            or manifest["cascade_sha256"] != file_digest(Path(manifest["model_path"]))):
        raise ValueError("Final evaluation requires a completed immutable, gold-free 900-row run.")
    verify_files(run, manifest["files"])
    rows = read_jsonl(run / "routing.jsonl")
    decisions = {int(key): value for key, value in read_json(run / "gpt/decisions.json").items()}
    budgets = manifest["budgets"]
    if not set(budgets) <= set(artifact["comparison_budgets"]):
        raise ValueError("Unfrozen evaluation budgets.")
    predictions = {str(budget): read_jsonl(run / f"predictions-{budget}.jsonl") for budget in budgets}
    for budget in budgets:
        if predictions[str(budget)] != apply_decisions(rows, select_targets(rows, budget), decisions):
            raise ValueError("Saved cascade decisions differ from the frozen routing plan.")
    # First final-gold access is after every budget prediction has been checked.
    gold, source = read_xlsx(gold_path, labeled=True)
    if len(gold) != 900:
        raise ValueError("Final evaluation requires all 900 rows.")
    for values in predictions.values():
        align([Text(record.excel_row, record.text) for record in gold], values)
    output.mkdir(parents=True, exist_ok=False)
    scores = {}
    for budget in budgets:
        score, changes = comparison(gold, rows, predictions[str(budget)], select_targets(rows, budget), decisions, config)
        scores[str(budget)] = score
        write_jsonl(output / f"changes-{budget}.jsonl", changes)
    result = {"status": "completed", "scores": scores, "selected_budget_from_validation": artifact["selected_budget"],
              "selected_prompt_from_validation": artifact["prompt_mode"], "parameters_updated": False,
              "gold_read_after_all_predictions": True, "gold_sha256": source["sha256"],
              "cascade_sha256": manifest["cascade_sha256"], "calls": manifest["calls"],
              "limitations": "Test1 has historical project exposure; no final labels selected this cascade. Holdout exposure includes Qwen checkpoint selection."}
    if file_digest(gold_path) != source["sha256"]:
        raise ValueError("Gold changed during evaluation.")
    write_json(output / "metrics.json", result)
    return result
