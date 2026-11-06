"""Predeclared pilot, paired confirmation, immutable selection and audit sidecars."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cascade_classifier.experiment import apply_decisions, comparison, verify_files
from cascade_classifier.routing import select_targets
from ensemble_classifier.files import read_json
from gpt_classifier.backend import Cache, file_lock
from gpt_classifier.data import Example, Text, canonical, digest, file_digest, read_jsonl, write_json, write_jsonl
from .data import implementation, load_study
from .execution import RequestRunner
from .prompt import build_request
from .retrieval import Retriever


def request_items(path):
    """A valid selective cohort may contain no eligible targets."""
    return read_jsonl(path) if path.stat().st_size else []


def arms(config, stage, pilot_selection=None):
    if stage not in {"pilot", "confirmation"}:
        raise ValueError("Stage must be pilot or confirmation.")
    result = [{"name": "text", "include_local": False, "method": None, "count": 0},
              {"name": "local", "include_local": True, "method": None, "count": 0}]
    methods = [(method, count) for method in config.retrieval_methods for count in config.example_counts]
    if stage == "confirmation":
        if not pilot_selection or pilot_selection["status"] != "completed":
            raise ValueError("Confirmation requires a completed pilot selection.")
        methods = [(pilot_selection["method"], pilot_selection["count"])]
    for method, count in methods:
        for local in (False, True):
            result.append({"name": f"{method}-{count}-{'local' if local else 'rag'}", "method": method,
                           "count": count, "include_local": local})
    return result


def stage_rows(directory, study, config, stage):
    identifiers = set(study["cohorts"]["pilot_ids" if stage == "pilot" else "confirmation_ids"])
    rows = [row for row in read_jsonl(directory / "oof-routing.jsonl") if row["excel_row"] in identifiers]
    # The pilot contains difficult targets already. Confirmation ranks its own entire cohort.
    selected = sorted(rows, key=lambda row: row["uncertainty_rank"]) if stage == "pilot" else select_targets(rows, max(config.transport.budgets))
    target_map = {row["excel_row"]: Text(**row) for row in read_jsonl(directory / "targets.jsonl")}
    return rows, selected, target_map


def plan(directory: Path, stage: str, *, retriever_factory=Retriever, runtime=None):
    study, config, bank, components = load_study(directory)
    destination = directory / stage
    if (destination / "plan.json").exists():
        saved = read_json(destination / "plan.json")
        verify_files(destination, saved["files"])
        if saved["study_sha256"] != file_digest(directory / "study.json"):
            raise ValueError("Existing stage belongs to another study.")
        return saved
    pilot = read_json(directory / "pilot/selection.json") if stage == "confirmation" else None
    recipes = arms(config, stage, pilot)
    rows, selected, target_map = stage_rows(directory, study, config, stage)
    retrieval_config = replace(config, retrieval_methods=(pilot["method"],), example_counts=(pilot["count"],)) if pilot else config
    retriever = retriever_factory(bank, components, retrieval_config)
    retriever.prepare_targets([target_map[row["excel_row"]] for row in selected])
    runtime = runtime or RequestRunner(config.transport, Path(config.response_cache)).backend_factory().metadata()
    destination.mkdir(parents=True, exist_ok=True)
    files, retrieval_cache, previews = {}, {}, {}
    for recipe in recipes:
        items, traces = [], []
        for row in selected:
            key = (row["excel_row"], recipe["method"], recipe["count"])
            if recipe["method"]:
                if key not in retrieval_cache:
                    retrieval_cache[key] = retriever.retrieve(target_map[row["excel_row"]], method=recipe["method"], count=recipe["count"])
                examples, diagnostics = retrieval_cache[key]
            else:
                examples, diagnostics = [], {}
            request = build_request(target_map[row["excel_row"]], row, examples,
                                    include_local=recipe["include_local"], config=config.transport, runtime=runtime)
            items.append({"excel_row": row["excel_row"], "request_key": Cache.key(request), "request": request})
            traces.append({"excel_row": row["excel_row"], "text_sha256": row["text_sha256"], "recipe": recipe,
                           "retrieved": examples, "retrieval_diagnostics": diagnostics, "request_key": Cache.key(request),
                           "request_bytes": len(canonical(request).encode()), "gold_in_request": False})
        arm = destination / recipe["name"]
        write_jsonl(arm / "requests.jsonl", items)
        write_jsonl(arm / "context-trace.jsonl", traces)
        for name in ("requests.jsonl", "context-trace.jsonl"):
            files[f"{recipe['name']}/{name}"] = file_digest(arm / name)
        _, previews[recipe["name"]] = RequestRunner(config.transport, Path(config.response_cache)).run(items, arm / "gpt")
    result = {"status": "planned", "stage": stage, "recipes": recipes, "targets": len(selected),
              "cohort_rows": len(rows), "retrieval": retriever.identity, "runtime": runtime,
              "study_sha256": file_digest(directory / "study.json"), "implementation": implementation(), "files": files,
              "pilot_selection_sha256": file_digest(directory / "pilot/selection.json") if pilot else None,
              "previews": previews, "new_batches": sum(row["new_batches"] for row in previews.values()),
              "classification_executed": False}
    write_json(destination / "plan.json", result)
    return result


def run(directory: Path, stage: str, *, execute=False, max_new_requests=None, runner_factory=RequestRunner):
    study, config, _, _ = load_study(directory)
    destination = directory / stage
    planned = read_json(destination / "plan.json")
    verify_files(destination, planned["files"])
    if planned["study_sha256"] != file_digest(directory / "study.json") or planned["implementation"] != implementation():
        raise ValueError("Stage code or study changed after planning.")
    if execute and (destination / "run.json").exists():
        previous = read_json(destination / "run.json")
        if previous.get("status") == "completed":
            verify_files(destination, previous["files"])
            if previous["plan_sha256"] != file_digest(destination / "plan.json"):
                raise ValueError("Completed stage plan changed.")
            return previous
    rows, selected, _ = stage_rows(directory, study, config, stage)
    limit = config.max_new_requests if max_new_requests is None else max_new_requests
    if type(limit) is not int or limit < 0:
        raise ValueError("An explicit nonnegative call limit is required.")
    runner = runner_factory(config.transport, Path(config.response_cache))
    previews = {}
    for recipe in planned["recipes"]:
        arm = destination / recipe["name"]
        items = request_items(arm / "requests.jsonl")
        _, previews[recipe["name"]] = runner.run(items, arm / "gpt")
    required = sum(row["new_batches"] for row in previews.values())
    if not execute:
        return {"status": "preview", "new_batches": required, "max_new_requests": limit, "arms": previews,
                "classification_executed": False}
    if required > limit:
        raise ValueError(f"Stage requires {required} new batches, above {limit}; no classification started.")
    calls, files, remaining = {}, {}, limit
    with file_lock(destination / ".stage.lock"):
        for recipe in planned["recipes"]:
            arm = destination / recipe["name"]
            decisions, report = runner.run(request_items(arm / "requests.jsonl"), arm / "gpt", execute=True, max_new_requests=remaining)
            remaining -= report["new_attempts"]
            predictions = apply_decisions(rows, selected, decisions)
            write_jsonl(arm / "predictions.jsonl", predictions)
            calls[recipe["name"]] = report
            for name in ("predictions.jsonl", "gpt/decisions.json", "gpt/run.json"):
                files[f"{recipe['name']}/{name}"] = file_digest(arm / name)
        result = {"status": "completed", "stage": stage, "plan_sha256": file_digest(destination / "plan.json"),
                  "calls": calls, "files": files, "new_attempts": limit - remaining,
                  "classification_executed": True, "test1_accessed": False, "test2_accessed": False}
        write_json(destination / "run.json", result)
        return result


def score(directory: Path, stage: str):
    study, config, _, _ = load_study(directory)
    destination = directory / stage
    if (destination / "selection.json").exists():
        saved = read_json(destination / "selection.json")
        if saved["run_sha256"] != file_digest(destination / "run.json"):
            raise ValueError("Scored stage changed; do not silently retune it.")
        return saved
    planned, completed = read_json(destination / "plan.json"), read_json(destination / "run.json")
    verify_files(destination, planned["files"])
    verify_files(destination, completed["files"])
    if completed["status"] != "completed" or completed["plan_sha256"] != file_digest(destination / "plan.json"):
        raise ValueError("Complete and persist every arm before opening validation labels.")
    rows, selected, _ = stage_rows(directory, study, config, stage)
    # Labels are loaded only after every planned prediction file has been checked.
    identifiers = {row["excel_row"] for row in rows}
    gold = [Example(**row) for row in read_jsonl(directory / "validation-gold.jsonl") if row["excel_row"] in identifiers]
    budgets = [config.transport.initial_budget] if stage == "pilot" else [0., *config.transport.budgets]
    results = {}
    for recipe in planned["recipes"]:
        arm = destination / recipe["name"]
        decisions = {int(key): value for key, value in read_json(arm / "gpt/decisions.json").items()}
        results[recipe["name"]] = {}
        for budget in budgets:
            chosen = selected if stage == "pilot" else select_targets(rows, budget)
            predictions = apply_decisions(rows, chosen, decisions)
            report, changes = comparison(gold, rows, predictions, chosen, decisions, config.transport)
            results[recipe["name"]][str(budget)] = report
            write_jsonl(arm / f"changes-{budget}.jsonl", changes)
    if stage == "pilot":
        candidates = []
        for method in config.retrieval_methods:
            for count in config.example_counts:
                pair = [results[f"{method}-{count}-{suffix}"][str(config.transport.initial_budget)] for suffix in ("rag", "local")]
                if any(row["fallback_instances"] for row in pair):
                    continue
                candidates.append((sum(row["metrics"]["f1_macro"] for row in pair) / 2,
                                   -count, method == "tfidf", method, count))
        if not candidates:
            raise ValueError("No complete retrieval pair; retain the run/cache and resolve failures before selection.")
        *_, method, count = max(candidates)
        selection = {"method": method, "count": count,
                     "criterion": "Mean paired RAG-only/RAG+local pilot macro-F1; ties prefer fewer examples, then lexical retrieval"}
    else:
        candidates = [(report["metrics"]["f1_macro"], -budget, -report["estimated_api_equivalent_usd"], recipe["name"], budget)
                      for recipe in planned["recipes"] for budget in budgets
                      if not (report := results[recipe["name"]][str(budget)])["fallback_instances"]]
        _, _, _, winner, budget = max(candidates, key=lambda candidate: (*candidate[:3], candidate[3] == "text"))
        selection = {"recipe": next(recipe for recipe in planned["recipes"] if recipe["name"] == winner),
                     "budget": budget, "criterion": "Confirmation macro-F1; ties prefer lower budget and allocated cost; zero-budget baseline is eligible"}
    result = {"status": "completed", "stage": stage, "scores": results, **selection,
              "plan_sha256": file_digest(destination / "plan.json"), "run_sha256": file_digest(destination / "run.json"),
              "selection_data": "Validation OOF only; pilot and confirmation similarity groups are disjoint",
              "test1_accessed": False, "test2_accessed": False}
    write_json(destination / "selection.json", result)
    return result


def freeze(directory: Path, output: Path):
    if output.exists():
        raise FileExistsError("Use a new frozen RAG package.")
    study, config, _, _ = load_study(directory)
    selection = read_json(directory / "confirmation/selection.json")
    for stage in ("pilot", "confirmation"):
        run_data = read_json(directory / stage / "run.json")
        verify_files(directory / stage, run_data["files"])
        scored = read_json(directory / stage / "selection.json")
        stage_plan = read_json(directory / stage / "plan.json")
        verify_files(directory / stage, stage_plan["files"])
        if (scored["run_sha256"] != file_digest(directory / stage / "run.json")
                or scored["plan_sha256"] != file_digest(directory / stage / "plan.json")
                or run_data["plan_sha256"] != scored["plan_sha256"]):
            raise ValueError("Selection changed after scoring.")
    confirmation = read_json(directory / "confirmation/plan.json")
    if confirmation["pilot_selection_sha256"] != file_digest(directory / "pilot/selection.json"):
        raise ValueError("Confirmation used another pilot selection.")
    artifact = {"version": 1, "type": "selective_rag_stacking", "config": config.to_dict(), "bank": study["bank"],
                "routing_rule": study["routing_rule"], "recipe": selection["recipe"], "budget": selection["budget"],
                "retrieval_identity": confirmation["retrieval"], "runtime": confirmation["runtime"],
                "implementation": implementation(), "study_sha256": file_digest(directory / "study.json"),
                "selection_sha256": file_digest(directory / "confirmation/selection.json"),
                "test1_accessed": False, "test2_accessed": False}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "model.json", artifact)
    write_json(output / "pilot.json", read_json(directory / "pilot/selection.json"))
    write_json(output / "validation.json", selection)
    write_json(output / "manifest.json", {name: file_digest(output / name) for name in ("model.json", "pilot.json", "validation.json")})
    return artifact
