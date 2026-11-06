"""Frozen inference with an enforced test2-before-test1 provenance chain."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from cascade_classifier.experiment import apply_decisions, comparison, verify_files
from cascade_classifier.routing import RoutingRule, rank_targets, select_targets
from ensemble_classifier.files import align, read_json, read_predictions
from ensemble_classifier.soft_sources import ROOT, location
from ensemble_classifier.soft_voting import QWEN_MODELS, aligned_scores
from ensemble_classifier.stacking import meta_probabilities
from ensemble_classifier.stacking_pipeline import load_stacking_artifact, saved_soft_inputs
from gpt_classifier.backend import Cache, file_lock
from gpt_classifier.data import Text, file_digest, read_jsonl, read_xlsx, write_json, write_jsonl
from gpt_classifier.delivery import export_test1, export_test2
from .config import RagConfig
from .data import implementation, training_bank
from .execution import RequestRunner
from .prompt import build_request
from .retrieval import Retriever


def load_model(path):
    verify_files(path.parent, read_json(path.with_name("manifest.json")))
    artifact = read_json(path)
    if artifact.get("version") != 1 or artifact.get("type") != "selective_rag_stacking" or artifact["implementation"] != implementation():
        raise ValueError("Frozen RAG model/source identity changed.")
    config = RagConfig.from_dict(artifact["config"])
    bank, components, identity = training_bank(config)
    if identity != artifact["bank"] or artifact["budget"] not in (0., *config.transport.budgets):
        raise ValueError("RAG bank or selected budget differs from validation.")
    recipe = artifact["recipe"]
    if (recipe["method"] is not None and (recipe["method"] not in config.retrieval_methods
                                          or recipe["count"] not in config.example_counts)):
        raise ValueError("Unvalidated retrieval recipe.")
    return artifact, config, bank, components


def require_test2(run: Path | None, model_sha: str, test2_sha: str):
    if run is None:
        raise ValueError("Official test1 requires a completed test2 run from this exact frozen RAG configuration.")
    report = read_json(run / "run.json")
    if (report.get("status") != "completed" or report.get("phase") != "test2" or report.get("rows") != 897
            or report.get("model_sha256") != model_sha or report.get("source", {}).get("sha256") != test2_sha
            or report.get("gold_accessed") is not False
            or not {"predictions.jsonl", "test2.xlsx"} <= set(report.get("files", {}))):
        raise ValueError("The required test2 run has an incompatible model, source or completion status.")
    verify_files(run, report["files"])
    return {"run": str(run.resolve()), "run_sha256": file_digest(run / "run.json"),
            "predictions_sha256": file_digest(run / "predictions.jsonl")}


def frozen_inputs(config, source_path, stacking_run):
    stack_path = location(config.transport.stacking_model)
    stack, _ = load_stacking_artifact(stack_path)
    saved = read_json(stacking_run / "run.json")
    if saved.get("status") != "completed" or saved["artifact_sha256"] != file_digest(stack_path):
        raise ValueError("Use a completed run from the unchanged four-member stacking.")
    targets, source, calibrated, members, snapshots = saved_soft_inputs(
        location(stack["soft_model"]), source_path, location(saved["soft_run"]))
    predictions = read_predictions(stacking_run / "predictions.jsonl")
    expected = meta_probabilities(calibrated, stack["meta_model"])
    align(targets, predictions)
    if (saved["input"]["sha256"] != source["sha256"] or saved["soft_files"] != snapshots
            or file_digest(stacking_run / "predictions.jsonl") != saved["predictions_sha256"]
            or not np.allclose(aligned_scores(predictions, "probabilities"), expected, atol=1e-12, rtol=0)):
        raise ValueError("Saved stacking probabilities or source identities changed.")
    return targets, source, calibrated, members, expected


def predict(model_path: Path, source_path: Path, stacking_run: Path, output: Path, *, test2_run=None,
            execute=False, max_new_requests=None, runner_factory=RequestRunner, retriever_factory=Retriever):
    artifact, config, bank, components = load_model(model_path)
    model_sha = file_digest(model_path)
    source_sha = file_digest(source_path)
    official_test2_sha = file_digest(ROOT / "data/test2.xlsx")
    test1_hashes = {file_digest(ROOT / "data" / name) for name in
                   ("test1.xlsx", "test1-rotulado-para-teste-estatistico.xlsx")}
    phase = "test1" if source_sha in test1_hashes else "test2" if source_sha == official_test2_sha else "other"
    test2 = require_test2(test2_run, model_sha, official_test2_sha) if phase == "test1" else None
    targets, source, calibrated, members, expected = frozen_inputs(config, source_path, stacking_run)
    native = {name: [row["label"] for row in members[name]] for name in QWEN_MODELS}
    rows = rank_targets(targets, expected, calibrated, RoutingRule(**artifact["routing_rule"]), native_labels=native)
    selected = select_targets(rows, artifact["budget"])
    target_map = {target.excel_row: target for target in targets}
    identity = {"model_sha256": model_sha, "source": source, "phase": phase,
                "stacking_run_sha256": file_digest(stacking_run / "run.json"), "test2_reference": test2}
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".prediction.lock"):
        if (output / "plan.json").exists() and read_json(output / "plan.json")["identity"] != identity:
            raise ValueError("Output belongs to another frozen prediction plan.")
        if (output / "run.json").exists():
            completed = read_json(output / "run.json")
            if completed["status"] == "completed":
                verify_files(output, completed["files"])
                return completed
        recipe, items, traces = artifact["recipe"], [], []
        if (output / "requests.jsonl").exists() and (output / "plan.json").exists():
            verify_files(output, read_json(output / "plan.json")["files"])
            items = read_jsonl(output / "requests.jsonl") if selected else []
        else:
            retriever = None
            if selected and recipe["method"]:
                retrieval_config = replace(config, retrieval_methods=(recipe["method"],), example_counts=(recipe["count"],))
                retriever = retriever_factory(bank, components, retrieval_config)
                if retriever.identity != artifact["retrieval_identity"]:
                    raise ValueError("Retrieval index/encoder differs from confirmation.")
                retriever.prepare_targets([target_map[row["excel_row"]] for row in selected])
            for row in selected:
                examples, diagnostics = retriever.retrieve(target_map[row["excel_row"]], method=recipe["method"], count=recipe["count"]) if retriever else ([], {})
                request = build_request(target_map[row["excel_row"]], row, examples, include_local=recipe["include_local"],
                                        config=config.transport, runtime=artifact["runtime"])
                items.append({"excel_row": row["excel_row"], "request_key": Cache.key(request), "request": request})
                traces.append({"excel_row": row["excel_row"], "retrieved": examples, "diagnostics": diagnostics,
                               "request_key": Cache.key(request), "gold_in_request": False})
            write_jsonl(output / "routing.jsonl", rows)
            write_jsonl(output / "requests.jsonl", items)
            write_jsonl(output / "context-trace.jsonl", traces)
            write_json(output / "plan.json", {"identity": identity,
                       "files": {name: file_digest(output / name) for name in ("routing.jsonl", "requests.jsonl", "context-trace.jsonl")}})
        runner = runner_factory(config.transport, Path(config.response_cache))
        limit = config.max_new_requests if max_new_requests is None else max_new_requests
        decisions, calls = runner.run(items, output / "gpt", execute=execute, max_new_requests=limit)
        if not execute:
            return {"status": "preview", **identity, "calls": calls, "classification_executed": False}
        predictions = apply_decisions(rows, selected, decisions)
        write_jsonl(output / "predictions.jsonl", predictions)
        if phase == "test2":
            export_test2(source_path, output / "test2.xlsx", predictions)
        elif phase == "test1":
            export_test1(ROOT / "data/test1.xlsx", output / "test1-comparison.xlsx", predictions)
        if file_digest(source_path) != source_sha or file_digest(model_path) != model_sha:
            raise ValueError("Model or source changed during prediction.")
        files = {name: file_digest(output / name) for name in
                 ("routing.jsonl", "requests.jsonl", "context-trace.jsonl", "predictions.jsonl", "gpt/decisions.json", "gpt/run.json")}
        for workbook in ("test2.xlsx", "test1-comparison.xlsx"):
            if (output / workbook).exists():
                files[workbook] = file_digest(output / workbook)
        result = {"status": "completed", **identity, "rows": len(rows), "files": files, "calls": calls,
                  "gold_accessed": False, "model_path": str(model_path.resolve()), "budget": artifact["budget"],
                  "recipe": recipe, "parameters_updated": False}
        write_json(output / "run.json", result)
        return result


def evaluate(run: Path, gold_path: Path, output: Path):
    if output.exists():
        raise FileExistsError("Use a new retrospective evaluation directory.")
    completed = read_json(run / "run.json")
    artifact, config, _, _ = load_model(Path(completed["model_path"]))
    if (completed["status"] != "completed" or completed["phase"] != "test1" or completed["rows"] != 900
            or completed["gold_accessed"] is not False
            or completed["model_sha256"] != file_digest(Path(completed["model_path"]))):
        raise ValueError("A completed gold-free 900-row test1 prediction run is required.")
    verify_files(run, completed["files"])
    reference = completed["test2_reference"]
    verified = require_test2(Path(reference["run"]), completed["model_sha256"], file_digest(ROOT / "data/test2.xlsx"))
    if reference != verified:
        raise ValueError("Test2 changed after test1 prediction.")
    rows, predictions = read_jsonl(run / "routing.jsonl"), read_jsonl(run / "predictions.jsonl")
    decisions = {int(key): value for key, value in read_json(run / "gpt/decisions.json").items()}
    selected = select_targets(rows, artifact["budget"])
    if predictions != apply_decisions(rows, selected, decisions):
        raise ValueError("Final predictions differ from frozen routing and GPT decisions.")
    # The professor's gold workbook is never opened before the test2 chain is validated.
    gold, source = read_xlsx(gold_path, labeled=True)
    if len(gold) != 900:
        raise ValueError("Retrospective scoring requires every original test1 row.")
    score, changed = comparison(gold, rows, predictions, selected, decisions, config.transport)
    baseline, _ = comparison(gold, rows, apply_decisions(rows, [], {}), [], {}, config.transport)
    if file_digest(gold_path) != source["sha256"]:
        raise ValueError("Gold changed during scoring.")
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "changed-cases.jsonl", changed)
    result = {"status": "completed", "stacking": baseline, "rag_cascade": score, "actual_calls": completed["calls"],
              "model_sha256": completed["model_sha256"], "test2_reference": reference, "gold_sha256": source["sha256"],
              "parameters_updated": False, "gold_read_after_test2_and_all_test1_predictions": True,
              "limitations": "Historical test1/holdout exposure and official cross-dataset duplicate texts remain; no test1 label selected this method."}
    write_json(output / "metrics.json", result)
    return result
