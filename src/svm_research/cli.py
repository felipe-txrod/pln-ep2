"""Prepare and execute offline SVM searches without accessing final datasets."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

from ensemble_classifier.files import read_json
from ensemble_classifier.soft_voting import apply_calibration, probability_metrics
from gpt_classifier.backend import file_lock
from gpt_classifier.data import LABELS, digest, file_digest, write_json, write_jsonl
from .data import load_config, load_data, location, runtime
from .search import baseline_candidate, broad_candidates, candidate_id, classification_metrics, search_phase


def implementation() -> dict[str, str]:
    source = Path(__file__).resolve().parents[1]
    paths = list((source / "svm_research").glob("*.py"))
    paths += [source / relative for relative in ("lexical_classifier/inference.py", "gpt_classifier/data.py",
                                               "ensemble_classifier/files.py", "ensemble_classifier/soft_sources.py",
                                               "ensemble_classifier/soft_voting.py")]
    return {str(path.relative_to(source)): file_digest(path) for path in sorted(paths)}


def protocol() -> dict:
    return {
        "objective": "Improve SVM and two-member soft voting without changing NorBERTo or default artifacts",
        "primary_metric": "macro_f1", "training_rows": 16093, "calibration_rows": 2027, "selection_rows": 1972,
        "phase_a": {"candidates": len(broad_candidates()), "folds": 3, "seed": 42,
                    "design": "Eight fixed feature designs per family; four logarithmic C values, two class weights"},
        "phase_b": {"candidate_cap": 61, "folds": 5, "seed_offset": 101,
                    "design": "Top three distinct representations, five local C values, class weights, hinge and union weights"},
        "svm_selection": "One-SE macro-F1 rule; then smallest C, fewest features, squared hinge, no weighting, mean F1, stable ID",
        "calibration_plan": "Nested grouped CV within 2027 rows: five outer folds, three inner folds; sigmoid default, isotonic only if mean inner log loss improves by more than paired SE. Fit final chosen calibrator on 2027 rows. Report nested selection-policy OOF separately from selected-method CV diagnostics.",
        "weight_plan": "Only SVM/NorBERTo; SVM weights 0,0.25,0.5,0.75,1. Nested 5x3 grouped CV on 1972 rows; within one SE of best macro-F1 prefer closest to 0.75. Report outer-fold weights, policy OOF and stability. Final weight uses all meta-validation with same inner selection rule.",
        "comparisons": {"A": "Original calibrated SVM 75% and original calibrated NorBERTo 25%",
                        "B": "Optimized calibrated SVM 75% and unchanged NorBERTo 25%",
                        "C": "Optimized calibrated SVM and unchanged NorBERTo, with nested-selected weights"},
        "primary_decision_before_test1": "Retain A unless B improves over A on meta-validation with paired similarity-group bootstrap 95% lower bound >0, positive delta in at least 4/5 grouped outer folds, and no class F1 loss >0.02. Consider C over B only with the same checks on nested weight-policy OOF and final weight matching at least 3/5 outer choices. Otherwise prefer B if eligible. These are conservative development screens, not hypothesis-test guarantees after adaptive search.",
        "bootstrap": {"repeats": 2000, "seed": 42, "unit": "historical similarity component on meta-validation; normalized text group on test1", "paired": True},
        "test1_plan": "Separate future command after freeze: save every A/B/C prediction before reading gold, verify exact historical A predictions and 0.49584421663197825 macro-F1, retain all 900 rows, report paired errors, group bootstrap, per-class metrics and descriptive McNemar. Never retune or change the development recommendation after scoring.",
        "test2_plan": "Never open during development; final delivery is outside this investigation",
        "forbidden": ["GPT calls", "Stacking", "RAG", "NorBERTo/LR/Qwen training", "historical artifact overwrite", "default promotion", "final-set hyperparameter selection"],
        "limitations": ["Search CV is adaptive, not unbiased end-to-end nested CV.",
                        "The original holdout and test1 have historical exposure.",
                        "Group bootstrap intervals condition on fitted predictions and do not capture all training/selection uncertainty.",
                        "This requested test1 investigation does not satisfy the assignment's original test2-before-test1 order; no test2 access is authorized here."]}


def prepare(config_path: Path, output: Path) -> dict:
    config = load_config(config_path)
    if output.exists():
        raise FileExistsError("Preparation requires a new output directory. Resume with search instead.")
    data = load_data(config, reproduce=True)
    index = data.selection
    labels = np.asarray([r.label for r in data.holdout])
    old_p = apply_calibration(data.old_margins, data.baseline["calibration"]["svm"])
    mixed = .75 * old_p + .25 * data.norberto
    native_prediction = np.asarray(LABELS)[data.old_margins.argmax(axis=1)]
    calibrated_prediction = np.asarray(LABELS)[old_p.argmax(axis=1)]
    ensemble_prediction = np.asarray(LABELS)[mixed.argmax(axis=1)]
    baseline_metrics = {
        "partition": "1972 historical meta-validation rows, development diagnostic only",
        "svm_native": classification_metrics(labels[index], native_prediction[index]),
        "svm_calibrated": classification_metrics(labels[index], calibrated_prediction[index]),
        "A": classification_metrics(labels[index], ensemble_prediction[index]),
        "probabilities_svm": probability_metrics(labels[index], old_p[index]),
        "probabilities_norberto": probability_metrics(labels[index], data.norberto[index]),
        "probabilities_A": probability_metrics(labels[index], mixed[index])}
    expected = read_json(location(config["baseline"]) / "run.json")["selection_metrics"]
    if (not np.isclose(baseline_metrics["A"]["f1_macro"], expected["f1_macro"], atol=1e-14)
            or baseline_metrics["A"]["confusion_matrix"] != expected["confusion_matrix"]):
        raise ValueError("The original soft-voting development baseline did not reproduce.")
    plan = {"status": "prepared", "config": config, "config_sha256": file_digest(config_path),
            "implementation": implementation(), "runtime": runtime(), "protocol": protocol(),
            "input_files": data.audit["files"], "test1_accessed": False, "test2_accessed": False}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    write_json(output / "audit.json", data.audit)
    write_json(output / "baseline.json", baseline_metrics)
    write_jsonl(output / "training-membership.jsonl", [{"excel_row": r.excel_row, "group_id": str(g),
                                                       "text_sha256": digest(r.text), "label": r.label}
                                                      for r, g in zip(data.training, data.groups)])
    write_json(output / "phase-a-candidates.json", broad_candidates())
    report(output)
    return {"status": "prepared", "output": str(output), "candidates": len(broad_candidates()),
            "phase_a_fits": 3 * len(broad_candidates()), "baseline_meta_f1": baseline_metrics["A"]["f1_macro"]}


def validate_plan(config_path: Path, output: Path):
    config = load_config(config_path)
    plan = read_json(output / "plan.json")
    if (config != plan["config"] or file_digest(config_path) != plan["config_sha256"]
            or implementation() != plan["implementation"] or runtime() != plan["runtime"]
            or protocol() != plan["protocol"]):
        raise ValueError("Research implementation, runtime or protocol changed; preserve this run and prepare a new directory.")
    for name, expected in plan["input_files"].items():
        if file_digest(Path(name)) != expected:
            raise ValueError(f"An audited input changed: {name}")
    return config, plan


def report(output: Path) -> None:
    plan, audit, baseline = (read_json(output / name) for name in ("plan.json", "audit.json", "baseline.json"))
    lines = ["# Final SVM and soft-voting investigation", "", "Status: prepared; experiments pending unless recorded below.",
             "", "## Original SVM audit", "", "Original LinearSVC artifact: `" + audit["original_svm"]["artifact_sha256"] + "`.",
             "HTML unescape, whitespace collapse and lowercase; accents and punctuation preserved. Word TF-IDF (1,2), min_df=2, max_df=.98, 60000 features; char_wb TF-IDF (3,5), min_df=3, max_df=1, 50000 features. Both sublinear TF, nonbinary, L2 normalized, float32, smooth IDF. FeatureUnion without branch weights. LinearSVC C=.05, squared hinge, L2 penalty, dual=auto, no class weighting, seed=42, tol=1e-4, max_iter=10000. The original search optimized accuracy; this study optimizes macro-F1.",
             "", "The original SVM has no native predict_proba. The frozen ensemble uses external one-vs-rest sigmoid calibration and the existing NorBERTo temperature; neither is changed during baseline reproduction.",
             "", f"All 3999 original holdout margins reproduced: max absolute difference {audit['baseline_reproduction_max_abs_difference']}. The original 75/25 development ensemble reproduced its saved confusion matrix and macro-F1.",
             "", "## Protocol and provenance", "", "| Role | Rows | Groups |", "|---|---:|---:|"]
    for role in ("training", "calibration", "weight_selection"):
        lines.append(f"| {role} | {audit['rows'][role]} | {audit['groups'][role]} |")
    lines += ["", "Labels come exclusively from original train.xlsx, verified by SHA-256 and row/text identities. Historical grouping: canonical identifier masking plus exhaustive character 3–5 TF-IDF cosine at .93, computed without labels. Reuse its frozen connected components; zero shared groups or normalized duplicates across roles. No final workbook was opened by preparation or search.",
              "", "## Search space and computational plan", "", "Four families: word, char, char_wb, word+char_wb. Eight predeclared feature designs per family vary ngrams, min_df (1–5), max_df (.95–1), sublinear/binary TF, norm (L1/L2/none), and feature caps. Each crosses C=.005/.05/.5/5 and class_weight=None/balanced. See phase-a-candidates.json for every exact combination. This fractional feature design is deliberately bounded and does not estimate isolated feature effects.",
              "", f"Phase A: {len(broad_candidates())} candidates, 3 grouped folds. Phase B: top three distinct representations, at most 61 candidates, 5 grouped folds with a different seed; local logarithmic C, class weighting, hinge loss and union branch weights. Linear kernels fit the sparse high-dimensional setting and available CPU/RAM; nonlinear kernels are excluded from the fixed budget.",
              "", "Every learned vectorizer is inside its CV pipeline. Pipeline caching is training-fold-specific, shared only for identical feature settings. Two worker processes, one numerical thread per fit; cache is reduced to the configured size between representations. Running fits may temporarily exceed the cache cap. Completed folds are checksummed and reused on resume. Convergence warnings make a candidate ineligible; other failures stop the run and remain visible. Full fold metrics, per-class scores, feature counts, timings and predictions are saved.",
              "", "Selection uses the predeclared one-SE rule, favoring smaller C and feature sets. Its CV scores are development diagnostics after multiple adaptive comparisons, not an unbiased nested-CV performance estimate.",
              "", "## Reproduced development baseline", "", "| Model | Accuracy | Macro-F1 |", "|---|---:|---:|"]
    for name in ("svm_native", "svm_calibrated", "A"):
        m = baseline[name]
        lines.append(f"| {name} | {m['accuracy']:.8f} | {m['f1_macro']:.8f} |")
    lines += ["", "These 1972 rows already selected historical ensemble weights; they are not an independent test. See baseline.json for per-class reports, confusion matrices and probability diagnostics."]
    for phase in ("a", "b"):
        path = output / f"phase-{phase}/search.json"
        if not path.exists():
            lines += ["", f"## Phase {phase.upper()} results", "", "Pending execution."]
            continue
        result = read_json(path)
        lines += ["", f"## Phase {phase.upper()} results", "", "Complete results: " + f"phase-{phase}/search.json and trials.csv.",
                  "", "```json", json.dumps(result["selection"], indent=2), "```", "",
                  "All trials are listed below; values are mean and sample standard deviation across folds.", "",
                  "| Candidate | Family | C | Class weight | Loss | Mean F1 | SD F1 | Eligible |", "|---|---|---:|---|---|---:|---:|---|"]
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(["candidate_id", "family", "C", "class_weight", "loss", "mean_f1_macro", "std_f1_macro", "eligible", "fit_seconds", "score_seconds"])
        for trial in sorted(result["trials"], key=lambda t: -t["mean_f1_macro"]):
            c = trial["candidate"]
            row = [trial["candidate_id"], c["family"], c["C"], c["class_weight"], c["loss"],
                   trial["mean_f1_macro"], trial["std_f1_macro"], trial["eligible"]]
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
            writer.writerow([*row, trial["fit_seconds"], trial["score_seconds"]])
        from gpt_classifier.data import atomic_text
        atomic_text(output / f"phase-{phase}/trials.csv", stream.getvalue())
    lines += ["", "## Calibration, ensemble comparisons and primary decision", "", "Pending subsequent stages; no optimized fitted/calibrated final model or frozen A/B/C decision exists yet.",
              "", "Predeclared calibration plan: " + plan["protocol"]["calibration_plan"],
              "", "Predeclared weight plan: " + plan["protocol"]["weight_plan"],
              "", "Predeclared decision: " + plan["protocol"]["primary_decision_before_test1"],
              "", "## Test1 paired evaluation and statistical limitations", "", "Pending until A/B/C are frozen. No test1 or test2 was read by this study. Historical test1 reference 49.5844% macro-F1 is context, not a newly measured score. Exact correspondence will be checked only after freezing. Class metrics, confusion matrices, paired differences/corrected/harmed counts, complementarity, group bootstrap intervals and descriptive McNemar remain pending.",
              "", *["- " + limitation for limitation in plan["protocol"]["limitations"]],
              "", "## Recommendation", "", "No new performance claim. Keep the existing 75% SVM / 25% NorBERTo ensemble and every historical artifact unchanged while the investigation is pending.", ""]
    from gpt_classifier.data import atomic_text
    atomic_text(output / "report.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/svm-research.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    search = commands.add_parser("search")
    search.add_argument("--phase", choices=("a", "b"), required=True)
    commands.add_parser("report")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.config, args.output_dir)
        else:
            with file_lock(args.output_dir / ".research.lock"):
                config, _ = validate_plan(args.config, args.output_dir)
                if args.command == "search":
                    data = load_data(config)
                    result = search_phase(data, config, args.output_dir, args.phase)
                    validate_plan(args.config, args.output_dir)
                    report(args.output_dir)
                    result = {"status": result["status"], "phase": args.phase, "selection": result["selection"]}
                else:
                    report(args.output_dir)
                    result = {"status": "report_updated"}
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
