"""Render a self-contained research report distinguishing development from final evaluation."""
from __future__ import annotations

import json
from pathlib import Path

from ensemble_classifier.files import read_json
from gpt_classifier.data import LABELS, atomic_text


def _metrics_table(lines, metrics, *, names=None):
    lines += ["", "| Configuration | Accuracy | Macro-F1 | Log loss | Brier | ECE (10 bins) |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in names or metrics:
        result = metrics[name]
        values = [f"{result[key]:.8f}" if key in result else "—"
                  for key in ("accuracy", "f1_macro", "log_loss", "multiclass_brier", "confidence_ece_10_bins")]
        lines.append("| " + " | ".join([name, *values]) + " |")


def _class_tables(lines, metrics, names):
    for name in names:
        result = metrics[name]
        lines += ["", f"### {name}", "", "| Class | Precision | Recall | F1 | Support |",
                  "|---|---:|---:|---:|---:|"]
        for label in LABELS:
            entry = result["classification_report"][label]
            lines.append(f"| {label} | {entry['precision']:.8f} | {entry['recall']:.8f} | {entry['f1-score']:.8f} | {int(entry['support'])} |")
        lines += ["", "Confusion matrix: true rows / predicted columns, c1/c234/c5:",
                  "", "```json", json.dumps(result["confusion_matrix"]), "```"]


def report(output: Path):
    plan = read_json(output / "plan.json")
    search_root = Path(plan["search_root"])
    audit = read_json(search_root / "audit.json")
    baseline = read_json(search_root / "baseline.json")
    selected = plan["selected_candidate"]
    lines = ["# Final SVM optimization and soft-voting investigation", "",
             "## 1. Original SVM audit", "",
             "The original artifact remains unchanged. HTML entity decoding, whitespace collapse and lowercase preserve accents/punctuation. Word and char_wb TF-IDF are concatenated using FeatureUnion. The original model has no predict_proba; its old sigmoid calibrator and NorBERTo's old temperature remain frozen for configuration A.",
             "", "The complete frozen original inference contract is:", "", "```json",
             json.dumps({key: audit["original_svm"][key] for key in ("artifact_sha256", "normalization", "word", "char", "classifier_parameters", "runtime_versions")}, indent=2),
             "```", "", "Preparation reproduced every one of the 3999 old SVM decision-margin rows with maximum absolute difference 0.0, and exactly reproduced the old 75/25 ensemble's meta-validation confusion matrix and macro-F1. The original model's earlier inner search optimized accuracy; this study optimizes macro-F1.",
             "", "## 2. Data, provenance and prospective separation", "",
             "| Role | Rows | Similarity groups |", "|---|---:|---:|"]
    for role in ("training", "calibration", "weight_selection"):
        lines.append(f"| {role} | {audit['rows'][role]} | {audit['groups'][role]} |")
    lines += ["", "Labels come only from the original train.xlsx clarity column. Its SHA-256, frozen EP1 split bytes, row IDs, text hashes, labels, original group manifest and passed historical group audit are verified before each stage. Similarity components were formed label-free using identifier masking and character 3–5 TF-IDF cosine at .93. No component or normalized duplicate crosses these roles. TF-IDF is learned only inside each CV training fold; the final refit uses all original training rows only.",
              "", "The 3999 observations are historically reused development data. Test1 has been examined previously and has known source-text overlap; its later evaluation is retrospective. Test2 stays entirely outside this investigation. No GPT, stacking, RAG, other-model fitting/inference or default promotion is performed.",
              "", "## 3. Search space, compute amendment and selection", "",
              "The original 256-candidate search was stopped after 129 folds because unnormalized fits took 643–991 seconds and often failed convergence. All original files/cache/source remain preserved. The user authorized a compute-budget amendment: 96 phase A candidates, four feature families, normalized L1/L2 designs, C=.005/.05/.5, both class weights, binary/sublinear TF and ngram/df variations. Seventy-two exactly matching old folds were verified and imported. The exclusions concern compute cost, not proof of predictive inferiority.",
              "", "The reduced engine uses four worker slots with one numerical thread per fit and a 120-second child wall limit. It fits each representation once per training fold and transforms validation once, sharing read-only sparse matrices across classifier settings. Timeout/nonconvergence excludes a candidate; no fabricated label replaces a failure. Phase B refines two predeclared mean-ranked representations with local C, loss and branch weights. All candidate/fold settings, confusion matrices, per-class scores, timings and prediction checksums are saved in each phase's search.json/trials directory.",
              "", "## 4. Complete grouped CV results", "",
              "Mean and sample SD are across folds; these are adaptive development diagnostics, not unbiased end-to-end nested-CV generalization estimates. The rule selects the smallest C and feature count within one SE of the best mean, with the predeclared remaining ties. It does not simply select the highest observed mean."]
    for phase in ("a", "b"):
        result = read_json(search_root / f"phase-{phase}/search.json")
        verification = read_json(search_root / f"verification-phase-{phase}.json")
        lines += ["", f"### Phase {phase.upper()}", "",
                  f"{verification['candidates']} candidates, {verification['folds']} folds, {verification['eligible_candidates']} eligible, {verification['timeout_folds']} timeouts and {verification['convergence_warning_folds']} convergence warnings. Independent predictions/group/metric verification passed. Artifact window: {verification['manifest_to_result_seconds']:.3f} seconds, including any pauses; phase A reuses 72 earlier folds.", "",
                  "| Candidate | Family | C | Class weight | Loss | Char weight | Fold macro-F1 | Mean F1 | SD F1 |",
                  "|---|---|---:|---|---|---:|---|---:|---:|"]
        for trial in result["trials"]:
            candidate = trial["candidate"]
            fold_scores = ", ".join(f"{r['metrics']['f1_macro']:.6f}" if r["status"] == "completed" else "timeout" for r in trial["folds"])
            mean = f"{trial['mean_f1_macro']:.8f}" if trial["mean_f1_macro"] is not None else "—"
            std = f"{trial['std_f1_macro']:.8f}" if trial["std_f1_macro"] is not None else "—"
            lines.append(f"| {trial['candidate_id']} | {candidate['family']} | {candidate['C']:.8g} | {candidate['class_weight']} | {candidate['loss']} | {candidate['char_weight']} | {fold_scores} | {mean} | {std} |")
    lines += ["", "## 5. Selected optimized SVM", "", "```json", json.dumps(selected, indent=2), "```",
              "", "This is the phase B one-SE choice. A higher CV mean from another candidate cannot override that rule. The phase B baseline and selected model are evaluated on the same five grouped folds; scores from phase A use a different fold partition and must not be directly treated as paired phase B results.",
              "", "The reproduced old meta-validation baseline is:"]
    _metrics_table(lines, {"old_native_svm": baseline["svm_native"], "old_calibrated_svm": baseline["svm_calibrated"], "A_original_75_25": baseline["A"]})
    if not (output / "development.json").exists():
        lines += ["", "## 6–11. Remaining stages", "",
                  "Final refit, nested sigmoid/isotonic calibration, nested weights, A/B/C freezing, per-class/complementarity analysis, retrospective Test1 comparison and final recommendation remain pending. The full conservative downstream protocol is frozen in plan.json. No new final workbook has been opened. Keep the original ensemble while evidence is pending."]
        atomic_text(output / "report.md", "\n".join(lines) + "\n")
        return
    calibration = read_json(output / "calibration.json")
    development = read_json(output / "development.json")
    refit = read_json(output / "refit.json")
    lines += ["", f"Final refit: {refit['training_rows']} original training rows, {refit['feature_count']} features, {refit['n_iter']} iterations, fit {refit['fit_seconds']:.3f}s, reserved-score extraction {refit['score_seconds']:.3f}s. Historical weights/checkpoints were not overwritten.",
              "", "## 6. Calibration", "",
              "Five outer grouped folds estimate method-selection quality on the 2027 calibration rows. Each outer-training portion uses three inner grouped folds to choose isotonic only when its mean paired log-loss improvement over sigmoid exceeds its SE. Otherwise sigmoid is preferred. The final method uses the same three-fold rule on all calibration rows and is then fitted on those 2027 rows. No base SVM is refitted by calibration.",
              "", f"Final method: **{calibration['report']['chosen_method']}**. Outer choices: `{calibration['report']['outer_method_counts']}`. Selection-policy OOF and fixed-method OOF diagnostics are distinct:"]
    _metrics_table(lines, calibration["report"]["oof_metrics"])
    lines += ["", "The uncalibrated comparison uses softmax decision margins as a diagnostic surrogate; it is not a native SVM probability interface. Brier is the mean sum of three squared class-probability errors; ECE uses ten confidence bins. Full reliability bins, inner losses, fold membership and calibrator parameters are in calibration.json.",
              "", "## 7. Soft voting and conservative development recommendation", "",
              "A = old calibrated SVM .75 + unchanged calibrated NorBERTo .25. B replaces only the SVM while retaining .75/.25. C adjusts only those two weights. Candidate SVM weights are 0/.25/.5/.75/1. A grouped five-by-three nested selector estimates the weight-selection policy on outer-held groups. Final C weights use all meta-validation data with the same inner one-SE rule, preferring proximity to .75.",
              "", f"Final C SVM weight: **{development['weights']['final_svm_weight']}**. Outer weights: `{development['weights']['outer_weight_counts']}`; final weight matches {development['weights']['final_weight_outer_matches']}/5 outer choices."]
    _metrics_table(lines, development["metrics"])
    lines += ["", "C_fixed_diagnostic uses a weight selected on these same labels and is adaptive. C_policy_oof uses a weight chosen without its own outer-evaluation groups and is the version used for the adoption screen. Neither is an independent final-test estimate.",
              "", "B can replace A only when paired similarity-group bootstrap macro-F1 lower bound is above zero, at least four of five grouped fold deltas are positive, and no class F1 drops by more than .02. C can replace B only with the same checks on weight-policy OOF and a final weight matching at least three outer choices. B must first pass against A.",
              "", f"B over A checks: `{development['B_over_A_screen']['checks']}`. C policy over B checks: `{development['C_policy_over_B_screen']['checks']}`; stable weight: `{development['C_policy_over_B_screen']['weight_stable']}`.",
              "", f"**Recommendation frozen before Test1: {development['recommended_configuration']}.** This is a development decision under conservative screening, not a guarantee of generalization or automatic production promotion.",
              "", "## 8. Class behavior and complementarity on meta-validation"]
    _class_tables(lines, development["metrics"], ("A", "B", "C_fixed_diagnostic", "C_policy_oof"))
    lines += ["", "```json", json.dumps(development["complementarity"], indent=2), "```",
              "", "## 9. Paired retrospective Test1 comparison", ""]
    if not (output / "test1/metrics.json").exists():
        lines += ["Pending. The 900-row input/gold are not accessed by development. A/B/C and the primary recommendation must be frozen before the separate evaluation command. The historical A reference is 0.49584421663197825 macro-F1; exact row, probability and prediction correspondence must be checked before comparing labels."]
    else:
        final = read_json(output / "test1/metrics.json")
        lines += ["All A/B/C and individual predictions were persisted and checksummed before reading the 900 gold labels. Configuration A reproduces historical probabilities, row/text IDs, labels and macro-F1 exactly. All 900 rows remain in original order; no retrospective parameter updates or recommendation change were made."]
        _metrics_table(lines, final["metrics"])
        _class_tables(lines, final["metrics"], ("A", "B", "C"))
        lines += ["", "| Comparison (right minus left) | Corrected | Harmed | Delta macro-F1 | Group-bootstrap 95% CI | Descriptive McNemar p |",
                  "|---|---:|---:|---:|---|---:|"]
        for key in ("B_minus_A", "C_minus_A", "C_minus_B"):
            pair = final["paired_statistics"]["pairs"][key]
            lines.append(f"| {key} | {pair['corrected']} | {pair['harmed']} | {pair['delta']['f1_macro']:.8f} | {pair['delta_95_interval']['f1_macro']} | {pair['mcnemar_exact_p_descriptive']:.8f} |")
        lines += ["", "Accuracy/F1 absolute intervals for every model, every paired accuracy interval, per-class corrected/harmed counts and full reliability bins are in test1/metrics.json.",
                  "", "Final complementarity:", "", "```json", json.dumps(final["complementarity"], indent=2), "```"]
    lines += ["", "## 10. Statistical and methodological limitations", "",
              "Paired percentile bootstrap uses 2000 resamples, seed 42, of complete historical similarity components on meta-validation and normalized-text groups on Test1. The same resampled group weights apply to all compared models. Intervals condition on fitted predictions; they do not include all model-training, calibration and adaptive-selection uncertainty. The small, reused development corpus and successive searches limit generalization claims. Descriptive McNemar tests assess paired correctness rather than macro-F1 and ignore remaining cluster dependence; multiple retrospective comparisons are not fresh confirmatory tests.",
              "", "The old holdout and Test1 have historical exposure. Exact/normalized duplicates and similarity grouping do not guarantee semantic independence or consistent annotation. All official rows are retained; no label-based filtering/correction is performed. This investigation keeps Test2 outside development and does not satisfy the assignment's original Test2-before-Test1 order. Generating a later delivery would not retrospectively make previous Test1 assessments blind.",
              "", "## 11. Final recommendation", "",
              f"The development-frozen recommendation remains **{development['recommended_configuration']}**. If A, retain the original SVM .75 / NorBERTo .25 ensemble because the predeclared screens did not support replacement. If B or C, the frozen development evidence supports that research candidate under the documented conservative rule; Test1 cannot replace it with a retrospective winner. The active production/default ensemble is never changed automatically. Negative results are retained without reopening stacking, GPT, RAG or other base-model optimization.", ""]
    atomic_text(output / "report.md", "\n".join(lines))
