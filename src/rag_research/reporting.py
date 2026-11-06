"""Reproducible offline evidence report; never executes encoders or GPT calls."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ensemble_classifier.files import read_json
from gpt_classifier.data import canonical, file_digest, read_jsonl, write_json
from .data import load_data


def historical_case_review(data):
    """Descriptive validation cases; never feed these labels back into selection."""
    targets = {row.excel_row: row for row in data.targets}
    cases, used = [], set()
    confirmation = [(i, log) for i, log in sorted(data.logged.items()) if log["stage"] == "confirmation"]
    criteria = (
        ("harmed_c234", lambda i, arm: data.gold[i] == data.routing[i]["label"] == "c234" and arm["label"] != "c234"),
        ("corrected", lambda i, arm: data.gold[i] != data.routing[i]["label"] and arm["label"] == data.gold[i]),
        ("highest_cosine_label_mismatch", lambda i, arm: arm["retrieved"][0]["label"] != data.gold[i]),
    )
    for kind, eligible in criteria:
        candidates = [(i, log["arms"]["hybrid-3-local"]) for i, log in confirmation
                      if i not in used and eligible(i, log["arms"]["hybrid-3-local"])]
        if not candidates:
            continue
        if kind == "highest_cosine_label_mismatch":
            candidates.sort(key=lambda pair: -pair[1]["retrieved"][0]["semantic_cosine"])
        i, arm = candidates[0]
        first = arm["retrieved"][0]
        used.add(i)
        cases.append({"kind": kind, "excel_row": i, "gold": data.gold[i], "stacking": data.routing[i]["label"],
                      "gpt": arm["label"], "local_only_gpt": data.logged[i]["arms"]["local"]["label"],
                      "query_excerpt": targets[i].text[:180], "first_example_row": first["excel_row"],
                      "first_example_label": first["label"], "first_example_excerpt": first["resp_text"][:180],
                      "first_e5_cosine": first["semantic_cosine"], "first_lexical_cosine": first["lexical_cosine"],
                      "retrieved_labels": [row["label"] for row in arm["retrieved"]],
                      "base_predictions": {name: row["label"] for name, row in data.routing[i]["local_predictions"].items()}})
    return cases


def paired_retrieval_interval(data, directory, proposed, reference, seed=42, samples=1000):
    audit_ids = {i for i, fold in data.folds.items() if fold in {3, 4}}
    values = []
    for name in (proposed, reference):
        measurements = {}
        for row in read_jsonl(directory / (name + ".jsonl")):
            i = row["excel_row"]
            if i in audit_ids:
                labels = [example["label"] for example in row["selected"]]
                measurements[i] = np.array([bool(labels) and labels[0] == data.gold[i],
                                            labels.count(data.gold[i]) / max(len(labels), 1)], dtype=float)
        if set(measurements) != audit_ids:
            raise ValueError("Retrieval audit coverage is incomplete.")
        values.append(measurements)
    groups = sorted({data.groups[i] for i in audit_ids})
    group_index = {group: index for index, group in enumerate(groups)}
    differences, counts = np.zeros((len(groups), 2)), np.zeros(len(groups))
    for i in audit_ids:
        index = group_index[data.groups[i]]
        differences[index] += values[0][i] - values[1][i]
        counts[index] += 1
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(samples):
        weights = rng.multinomial(len(groups), np.ones(len(groups)) / len(groups))
        bootstrap.append((weights @ differences) / (weights @ counts))
    interval = np.quantile(bootstrap, [.025, .975], axis=0)
    estimate = differences.sum(0) / counts.sum()
    return {"audit_rows": len(audit_ids), "hit_at_1_difference": float(estimate[0]),
            "purity_at_3_difference": float(estimate[1]), "hit_at_1_95pct": interval[:, 0].tolist(),
            "purity_at_3_95pct": interval[:, 1].tolist(), "samples": samples,
            "note": "Paired group bootstrap on audit only for the development-selected proposal; not a GPT quality interval."}


def render(summary):
    r, a = summary["retrieval"], summary["interventions"]
    history = summary["historical_problem"]
    population = a["results"]["baseline"]["metrics"]["rows"]
    logged = a["support_queries"]
    selected, baseline = r["selected_variant"], "tfidf_e5--none--legacy"
    proposal = r["proposal_metrics"]
    lines = ["# Offline RAG and selective-GPT investigation", "",
             "[Machine-readable evidence](summary.json)", "",
             "This study made **zero new GPT calls**, did not use test1/test2 for optimization or configuration selection, and never retrained SVM, LR, NorBERTo, Qwen or stacking. Historical experiment packages and response caches were preserved. New small utility classifiers are research-only policies, not replacements for the trained base LR.", "",
             "## Verified historical problem", "",
             f"On the {history['queries']} confirmation queries, hybrid-3-local corrected {history['categories'].get('corrected', 0)} stacking errors and harmed {history['categories'].get('harmed', 0)} correct decisions. Of the harms, {history['harmed_c234']} had true class c234. Its first retrieved example matched the query label {history['first_hits']}/{history['queries']} times ({history['first_neighbor_hit_at_1']:.2%}). RAG changed {history['rag_changed_local_only_decisions']} GPT decisions versus local predictions alone: {history['rag_exclusive_correct']} exclusive correct decisions versus {history['local_only_exclusive_correct']} lost correct decisions.", "",
             f"Mean first-neighbor E5 cosine was {history['mean_first_e5_cosine']:.4f} despite these mismatches. High semantic similarity therefore did not establish rating agreement in this domain. Topic resemblance and perceived clarity are different signals; the latter explanation remains a hypothesis rather than a proven cause.", "",
             "| Descriptive validation case | Query row | Gold | Stacking | GPT | First training row / label | E5 cosine |", "|---|---:|---|---|---|---|---:|"]
    for case in summary["historical_case_review"]:
        lines.append(f"| {case['kind']} | {case['excel_row']} | {case['gold']} | {case['stacking']} | {case['gpt']} | {case['first_example_row']} / {case['first_example_label']} | {case['first_e5_cosine']:.4f} |")
    lines += ["", "These are retrospective error diagnostics, not development examples for a new prompt. Query/example excerpts, all four native classifier predictions, local-only GPT decisions and the three retrieved labels are included in the aggregate; full original traces remain local. A retrieved example's subjective clarity rating cannot be transferred to a target solely from topic or wording overlap. The original request and attachments are unavailable, so this review cannot establish why the human assessor assigned a particular score.", "",
             "The highest-cosine mismatch shares a long administrative attachment-forwarding template, yet has different gold labels; its lexical cosine remains below the duplicate exclusion threshold. The harmed c234 example has three base votes for c5 while stacking is correct, demonstrating why majority support alone cannot guarantee a safe override. In the corrected case, RAG changes local-only GPT from c234 to the correct c5 despite the first neighbor being c1. Retrieved label agreement is therefore insufficient to explain individual successes or failures causally.", "",
             "| Confirmation true class (analysis only) | Corrected | Harmed | Net correct change |", "|---|---:|---:|---:|"]
    for label, values in history["per_true_class"].items():
        lines.append(f"| {label} | {values['corrected']} | {values['harmed']} | {values['corrected'] - values['harmed']:+d} |")
    lines += ["", "The concentration of harm in c234 motivates diagnostic protection rules, but does not validate them. The inference-time protection rule uses the **predicted stacking class**, never the target's true class.", "",
             "## Frozen NorBERTo extraction", "",
             "The checkpoint is ModernBERT with 22 layers, hidden size 768 and LoRA rank 4 on Wqkv. The adapter is the unchanged EP1 artifact (`d693f6e2…`). The classifier was trained with masked mean pooling. We extracted mean and first-token representations at layers 11 and 22, in the same frozen FP32/SDPA forward pass, with the original cased 512-token head/tail preprocessing and L2 normalization.", "",
             "The tokenizer actually inserts [CLS] ID 3 and [SEP] ID 4. CLS features use the actual first token. The installed Transformers model exposes its last `hidden_states` entry before `final_norm`; the last-layer view uses `last_hidden_state` after that normalization, matching the classifier input. Intermediate layer 11 uses its own pre-final-normalization state.", "",
             f"Feature exports retain {r['encoder_export']['bank_rows']:,} eligible training examples and {r['encoder_export']['target_rows']:,} validation rows. Missing texts are encoded once for all requested views, and cached vectors are reused. Existing E5 vectors are also reused. The older NorBERTo-logistic cache contains **unadapted base-model** vectors and cannot represent this adapter; it was preserved rather than mislabeled or overwritten.", "",
             "## Retrieval comparison", "",
             f"There are 171 predeclared method/view/policy variants, plus random and length controls. Development uses original OOF folds 0/1/2 ({r['development_rows']} rows), and audit uses folds 3/4 ({r['audit_rows']} rows), with similarity groups disjoint. The proposal selected **only on development** is `{selected}`. The seven-method table fixes the NorBERTo view selected from NorBERTo-only development performance: `{r['comparison_view']}`. It holds the legacy MMR policy fixed.", "",
             "Hit@1 is first-neighbor class agreement. Class Recall@3 means at least one of the three examples has the true class; purity@3 is the fraction matching it. These use gold only after retrieval. Document-level label recall is also exported with the class-bank-size denominator. None is a measure of GPT macro-F1.", "",
             "| Method | Hit@1 | Class Recall@3 | Purity@3 | Unique classes |", "|---|---:|---:|---:|---:|"]
    for name, m in r["comparison_metrics"].items():
        lines.append(f"| {name} | {m['hit_at_1']:.2%} | {m['class_recall_at_3']:.2%} | {m['label_purity_at_3']:.2%} | {m['mean_unique_classes']:.3f} |")
    lines += ["", "These rows describe all 1,972 validation queries. Configuration choice uses development only; audit values for the chosen proposal are shown separately below.", "",
              "| Selected proposal split | Balanced Hit@1 | Balanced purity@3 | Class Recall@3 |", "|---|---:|---:|---:|"]
    for split in ("development", "audit"):
        m = proposal[split]
        lines.append(f"| {split} | {m['balanced_hit_at_1']:.2%} | {m['balanced_label_purity_at_3']:.2%} | {m['class_recall_at_3']:.2%} |")
    ci = r["paired_audit_interval"]
    lines += ["", f"Against the current TF-IDF+E5/legacy policy, the proposal's audit Hit@1 difference is {ci['hit_at_1_difference']:+.2%}; descriptive group-bootstrap 95% interval [{ci['hit_at_1_95pct'][0]:+.2%}, {ci['hit_at_1_95pct'][1]:+.2%}]. Purity@3 changes by {ci['purity_at_3_difference']:+.2%}. This interval describes retrieval label matching, not GPT improvement.", "",
              "## Pooling and layer ablation", "",
              "| NorBERTo view / policy | Development balanced Hit@1 | Audit balanced Hit@1 | Audit purity@3 |", "|---|---:|---:|---:|"]
    for name, metrics in r["pooling_metrics"].items():
        lines.append(f"| {name} | {metrics['development']['balanced_hit_at_1']:.2%} | {metrics['audit']['balanced_hit_at_1']:.2%} | {metrics['audit']['label_purity_at_3']:.2%} |")
    lines += ["", "| Encoder view | Random-pair cosine mean | Standard deviation | Mean-vector norm |", "|---|---:|---:|---:|"]
    for name, geometry in r["geometry"].items():
        lines.append(f"| {name} | {geometry['mean_random_pair_cosine']:.4f} | {geometry['std_random_pair_cosine']:.4f} | {geometry['mean_vector_norm']:.4f} |")
    lines += ["", "The four frozen views are not assumed to improve retrieval. High random-pair cosine and mean-vector norm indicate anisotropy: even unrelated training pairs can have high cosine. The wide layer11-mean spread further cautions against transferring an absolute cosine threshold between views. E5 uses its original query prefix and head truncation; NorBERTo uses its original head/tail policy, so cross-encoder differences include tokenization and truncation differences. Selecting layer22-CLS because its audit number looks higher would violate the development-only selection rule.", "",
              "## MMR and diversity ablation", "",
              "Legacy MMR has relevance weight 0.8, unseen-class bonus 0.04, maximum two examples per class and 120 candidates. All variants retain exact/near-duplicate exclusion, unique similarity groups and train-only examples. Fusion uses equal reciprocal ranks, not raw mixed score scales.", "",
              "The class bonus is identical for every candidate at the first selection, with no redundancy yet. It cannot explain the weak first neighbor by itself. The bonus/cap mainly affect the second and third examples. Candidate rank and cosine shifts show when diversity replaces a closer neighbor.", "",
              "| Current TF-IDF+E5 policy | Hit@1 | Recall@3 | Purity@3 | Unique classes | Mean selected rank |", "|---|---:|---:|---:|---:|---:|"]
    for name, m in r["legacy_ablation_metrics"].items():
        lines.append(f"| {name} | {m['hit_at_1']:.2%} | {m['class_recall_at_3']:.2%} | {m['label_purity_at_3']:.2%} | {m['mean_unique_classes']:.3f} | {m['mean_candidate_rank']:.2f} |")
    lines += ["", "| Dense retriever / policy | Recall@3 | Purity@3 | Unique classes | Mean selected rank |", "|---|---:|---:|---:|---:|"]
    for name, m in r["dense_ablation_metrics"].items():
        lines.append(f"| {name} | {m['class_recall_at_3']:.2%} | {m['label_purity_at_3']:.2%} | {m['mean_unique_classes']:.3f} | {m['mean_candidate_rank']:.2f} |")
    lines += ["", "For the E5 and selected NorBERTo-only legacy arms, every query retrieves all three classes. This makes Recall@3 exactly 100% and purity exactly one third; it is a consequence of forced diversity, not successful identification of the target class. Removing the cap alone does not fix it; removing the class bonus does. Dense cosine scores are tightly concentrated, so a fixed 0.04 bonus can dominate small relevance differences. Reciprocal-rank fusion changes the score spread, explaining why the same policy behaves differently across dense and fused arms.", "",
              "For the current hybrid, removing both bonus and cap raises purity while lowering class coverage; expanding the pool from 40 to 240 produces no aggregate benefit here. Removing redundancy alone does not solve the mismatch. Neither homogeneous nor class-diverse demonstrations are presumed better for GPT. These are retrieval tradeoffs, not confirmed GPT gains.", "",
              "## Per-class retrieval diagnostics", "",
              "| Method | True class | Hit@1 | Class Recall@3 | Purity@3 |", "|---|---|---:|---:|---:|"]
    for name, m in r["comparison_metrics"].items():
        for label, values in m["per_class"].items():
            lines.append(f"| {name} | {label} | {values['hit_at_1']:.2%} | {values['class_recall_at_3']:.2%} | {values['label_purity_at_3']:.2%} |")
    lines += ["", "| Method | Retrieved c1 | Retrieved c234 | Retrieved c5 | Mean lexical cosine | Mean E5 cosine | Mean NorBERTo cosine |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in r["comparison_metrics"].items():
        proportions = m["retrieved_label_distribution"]
        cosines = ["n/a" if m[key] is None else f"{m[key]:.4f}" for key in
                   ("mean_lexical_cosine", "mean_semantic_cosine", "mean_norberto_cosine")]
        lines.append(f"| {name} | {proportions['c1']:.2%} | {proportions['c234']:.2%} | {proportions['c5']:.2%} | " + " | ".join(cosines) + " |")
    lines += ["", "Per-class similarity results are preserved in the machine-readable aggregate and full local retrieval report. Similarities are reported in their own encoder space; they are not comparable confidence probabilities.", "",
              "## Routing versus acceptance", "",
              "Pre-call models see calibrated stacking/base probabilities, native class predictions, margin, entropy, disagreement and optionally the original retrieved-example statistics. Post-call models can additionally see the returned GPT class and its agreement with stacking, base votes and examples. Query gold never enters either feature function. New NorBERTo retrieval contexts are **not** paired with old GPT answers as if the API had seen them.", "",
              "The utility model estimates P(corrected)-P(harmed). It is a small regularized logistic model fitted inside each training fold. Five original group-disjoint outer folds assess the added policy conditionally on fixed historical upstream predictions; three inner group folds choose family, regularization, threshold and routing budget. The joint selector includes the family search, avoiding selection of a family on the same OOF scores used to claim its gain.", "",
              "This is not a fresh end-to-end nested evaluation: a policy-training row's historical OOF stacking predictor may have been fitted using labels from a policy-evaluation fold. Own-group meta predictions remain OOF, and all newly fitted policy transforms/models exclude evaluation groups, but the upstream OOF matrix is frozen rather than regenerated for each policy split. Historical Qwen exposure and prior router development also remain. These restrictions prevent claiming an unbiased new cascade generalization estimate from this replay, even for a positive family score.", "",
              f"Only {logged} historical queries have observed replies. Routing replay is restricted to this support; the other {population - logged} targets keep stacking because the replay policy never queries them. Effects of a new router outside this support cannot be identified without new GPT outcomes. Acceptance-only references use all logged calls ({logged/population:.2%} of this retrospective population), not a deployable 10% budget guarantee. Routed policies have per-fold 5/10% caps and include zero routing.", "",
              f"| Grouped OOF policy | Correct /{population} | Accuracy | Macro-F1 | Corrected | Harmed | Requested targets in replay |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in a["results"].items():
        m = value["metrics"]
        lines.append(f"| {name} | {m['correct']} | {m['accuracy']:.2%} | {m['f1_macro']:.2%} | {value['corrected']} | {value['harmed']} | {value['requested_targets_in_replay']} |")
    interval = a["joint_selection_bootstrap"]["macro_f1_difference_95pct"]
    lines += ["", f"The joint selector's descriptive 95% group-bootstrap interval for macro-F1 difference is [{interval[0]:+.2%}, {interval[1]:+.2%}]. It does not establish a benefit. Individual family improvements are exploratory; retrospective hypothesis generation on these historical results is an additional source of optimism.", "",
              "## Fixed-rule controls and safe pre-call skips", "",
              "| Fixed rule | Correct | Macro-F1 | Corrected | Harmed | Equivalent requested targets | Targets safely skipped |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in a["reference_policies_descriptive_only"].items():
        if "safe_pre_call_skips" not in value:
            continue
        lines.append(f"| {name} | {value['metrics']['correct']} | {value['metrics']['f1_macro']:.2%} | {value['corrected']} | {value['harmed']} | {value['equivalent_pre_call_requested_targets']} | {value['safe_pre_call_skips']} |")
    lines += ["", "Fixed-rule results are descriptive and were not selected on a fresh test set. A pre-call skip is safe relative to its corresponding post-call rule when **none of the three valid GPT labels can both pass the rule and change stacking**. Exhaustive synthetic tests and the logged replay verify prediction equivalence. Counts are target instances, not API batches; a future plan must rebatch them to measure call savings. This does not prove that the rule itself beats stacking.", "",
              "| Same-budget uncertainty reference | Correct | Macro-F1 | Requested targets |", "|---|---:|---:|---:|"]
    for name, value in a["reference_policies_descriptive_only"].items():
        if name.startswith("uncertainty_"):
            lines.append(f"| {name} | {value['metrics']['correct']} | {value['metrics']['f1_macro']:.2%} | {value['requested_targets']} |")
    h = a["pilot_trained_confirmation_holdout"]
    lines += ["", f"A separate chronological check chose the policy only on {h['training_queries']} pilot queries and evaluated the remaining {h['evaluation_rows']} confirmation rows. It selected `{h['selected_using_pilot_only']['rule'] or h['selected_using_pilot_only']['phase']}`: {h['metrics']['correct']} correct, macro-F1 {h['metrics']['f1_macro']:.2%}, versus stacking {h['baseline']['correct']} and {h['baseline']['f1_macro']:.2%}.", "",
              "## Screening for any later paid evaluation", "",
              f"Development-selected retrieval proposal: `{selected}`. Offline screening result: **{'eligible for a limited future matched trial' if r['paid_trial_screen_passed'] else 'does not pass the predeclared paid-trial screen'}**. Screening requires a positive lower audit Hit@1 difference interval and no mean purity decline versus the current hybrid. Class coverage/diversity/relevance remain tradeoffs requiring review. No paid call is authorized or executed by this report.", "",
              f"Learned intervention positive-lower-bound screen: **{'passed' if interval[0] > 0 else 'not passed'}**. Do not promote a rule because it looks best retrospectively, or add more paid rounds to rescue an unconfirmed result. No production default is changed by this report. If a retrieval candidate passes the auxiliary screen, a small future matched trial must freeze prompts/routing first, retain a text+local control, and compare pure relevance with constrained MMR before claiming GPT quality improvement.", "",
              "## Decisions from the measured evidence", "",
              "- Retain the historical default unless a separately authorized validation establishes a better cascade. Fixed-upstream replay cannot promote a new default.",
              f"- Keep NorBERTo retrieval as an optional offline research component. The development-selected proposal {'passes only the auxiliary retrieval screen, not a GPT quality test' if r['paid_trial_screen_passed'] else 'fails the paid-trial screen and does not earn promotion'}.",
              "- Candidate-pool expansion and learned acceptance need demonstrated benefit before adding their complexity to the active pipeline.",
              "- Keep the bonus/cap/redundancy switches and safe pre-call equivalence checks for reproducible diagnostics. Diversity's measurable effect does not establish a better GPT prompt.",
              f"- {'Only the development-selected retrieval proposal merits consideration for a limited matched trial; its offline result does not demonstrate GPT improvement.' if r['paid_trial_screen_passed'] else 'No new retrieval configuration currently justifies a paid trial under the recorded evidence criteria.'} A later authorized study would need fresh grouped validation, frozen routing/prompts, a local-only control and explicit cost limits. Do not pick a new winner by reusing this audit.", "",
              "## Reproducibility and boundaries", "",
              f"Legacy retrieval trace reproduction: {r['legacy_trace_reproduction']['matched']}/{r['legacy_trace_reproduction']['total']} exact example-ID sequences. The raw output contains 171 complete trace files, baseline traces, a source snapshot and hashes; model weights, cache namespaces, IDs and folds are pinned. All embedding arrays are finite/unit-normalized and caches are checksummed. Historical checkpoints, experiments and GPT responses were not overwritten.", "",
              "Research token usage, new GPT cost and remote latency are all zero. Local computation time is exported separately; replayed requested-target counts do not represent new calls or a measured reduction in billed API batches.", "",
              "No test1/test2 optimization or final prediction was performed. Nested meta-model OOF prevents own-group meta fitting, subject to the fixed-upstream limitation above. No original base classifier or stacking was refitted. Future GPT evaluation must use newly valid cache keys for changed contexts; old replies cannot demonstrate improvements from new demonstrations.", ""]
    return "\n".join(lines)


def run(config, retrieval: Path, interventions: Path, output: Path):
    if output.exists():
        raise FileExistsError("Use a new report directory.")
    report = read_json(retrieval / "retrieval-report.json")
    acceptance = read_json(interventions / "acceptance-report.json")
    protocol = read_json(retrieval / "protocol.json")
    intervention_protocol = read_json(interventions / "protocol.json")
    if any(canonical(config.to_dict()) != canonical(value["config"]) for value in (protocol, intervention_protocol)):
        raise ValueError("Use the recorded offline configuration when consolidating reports.")
    if report["status"] != "completed" or acceptance["status"] != "completed":
        raise ValueError("Complete both offline studies before reporting.")
    if (file_digest(retrieval / "protocol.json") != report["protocol_sha256"]
            or file_digest(interventions / "protocol.json") != acceptance["protocol_sha256"]):
        raise ValueError("An offline protocol changed after evaluation.")
    for name, expected in report["files"].items():
        if file_digest(retrieval / name) != expected:
            raise ValueError("Retrieval evidence changed after evaluation.")
    for name, expected in acceptance["files"].items():
        if file_digest(interventions / name) != expected:
            raise ValueError("Intervention evidence changed after evaluation.")
    data = load_data(config)
    if protocol["source"] != data.identities or intervention_protocol["source"] != data.identities:
        raise ValueError("Offline experiments use different historical sources.")
    proposal = report["development_selected_variant"]
    views = protocol["norberto"]["views"]
    view = max(views, key=lambda v: (report["metrics"][f"norberto--{v}--legacy"]["development"]["balanced_hit_at_1"],
                                   report["metrics"][f"norberto--{v}--legacy"]["development"]["balanced_label_purity_at_3"]))
    comparison = {}
    for method in ("tfidf", "e5", "norberto", "tfidf_e5", "tfidf_norberto", "e5_norberto", "tfidf_e5_norberto"):
        key = method + "--" + (view if "norberto" in method else "none") + "--legacy"
        comparison[method] = report["metrics"][key]["all"]
    for name in ("random", "length"):
        comparison[name] = report["metrics"][name]["all"]
    interval = paired_retrieval_interval(data, retrieval, proposal, "tfidf_e5--none--legacy", config.seed)
    subset = {"selected_variant": proposal, "proposal_metrics": report["metrics"][proposal],
              "reference_metrics": report["metrics"]["tfidf_e5--none--legacy"],
              "comparison_view": view, "comparison_metrics": comparison,
              "pooling_metrics": {f"{v}/{p}": report["metrics"][f"norberto--{v}--{p}"] for v in views for p in ("legacy", "relevance")},
              "legacy_ablation_metrics": {p["policy"]["name"]: report["metrics"][p["name"]]["all"] for p in protocol["specifications"] if p["method"] == "tfidf_e5"},
              "dense_ablation_metrics": {p["name"]: report["metrics"][p["name"]]["all"] for p in protocol["specifications"]
                                         if (p["method"] == "e5" or p["method"] == "norberto" and p["view"] == view)
                                         and p["policy"]["name"] in {"legacy", "no_bonus", "no_cap", "no_bonus_or_cap", "relevance", "no_redundancy"}},
              "geometry": report["geometry"], "development_rows": report["development_rows"], "audit_rows": report["audit_rows"],
              "encoder_export": protocol["norberto"],
              "paired_audit_interval": interval, "legacy_trace_reproduction": report["legacy_trace_reproduction"],
              "elapsed_seconds": report["elapsed_seconds"], "paid_trial_screen_passed": bool(interval["hit_at_1_95pct"][0] > 0 and interval["purity_at_3_difference"] >= 0)}
    compact_acceptance = {key: acceptance[key] for key in ("results", "joint_selection_bootstrap", "reference_policies_descriptive_only",
                         "pilot_trained_confirmation_holdout", "fits", "support_queries", "unqueried_counterfactuals_unavailable", "limitations")}
    history = dict(acceptance["historical_analysis"]["by_stage"]["confirmation"])
    confirmation = [(i, log) for i, log in data.logged.items() if log["stage"] == "confirmation"]
    history["first_hits"] = round(history["first_neighbor_hit_at_1"] * history["queries"])
    history["harmed_c234"] = sum(data.gold[i] == "c234" and data.routing[i]["label"] == "c234" and log["arms"]["hybrid-3-local"]["label"] != "c234" for i, log in confirmation)
    history["mean_first_e5_cosine"] = float(np.mean([log["arms"]["hybrid-3-local"]["retrieved"][0]["semantic_cosine"] for _, log in confirmation]))
    history["per_true_class"] = {label: {
        "corrected": sum(data.gold[i] == label and log["arms"]["hybrid-3-local"]["label"] == label and data.routing[i]["label"] != label for i, log in confirmation),
        "harmed": sum(data.gold[i] == label and log["arms"]["hybrid-3-local"]["label"] != label and data.routing[i]["label"] == label for i, log in confirmation)}
        for label in ("c1", "c234", "c5")}
    summary = {"status": "offline_only", "config": config.to_dict(), "retrieval": subset, "interventions": compact_acceptance,
               "historical_problem": history, "historical_case_review": historical_case_review(data),
               "source": data.identities, "new_gpt_calls": 0, "base_model_training_updates": 0,
               "test1_accessed": False, "test2_accessed": False, "production_default_changed": False,
               "reporter_sha256": file_digest(Path(__file__)),
               "retrieval_report_sha256": file_digest(retrieval / "retrieval-report.json"),
               "acceptance_report_sha256": file_digest(interventions / "acceptance-report.json")}
    output.mkdir(parents=True)
    write_json(output / "summary.json", summary)
    (output / "report.md").write_text(render(summary))
    return {"status": "reported", "report": str(output / "report.md"), "new_gpt_calls": 0,
            "retrieval_candidate": proposal, "paid_trial_screen_passed": subset["paid_trial_screen_passed"]}
