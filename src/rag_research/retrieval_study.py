"""Grouped development/audit retrieval diagnostics, never a GPT quality evaluation."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import itertools
from pathlib import Path
import sys
import time

import numpy as np

from gpt_classifier.data import LABELS, canonical, digest, file_digest, write_json, write_jsonl
from gpt_classifier.embeddings import LocalEmbeddings
from ensemble_classifier.files import read_json
from .cli import eligible_selector
from .data import load_data
from .retrieval import ConfigurableRetriever, ablations


def measure_retrieval(observations, bank_counts):
    def summarize(items):
        if not items:
            return {"rows": 0}
        n = len(items)
        labels = Counter(label for row in items for label in row["labels"])
        total = sum(labels.values())
        result = {"rows": n, "hit_at_1": sum(row["hit1"] for row in items) / n,
                  "class_recall_at_3": sum(row["coverage3"] for row in items) / n,
                  "label_purity_at_3": sum(row["purity3"] for row in items) / n,
                  "example_recall_at_3": sum(row["matched"] / bank_counts[row["gold"]] for row in items) / n,
                  "mean_unique_classes": sum(len(set(row["labels"])) for row in items) / n,
                  "shortfalls": sum(len(row["labels"]) < 3 for row in items),
                  "retrieved_label_counts": dict(labels),
                  "retrieved_label_distribution": {label: labels[label] / max(total, 1) for label in LABELS}}
        for name in ("lexical_cosine", "semantic_cosine", "norberto_cosine", "candidate_rank"):
            values = [value for row in items for value in row[name] if value is not None]
            result["mean_" + name] = float(np.mean(values)) if values else None
        return result
    result = summarize(observations)
    result["per_class"] = {label: summarize([row for row in observations if row["gold"] == label]) for label in LABELS}
    for metric in ("hit_at_1", "class_recall_at_3", "label_purity_at_3"):
        available = [value[metric] for value in result["per_class"].values() if value["rows"]]
        result["balanced_" + metric] = float(np.mean(available)) if available else None
    return result


def observe(identifier, gold, examples):
    labels = [row["label"] for row in examples]
    matched = labels.count(gold)
    return {"excel_row": identifier, "gold": gold, "labels": labels,
            "hit1": bool(labels and labels[0] == gold), "coverage3": matched > 0,
            "purity3": matched / len(labels) if labels else 0., "matched": matched,
            **{name: [row.get(name) for row in examples] for name in
               ("lexical_cosine", "semantic_cosine", "norberto_cosine", "candidate_rank")}}


def load_vectors(directory, data, selector):
    metadata = read_json(directory / "embeddings.json")
    if file_digest(directory / "norberto-vectors.npz") != metadata["archive_sha256"] or metadata["source"] != data.identities:
        raise ValueError("Frozen feature provenance changed.")
    bank_hash = digest(canonical([(row.excel_row, digest(row.text)) for row in selector.records]))
    target_hash = digest(canonical([(row.excel_row, digest(row.text)) for row in data.targets]))
    if bank_hash != metadata["bank_sha256"] or target_hash != metadata["target_sha256"]:
        raise ValueError("Feature text/order provenance differs from the retrieval corpus.")
    with np.load(directory / "norberto-vectors.npz", allow_pickle=False) as archive:
        if archive["bank_ids"].tolist() != [row.excel_row for row in selector.records] or archive["target_ids"].tolist() != [row.excel_row for row in data.targets]:
            raise ValueError("Feature IDs differ from the verified source.")
        bank = {view: archive[view][:len(selector.records)].copy() for view in metadata["views"]}
        targets = {view: archive[view][len(selector.records):].copy() for view in metadata["views"]}
    return bank, targets, metadata


def run(config, embeddings: Path, output: Path):
    if output.exists():
        raise FileExistsError("Use a new offline retrieval directory; prior studies are immutable.")
    data = load_data(config)
    selector = eligible_selector(data)
    bank_vectors, query_vectors, norberto = load_vectors(embeddings, data, selector)
    e5 = LocalEmbeddings(Path(config.e5_cache))
    bank_vectors["e5"] = e5.encode([row.text for row in selector.records])
    query_vectors["e5"] = e5.encode([row.text for row in data.targets])
    index = ConfigurableRetriever(selector, data.bank_groups, data.targets, data.groups, bank_vectors, query_vectors)
    views = list(norberto["views"])
    policies = ablations()
    specs = []
    for method in index.METHODS:
        for view in views if "norberto" in index.METHODS[method] else [None]:
            for policy in policies:
                name = "--".join([method, view or "none", policy.name])
                specs.append((name, method, view, policy))
    output.mkdir(parents=True)
    protocol = {"config": config.to_dict(), "source": data.identities, "norberto": norberto,
                "implementation": {name: file_digest(Path(__file__).with_name(name)) for name in
                                   ("config.py", "data.py", "norberto_embeddings.py", "retrieval.py", "retrieval_study.py", "cli.py")},
                "e5": e5.metadata, "bank_counts": dict(Counter(row.label for row in selector.records)),
                "specifications": [{"name": n, "method": m, "view": v, "policy": asdict(p)} for n, m, v, p in specs],
                "development_outer_folds": [0, 1, 2], "audit_outer_folds": [3, 4],
                "selection": "Development balanced Hit@1, then balanced purity@3, then fewer constraints; audit is not used to tune.",
                "metric_note": "class_recall_at_3 means the true class appears at least once; example_recall_at_3 divides matches by that class's bank size.",
                "new_gpt_calls": 0, "base_training_updates": 0, "test_sets_accessed": False}
    write_json(output / "protocol.json", protocol)
    write_jsonl(output / "bank.jsonl", [{"excel_row": row.excel_row, "resp_text": row.text, "label": row.label,
                                       "group_id": data.bank_groups[row.excel_row]} for row in selector.records])
    results = defaultdict(list)
    files = {name: (output / (name + ".jsonl")).open("w") for name, *_ in specs}
    for baseline in ("random", "length"):
        files[baseline] = (output / (baseline + ".jsonl")).open("w")
    legacy_matches, legacy_total, start = 0, 0, time.monotonic()
    try:
        for position, target in enumerate(data.targets):
            for name, method, view, policy in specs:
                examples, diagnostics = index.retrieve(target, method=method, view=view, policy=policy)
                observed = observe(target.excel_row, data.gold[target.excel_row], examples)
                results[name].append(observed)
                compact = [{key: value for key, value in row.items() if key != "resp_text"} for row in examples]
                files[name].write(canonical({"excel_row": target.excel_row, "text_sha256": target.text_sha256,
                                            "selected": compact, "diagnostics": diagnostics}) + "\n")
                if method == "tfidf_e5" and policy.name == "legacy" and target.excel_row in data.logged:
                    old = data.logged[target.excel_row]["arms"]["hybrid-3-local"]["retrieved"]
                    legacy_total += 1
                    legacy_matches += [r["excel_row"] for r in examples] == [r["excel_row"] for r in old]
            for baseline in ("random", "length"):
                examples = index.baseline(target, method=baseline)
                results[baseline].append(observe(target.excel_row, data.gold[target.excel_row], examples))
                files[baseline].write(canonical({"excel_row": target.excel_row, "selected": examples}) + "\n")
            if position == 0 or (position + 1) % 50 == 0 or position + 1 == len(data.targets):
                print(f"Offline retrieval: {position + 1}/{len(data.targets)} queries; {len(specs)} fixed variants; {time.monotonic() - start:.1f}s", file=sys.stderr, flush=True)
    finally:
        for file in files.values():
            file.close()
    development = {i for i, fold in data.folds.items() if fold in {0, 1, 2}}
    audit = set(data.folds) - development
    confirmation_logged = {i for i, value in data.logged.items() if value["stage"] == "confirmation"}
    summaries = {}
    for name, observations in results.items():
        summaries[name] = {part: measure_retrieval([row for row in observations if row["excel_row"] in ids], protocol["bank_counts"])
                           for part, ids in (("all", set(data.folds)), ("development", development), ("audit", audit),
                                             ("historical_confirmation_queries", confirmation_logged))}
    winner = max((name for name, *_ in specs), key=lambda name: (
        summaries[name]["development"]["balanced_hit_at_1"],
        summaries[name]["development"]["balanced_label_purity_at_3"], name.endswith("--relevance")))
    geometry = {}
    rng = np.random.default_rng(config.seed)
    left, right = rng.integers(0, len(selector.records), size=(2, 2000))
    for view, values in bank_vectors.items():
        pair_cosines = (values[left] * values[right]).sum(1)
        geometry[view] = {"mean_random_pair_cosine": float(pair_cosines.mean()), "std_random_pair_cosine": float(pair_cosines.std()),
                          "mean_vector_norm": float(np.linalg.norm(values.mean(0)))}
    report = {"status": "completed", "protocol_sha256": file_digest(output / "protocol.json"),
              "queries": len(data.targets), "specifications": len(specs), "development_rows": len(development),
              "audit_rows": len(audit), "development_selected_variant": winner,
              "metrics": summaries, "geometry": geometry,
              "legacy_trace_reproduction": {"matched": legacy_matches, "total": legacy_total},
              "elapsed_seconds": time.monotonic() - start, "embedding_stats": e5.stats,
              "new_gpt_calls": 0, "classifier_quality_improvement_demonstrated": False,
              "files": {name + ".jsonl": file_digest(output / (name + ".jsonl")) for name in files},
              "limitations": ["Label matching is an auxiliary metric, not evidence of GPT macro-F1 improvement.",
                              "Historical Qwen checkpoint exposure and previously tuned OOF routing remain inherited.",
                              "Audit variants are reported descriptively; only development selects the proposal."]}
    write_json(output / "retrieval-report.json", report)
    return {key: value for key, value in report.items() if key not in {"metrics", "files"}}
