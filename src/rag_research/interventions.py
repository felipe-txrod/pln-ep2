"""Historical intervention diagnostics and inference-only benefit features."""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np

from gpt_classifier.data import LABELS


def intervention_utility(stacking, gpt, gold):
    return int(gpt == gold) - int(stacking == gold)


def pre_call_features(row, examples=(), *, include_retrieval=True):
    """Only predictions and training-example evidence; no target gold or GPT answer."""
    probabilities = row["probabilities"]
    if set(probabilities) != set(LABELS) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values()):
        raise ValueError("Invalid stacking probabilities.")
    features = {"stack_p_" + label: probabilities[label] for label in LABELS}
    features.update({"stack_class_" + label: float(row["label"] == label) for label in LABELS})
    for name in ("maximum_probability", "margin", "entropy", "disagreement"):
        value = row[name]
        if not math.isfinite(value):
            raise ValueError("Nonfinite uncertainty features.")
        features["stack_" + name] = value
    members = row["local_predictions"]
    votes = Counter(value["label"] for value in members.values())
    for label in LABELS:
        features["vote_fraction_" + label] = votes[label] / len(members)
    for name, value in sorted(members.items()):
        if value["label"] not in LABELS or set(value["probabilities"]) != set(LABELS):
            raise ValueError("Invalid native member evidence.")
        for label in LABELS:
            features[f"{name}_native_{label}"] = float(value["label"] == label)
            features[f"{name}_p_{label}"] = value["probabilities"][label]
    if include_retrieval:
        counts = Counter(example["label"] for example in examples)
        size = len(examples)
        features["retrieval_count"] = size
        features["retrieval_unique_classes"] = len(counts)
        for label in LABELS:
            features["retrieval_fraction_" + label] = counts[label] / max(size, 1)
            features["retrieval_first_" + label] = float(bool(examples) and examples[0]["label"] == label)
        features["retrieval_label_entropy"] = -sum((count / size) * math.log(count / size) for count in counts.values()) if size else 0.
        features["stack_retrieval_support"] = counts[row["label"]] / max(size, 1)
        for name in ("lexical_cosine", "semantic_cosine", "relevance"):
            values = [example[name] for example in examples if example.get(name) is not None]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Nonfinite retrieval evidence.")
            features["retrieval_" + name + "_first"] = values[0] if values else 0.
            features["retrieval_" + name + "_mean"] = float(np.mean(values)) if values else 0.
            features["retrieval_" + name + "_range"] = max(values) - min(values) if values else 0.
    return features


def post_call_features(row, gpt_label, examples=(), *, include_retrieval=True):
    if gpt_label not in LABELS:
        raise ValueError("An observed valid GPT label is required after the call.")
    features = pre_call_features(row, examples, include_retrieval=include_retrieval)
    for label in LABELS:
        features["gpt_class_" + label] = float(gpt_label == label)
    features["gpt_matches_stack"] = float(gpt_label == row["label"])
    features["gpt_local_vote_fraction"] = sum(value["label"] == gpt_label for value in row["local_predictions"].values()) / len(row["local_predictions"])
    features["gpt_stack_probability"] = row["probabilities"][gpt_label]
    features["gpt_stack_probability_gap"] = row["maximum_probability"] - row["probabilities"][gpt_label]
    if include_retrieval:
        features["gpt_retrieval_fraction"] = sum(example["label"] == gpt_label for example in examples) / max(len(examples), 1)
        features["gpt_matches_first_example"] = float(bool(examples) and examples[0]["label"] == gpt_label)
    return features


def acceptance_rule(name, row, gpt_label, examples):
    """Predeclared simple post-call rules; none sees the target's true class."""
    votes = sum(member["label"] == gpt_label for member in row["local_predictions"].values())
    support = sum(example["label"] == gpt_label for example in examples)
    if name == "protect_c234":
        return row["label"] != "c234" or gpt_label == "c234"
    if name == "two_base_votes":
        return votes >= 2
    if name == "three_base_votes":
        return votes >= 3
    if name == "two_examples":
        return support >= 2
    if name == "votes_and_example":
        return votes >= 2 and support >= 1
    if name == "protect_and_votes":
        return (row["label"] != "c234" or gpt_label == "c234") and votes >= 2
    raise ValueError("Unknown acceptance rule.")


def rule_can_change_prediction(name, row, examples):
    """Skip a call when no possible valid answer can pass the rule and change stacking.

    This equivalence uses only pre-call evidence and all three possible labels,
    never the actual GPT answer or the target's true class.
    """
    return any(label != row["label"] and acceptance_rule(name, row, label, examples) for label in LABELS)


def historical_analysis(data):
    events, buckets = [], defaultdict(Counter)
    stages = {}
    for identifier, log in data.logged.items():
        row = data.routing[identifier]
        context = log["arms"]["hybrid-3-local"]
        gpt, examples, gold = context["label"], context["retrieved"], data.gold[identifier]
        utility = intervention_utility(row["label"], gpt, gold)
        changed = row["label"] != gpt
        category = "corrected" if utility > 0 else "harmed" if utility < 0 else "both_wrong_changed" if changed else "unchanged"
        votes = sum(member["label"] == gpt for member in row["local_predictions"].values())
        support = sum(example["label"] == gpt for example in examples)
        event = {"excel_row": identifier, "group_id": data.groups[identifier], "outer_fold": data.folds[identifier],
                 "stage": log["stage"], "stacking": row["label"], "gpt": gpt, "gold_for_analysis_only": gold,
                 "category": category, "utility": utility, "base_votes_for_gpt": votes,
                 "examples_for_gpt": support, "first_example_label": examples[0]["label"] if examples else None,
                 "first_example_matches_gold_analysis_only": bool(examples) and examples[0]["label"] == gold,
                 "first_semantic_cosine": examples[0].get("semantic_cosine") if examples else None,
                 "first_lexical_cosine": examples[0].get("lexical_cosine") if examples else None,
                 "pre_call_features": pre_call_features(row, examples),
                 "post_call_features": post_call_features(row, gpt, examples)}
        events.append(event)
        for name, value in (("stage", log["stage"]), ("true_class_analysis_only", gold), ("stacking_class", row["label"]),
                            ("gpt_class", gpt), ("base_votes", str(votes)), ("example_support", str(support)),
                            ("first_example_agrees_gpt", str(bool(examples) and examples[0]["label"] == gpt))):
            buckets[name + ":" + value][category] += 1
    for stage in ("pilot", "confirmation"):
        selected = [event for event in events if event["stage"] == stage]
        identifiers = [event["excel_row"] for event in selected]
        differences = [i for i in identifiers if data.logged[i]["arms"]["local"]["label"] != data.logged[i]["arms"]["hybrid-3-local"]["label"]]
        stages[stage] = {"queries": len(identifiers),
                         "categories": dict(Counter(event["category"] for event in selected)),
                         "first_neighbor_hit_at_1": sum(event["first_example_matches_gold_analysis_only"] for event in selected) / len(selected),
                         "rag_changed_local_only_decisions": len(differences),
                         "rag_exclusive_correct": sum(data.logged[i]["arms"]["hybrid-3-local"]["label"] == data.gold[i] for i in differences),
                         "local_only_exclusive_correct": sum(data.logged[i]["arms"]["local"]["label"] == data.gold[i] for i in differences)}
    mismatches = sorted((event for event in events if not event["first_example_matches_gold_analysis_only"]),
                        key=lambda event: (-(event["first_semantic_cosine"] or 0), event["excel_row"]))
    return {"by_stage": stages, "buckets": {name: dict(value) for name, value in buckets.items()},
            "high_similarity_label_mismatches": [{key: event[key] for key in
                ("excel_row", "stage", "first_semantic_cosine", "first_lexical_cosine", "first_example_label", "gold_for_analysis_only", "category")} for event in mismatches[:30]],
            "features_never_include_query_gold": True}, events
