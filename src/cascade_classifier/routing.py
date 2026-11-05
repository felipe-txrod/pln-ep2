"""Label-free uncertainty ranking; labels are used only to select a validation recipe."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from ensemble_classifier.soft_voting import QWEN_MODELS, probabilities
from gpt_classifier.data import LABELS, Text, digest


@dataclass(frozen=True)
class RoutingRule:
    confidence_weight: float = .75
    maximum_confidence: float = .60
    maximum_margin: float = .15

    def __post_init__(self):
        for value in asdict(self).values():
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Routing parameters must be finite fractions.")


def rank_targets(records: list[Text], stacking, members, rule: RoutingRule, *, native_labels=None) -> list[dict]:
    if any(type(record) is not Text for record in records):
        raise TypeError("Routing accepts Text targets only, never gold labels.")
    p = probabilities(stacking)
    if len(p) != len(records) or set(members) != set(QWEN_MODELS):
        raise ValueError("Routing requires aligned stacking and all four members.")
    base = [probabilities(members[name]) for name in QWEN_MODELS]
    if any(part.shape != p.shape for part in base) or len({r.excel_row for r in records}) != len(records):
        raise ValueError("Duplicate rows or misaligned base predictions.")
    ordered = np.sort(p, axis=1)
    margin, maximum = ordered[:, -1] - ordered[:, -2], ordered[:, -1]
    entropy = -np.sum(p * np.log(np.clip(p, 1e-12, 1)), axis=1) / np.log(3)
    confidence = (entropy + 1 - margin + (1 - maximum) * 1.5) / 3
    if native_labels is None:
        votes = np.stack([part.argmax(axis=1) for part in base])
    else:
        if (set(native_labels) != set(QWEN_MODELS)
                or any(len(native_labels[name]) != len(records) or not set(native_labels[name]) <= set(LABELS)
                       for name in QWEN_MODELS)):
            raise ValueError("Native predictions are not aligned with all four members.")
        votes = np.asarray([[LABELS.index(label) for label in native_labels[name]] for name in QWEN_MODELS])
    disagreement = sum(votes[i] != votes[j] for i in range(4) for j in range(i + 1, 4)) / 6
    score = rule.confidence_weight * confidence + (1 - rule.confidence_weight) * disagreement
    result = []
    for index, record in enumerate(records):
        result.append({"excel_row": record.excel_row, "text_sha256": digest(record.text),
                       "label": LABELS[int(p[index].argmax())], "probabilities": dict(zip(LABELS, p[index].tolist())),
                       "maximum_probability": float(maximum[index]), "margin": float(margin[index]),
                       "entropy": float(entropy[index]), "disagreement": float(disagreement[index]),
                       "uncertainty": float(score[index]),
                       "eligible": bool(maximum[index] <= rule.maximum_confidence
                                        or margin[index] <= rule.maximum_margin or disagreement[index] > 0),
                       "local_predictions": {name: {"label": LABELS[int(votes[i][index])],
                             "probabilities_are_calibrated": True,
                             "probabilities": dict(zip(LABELS, base[i][index].tolist()))}
                             for i, name in enumerate(QWEN_MODELS)}})
    ranking = sorted(result, key=lambda row: (-row["uncertainty"], row["text_sha256"], row["excel_row"]))
    for rank, row in enumerate(ranking, 1):
        row["uncertainty_rank"] = rank
    return result


def select_targets(rows: list[dict], budget: float) -> list[dict]:
    if type(budget) not in (int, float) or not math.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError("Budget must be a finite fraction.")
    cap = math.floor(len(rows) * budget)
    return sorted((row for row in rows if row["eligible"]), key=lambda row: row["uncertainty_rank"])[:cap]


def choose_rule(records, stacking, members, labels, budget=.10, *, native_labels=None):
    """Maximize captured OOF stacking errors at the initial budget; fixed small search."""
    if len(labels) != len(records) or set(labels) != set(LABELS):
        raise ValueError("Invalid validation labels.")
    truth = {record.excel_row: label for record, label in zip(records, labels)}
    trials = []
    for weight in (.5, .75, .9):
        for confidence in (.5, .6):
            for margin in (.05, .15):
                rule = RoutingRule(weight, confidence, margin)
                routed = select_targets(rank_targets(records, stacking, members, rule, native_labels=native_labels), budget)
                trials.append({"rule": asdict(rule), "selected": len(routed),
                               "captured_errors": sum(row["label"] != truth[row["excel_row"]] for row in routed)})
    best = max(trials, key=lambda trial: (trial["captured_errors"], -trial["selected"],
                                        trial["rule"]["confidence_weight"], -trial["rule"]["maximum_confidence"],
                                        -trial["rule"]["maximum_margin"]))
    return RoutingRule(**best["rule"]), trials
