"""Combine labels without assuming calibrated probabilities."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

LABELS = ("c1", "c234", "c5")


def local_decision(labels: Sequence[str], strategy: str = "consensus") -> str | None:
    """Return None when the selected arbitrator must provide the final label."""
    if not labels or any(label not in LABELS for label in labels):
        raise ValueError("Every member must provide a predefined class label.")
    counts = Counter(labels)
    if strategy in {"consensus", "consensus_gpt"}:
        return labels[0] if len(counts) == 1 else None
    if strategy in {"majority", "majority_gpt"}:
        label, votes = counts.most_common(1)[0]
        return label if votes > len(labels) / 2 else None
    raise ValueError("Unknown ensemble strategy.")
