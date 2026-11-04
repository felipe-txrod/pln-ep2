from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from .data import LABELS, Example, clean, group


def development_split(records: list[Example], *, seed: int = 42, limit: int = 150):
    labels = np.array([r.label for r in records])
    groups = np.array([group(r.text) for r in records])
    for label in LABELS:
        if len(set(groups[labels == label])) < 5:
            raise ValueError("Desenvolvimento exige ao menos cinco grupos por classe.")
    train, valid = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                        .split(np.zeros(len(records)), labels, groups))
    if limit < len(valid):
        if limit < 3:
            raise ValueError("A amostra de desenvolvimento deve conter ao menos três registros.")
        # Exact proportional sample, retaining every class even when n-limit<3.
        by_class = [valid[labels[valid] == label] for label in LABELS]
        counts = np.array([len(indices) for indices in by_class])
        if np.any(counts == 0):
            raise ValueError("O fold de validação perdeu uma classe.")
        targets = counts / counts.sum() * limit
        quotas = np.maximum(1, np.floor(targets).astype(int))
        while quotas.sum() > limit:
            quotas[int(np.argmax(np.where(quotas > 1, quotas - targets, -np.inf)))] -= 1
        while quotas.sum() < limit:
            quotas[int(np.argmax(np.where(quotas < counts, targets - quotas, -np.inf)))] += 1
        rng = np.random.default_rng(seed)
        valid = np.concatenate([rng.permutation(indices)[:quota] for indices, quota in zip(by_class, quotas)])
    valid = sorted(valid)
    if set(groups[train]) & set(groups[valid]):
        raise ValueError("Sobreposição de grupos entre banco e desenvolvimento.")
    if set(labels[train]) != set(LABELS) or set(labels[valid]) != set(LABELS):
        raise ValueError("Partição perdeu uma classe.")
    return [records[i] for i in train], [records[i] for i in valid]


class Selector:
    """Train-only lexical retrieval with class balance and diversity penalty."""

    def __init__(self, records: list[Example], *, max_chars: int = 1600):
        groups = defaultdict(list)
        for r in records:
            if r.label not in LABELS:
                raise ValueError("Rótulo inválido no banco.")
            groups[group(r.text)].append(r)
        # Never turn annotation disagreements into contradictory demonstrations.
        self.records = [min(rs, key=lambda r: r.excel_row) for rs in groups.values()
                        if len({r.label for r in rs}) == 1 and clean(rs[0].text)
                        and len(clean(rs[0].text)) <= max_chars]
        self.records.sort(key=lambda r: r.excel_row)
        self.group_ids = [group(r.text) for r in self.records]
        self.summary = {"source_rows": len(records), "eligible_examples": len(self.records),
                        "conflicting_groups_excluded": sum(len({r.label for r in rs}) > 1 for rs in groups.values()),
                        "max_example_characters": max_chars}
        if set(r.label for r in self.records) != set(LABELS):
            raise ValueError("O banco elegível deve conter todas as classes; confira tamanho/conflitos.")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=40000,
                                          sublinear_tf=True, strip_accents=None,
                                          token_pattern=r"(?u)\b\w+\b", dtype=np.float64)
        self.matrix = self.vectorizer.fit_transform([clean(r.text) for r in self.records])
        self.indices = {label: np.array([i for i, r in enumerate(self.records) if r.label == label]) for label in LABELS}
        self.centrality = np.zeros(len(self.records))
        for indices in self.indices.values():
            center = np.asarray(self.matrix[indices].mean(axis=0)).ravel()
            center /= max(np.linalg.norm(center), 1e-12)
            self.centrality[indices] = np.asarray(self.matrix[indices] @ center).ravel()

    def select(self, text: str, *, strategy: str, per_class: int) -> list[Example]:
        if strategy == "zero":
            return []
        if strategy not in {"similar", "representative"} or per_class < 1:
            raise ValueError("Seleção inválida.")
        target_group = group(text)
        relevance = self.centrality if strategy == "representative" else (
            self.matrix @ self.vectorizer.transform([clean(text)]).T).toarray().ravel()
        chosen = {label: [] for label in LABELS}
        for label, indices in self.indices.items():
            eligible = [int(i) for i in indices if self.group_ids[i] != target_group]
            for _ in range(per_class):
                if not eligible:
                    raise ValueError(f"Exemplos insuficientes para {label}, excluído o próprio alvo.")
                scores = relevance[eligible].copy()
                if chosen[label]:
                    redundant = (self.matrix[eligible] @ self.matrix[chosen[label]].T).toarray().max(axis=1)
                    scores = 0.8 * scores - 0.2 * redundant
                best = eligible[int(np.argmax(scores))]  # Ties use stable Excel order.
                chosen[label].append(best)
                eligible.remove(best)
        # Interleave classes, avoiding a block of last-class demonstrations.
        return [self.records[chosen[label][k]] for k in range(per_class) for label in LABELS]
