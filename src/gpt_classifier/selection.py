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
    """Train-only lexical/hybrid retrieval with class balance and diversity."""

    def __init__(self, records: list[Example], *, max_chars: int = 1600, embeddings=None,
                 semantic_weight: float = 0.5, length_weight: float = 0.0):
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
        self.embeddings = embeddings
        self.semantic_weight, self.length_weight = semantic_weight, length_weight
        self.semantic_matrix = None
        self.lengths = np.log1p([len(clean(r.text)) for r in self.records])
        if embeddings is not None:
            self.semantic_matrix = embeddings.encode([r.text for r in self.records])
            self.summary.update(semantic_encoder=embeddings.metadata, fusion="weighted_rrf_k60_per_class",
                                semantic_weight=semantic_weight, length_weight=length_weight)

    def prepare_targets(self, texts: list[str]):
        if self.embeddings is not None:
            self.embeddings.encode(texts)  # Batch once, then per-target reads reuse the cache.

    @staticmethod
    def _rank_scores(values):
        # Equal scores share a rank; zero lexical matches cannot bias fusion by row order.
        from scipy.stats import rankdata
        ranks = rankdata(-values, method="min")
        return 61.0 / (60.0 + ranks)

    def relevance(self, text: str, *, strategy: str, balanced: bool = True):
        if strategy == "representative":
            return self.centrality
        lexical = (self.matrix @ self.vectorizer.transform([clean(text)]).T).toarray().ravel()
        if strategy != "hybrid":
            return lexical
        if self.embeddings is None:
            raise ValueError("Estratégia hybrid exige encoder local.")
        semantic = self.semantic_matrix @ self.embeddings.encode([text])[0]
        scores = np.zeros(len(self.records))
        target_group = group(text)
        partitions = self.indices.values() if balanced else [np.arange(len(self.records))]
        for indices in partitions:
            eligible = indices[np.array([self.group_ids[i] != target_group for i in indices])]
            fused = ((1 - self.semantic_weight) * self._rank_scores(lexical[eligible])
                     + self.semantic_weight * self._rank_scores(semantic[eligible]))
            if self.length_weight:
                length_similarity = np.exp(-np.abs(self.lengths[eligible] - np.log1p(len(clean(text)))))
                fused = (1 - self.length_weight) * fused + self.length_weight * length_similarity
            scores[eligible] = fused
        return scores

    def select(self, text: str, *, strategy: str, per_class: int) -> list[Example]:
        if strategy == "zero":
            return []
        if strategy not in {"similar", "representative", "hybrid"} or per_class < 1:
            raise ValueError("Seleção inválida.")
        target_group = group(text)
        relevance = self.relevance(text, strategy=strategy)
        chosen = {label: [] for label in LABELS}
        for label, indices in self.indices.items():
            eligible = [int(i) for i in indices if self.group_ids[i] != target_group]
            for _ in range(per_class):
                if not eligible:
                    raise ValueError(f"Exemplos insuficientes para {label}, excluído o próprio alvo.")
                scores = relevance[eligible].copy()
                if chosen[label]:
                    redundant = (self.matrix[eligible] @ self.matrix[chosen[label]].T).toarray().max(axis=1)
                    if strategy == "hybrid":
                        semantic_redundant = (self.semantic_matrix[eligible] @ self.semantic_matrix[chosen[label]].T).max(axis=1)
                        redundant = (1 - self.semantic_weight) * redundant + self.semantic_weight * semantic_redundant
                    scores = 0.8 * scores - 0.2 * redundant
                best = eligible[int(np.argmax(scores))]  # Ties use stable Excel order.
                chosen[label].append(best)
                eligible.remove(best)
        # Interleave classes, avoiding a block of last-class demonstrations.
        return [self.records[chosen[label][k]] for k in range(per_class) for label in LABELS]


def make_selector(records, config, *, retrieval_cache=None):
    if config.strategy == "zero":
        return None
    embeddings = None
    if config.strategy == "hybrid":
        from .embeddings import DEFAULT_CACHE, LocalEmbeddings
        embeddings = LocalEmbeddings(DEFAULT_CACHE if retrieval_cache is None else retrieval_cache)
    return Selector(records, max_chars=config.max_example_chars, embeddings=embeddings,
                    semantic_weight=config.hybrid_semantic_weight, length_weight=config.hybrid_length_weight)
