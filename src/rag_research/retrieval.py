"""Configurable rank fusion and MMR without query labels or remote inference."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.stats import rankdata

from gpt_classifier.data import Text, clean, digest, group
from rag_cascade.retrieval import near_duplicate


@dataclass(frozen=True)
class SelectionPolicy:
    name: str = "legacy"
    relevance_weight: float = .8
    class_bonus: float = .04
    class_cap: int | None = 2
    candidate_pool: int = 120

    def __post_init__(self):
        if not 0 <= self.relevance_weight <= 1 or not 0 <= self.class_bonus <= 1:
            raise ValueError("Invalid MMR weights.")
        if self.candidate_pool < 3 or (self.class_cap is not None and self.class_cap < 1):
            raise ValueError("Invalid candidate pool or class cap.")


def ablations():
    return [SelectionPolicy(), SelectionPolicy("no_bonus", class_bonus=0),
            SelectionPolicy("no_cap", class_cap=None),
            SelectionPolicy("no_bonus_or_cap", class_bonus=0, class_cap=None),
            SelectionPolicy("relevance", 1., 0., None),
            SelectionPolicy("no_redundancy", 1.),
            SelectionPolicy("pool40", candidate_pool=40),
            SelectionPolicy("pool240", candidate_pool=240),
            SelectionPolicy("weight50", relevance_weight=.5)]


def fuse_ranks(scores: list[np.ndarray]) -> np.ndarray:
    """Equal-weight reciprocal-rank fusion; ties share their minimum rank."""
    if not scores or any(value.shape != scores[0].shape or not np.isfinite(value).all() for value in scores):
        raise ValueError("Fusion requires aligned finite score arrays.")
    return np.mean([61. / (60. + rankdata(-value, method="min")) for value in scores], axis=0)


class ConfigurableRetriever:
    """The retrieve(Text, method=..., count=3) contract matches the existing prompt builder.

    Matrices and query features are injected. A query's gold label is never accepted.
    A view denotes a frozen NorBERTo layer/pooling combination, not a refitted model.
    """

    METHODS = {"tfidf": ("tfidf",), "e5": ("e5",), "norberto": ("norberto",),
               "tfidf_e5": ("tfidf", "e5"), "tfidf_norberto": ("tfidf", "norberto"),
               "e5_norberto": ("e5", "norberto"), "tfidf_e5_norberto": ("tfidf", "e5", "norberto")}

    def __init__(self, selector, bank_groups, targets, query_groups, bank_vectors, query_vectors):
        self.selector, self.records, self.bank_groups = selector, selector.records, bank_groups
        self.targets = {target.excel_row: target for target in targets}
        self.query_groups = query_groups
        self.target_indices = {target.excel_row: index for index, target in enumerate(targets)}
        self.bank_vectors, self.query_vectors = bank_vectors, query_vectors
        if set(bank_vectors) != set(query_vectors):
            raise ValueError("Bank/query encoder views differ.")
        for name, vectors in bank_vectors.items():
            query = query_vectors[name]
            if vectors.ndim != 2 or query.shape != (len(targets), vectors.shape[1]) or len(vectors) != len(self.records):
                raise ValueError("Embedding matrices do not match source rows.")
            if (not np.isfinite(vectors).all() or not np.isfinite(query).all()
                    or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
                    or not np.allclose(np.linalg.norm(query, axis=1), 1, atol=1e-4)):
                raise ValueError("Retrieval requires finite unit embeddings.")
        self.normalized = [group(record.text) for record in self.records]
        self.normalized_array = np.asarray(self.normalized)
        self.group_array = np.asarray([bank_groups[record.excel_row] for record in self.records])
        self.labels = np.asarray([record.label for record in self.records])
        self.lengths = np.asarray([len(clean(record.text)) for record in self.records])
        self.lexical = (selector.matrix @ selector.vectorizer.transform([clean(target.text) for target in targets]).T).toarray().T
        self.cosines = {}
        self.policy = SelectionPolicy()
        self.view = "layer22_mean"
        self._contexts = {}

    def prepare_targets(self, targets):
        for target in targets:
            if type(target) is not Text or target.excel_row not in self.targets or self.targets[target.excel_row].text != target.text:
                raise ValueError("Prepare verified label-free query embeddings before retrieval.")

    def similarities(self, source):
        if source not in self.cosines:
            # GEMV preserves the historical E5 arithmetic and ranking tie behavior.
            self.cosines[source] = np.stack([self.bank_vectors[source] @ vector for vector in self.query_vectors[source]])
        return self.cosines[source]

    @lru_cache(maxsize=300000)
    def _query_duplicate(self, identifier, index):
        return near_duplicate(self.targets[identifier].text, self.records[index].text, threshold=.85)

    @lru_cache(maxsize=500000)
    def _pair_duplicate(self, left, right):
        return near_duplicate(self.records[left].text, self.records[right].text, threshold=.85)

    def context(self, target: Text, method: str, view: str | None):
        self.prepare_targets([target])
        if method not in self.METHODS:
            raise ValueError("Unknown retrieval method.")
        key = (target.excel_row, method, view if "norberto" in self.METHODS[method] else None)
        if key in self._contexts:
            return self._contexts[key]
        position = self.target_indices[target.excel_row]
        sources = self.METHODS[method]
        if "norberto" in sources and view not in self.bank_vectors:
            raise ValueError("A prepared NorBERTo view is required.")
        lexical = self.lexical[position]
        values = {"tfidf": lexical}
        for source in sources:
            if source != "tfidf":
                values[source] = self.similarities("e5" if source == "e5" else view)[position]
        allowed = (self.normalized_array != group(target.text)) & (lexical < .95)
        component = self.query_groups.get(target.excel_row)
        if component is not None:
            allowed &= self.group_array != component
        eligible = np.flatnonzero(allowed)
        if method == "tfidf":
            eligible = eligible[lexical[eligible] >= .03]
        if "e5" in sources:
            eligible = eligible[values["e5"][eligible] >= .45]
        if "norberto" in sources:
            eligible = eligible[values["norberto"][eligible] >= 0.]
        relevance = np.zeros(len(self.records))
        if len(sources) == 1:
            relevance = values[sources[0]]
        elif len(eligible):
            relevance[eligible] = fuse_ranks([values[source][eligible] for source in sources])
        ordered = eligible[np.argsort(-relevance[eligible], kind="stable")]
        positions = np.zeros(len(self.records), dtype=int)
        positions[ordered] = np.arange(1, len(ordered) + 1)
        result = (ordered, relevance, values, sources, positions)
        # Keep only one target's contexts; full query-by-bank matrices already dominate memory.
        if self._contexts and next(iter(self._contexts))[0] != target.excel_row:
            self._contexts.clear()
        self._contexts[key] = result
        return result

    def retrieve(self, target: Text, *, method: str, count: int = 3, view=None, policy=None):
        if type(target) is not Text or count < 1:
            raise TypeError("Retrieval accepts label-free Text targets and a positive count.")
        policy = policy or self.policy
        view = view or self.view
        ordered, relevance, values, sources, positions = self.context(target, method, view)
        pool = []
        for index in ordered[:policy.candidate_pool * 10]:
            index = int(index)
            if not self._query_duplicate(target.excel_row, index):
                pool.append(index)
            if len(pool) == policy.candidate_pool:
                break
        chosen, counts, used = [], Counter(), set()
        redundancy = np.zeros(len(pool))
        lexical_max = np.zeros(len(pool))
        while len(chosen) < count:
            candidates = []
            for position, index in enumerate(pool):
                record = self.records[index]
                component = self.bank_groups[record.excel_row]
                if index in chosen or component in used or (policy.class_cap is not None and counts[record.label] >= policy.class_cap):
                    continue
                if chosen and lexical_max[position] >= .95:
                    continue
                if any(self._pair_duplicate(min(index, other), max(index, other)) for other in chosen):
                    continue
                value = policy.relevance_weight * float(relevance[index]) - (1 - policy.relevance_weight) * redundancy[position]
                value += policy.class_bonus if not counts[record.label] else 0.
                candidates.append((value, -record.excel_row, index))
            if not candidates:
                break
            _, _, best = max(candidates)
            chosen.append(best)
            counts[self.records[best].label] += 1
            used.add(self.bank_groups[self.records[best].excel_row])
            if len(chosen) < count:
                pair = (self.selector.matrix[pool] @ self.selector.matrix[best].T).toarray().ravel()
                lexical_max = np.maximum(lexical_max, pair)
                for source in sources:
                    if source != "tfidf":
                        name = "e5" if source == "e5" else view
                        pair = np.maximum(pair, self.bank_vectors[name][pool] @ self.bank_vectors[name][best])
                redundancy = np.maximum(redundancy, pair)
        examples = [{"excel_row": self.records[i].excel_row, "resp_text": self.records[i].text,
                     "text_sha256": digest(self.records[i].text), "label": self.records[i].label,
                     "group_id": self.bank_groups[self.records[i].excel_row], "relevance": float(relevance[i]),
                     "lexical_cosine": float(values["tfidf"][i]),
                     "semantic_cosine": float(values["e5"][i]) if "e5" in values else None,
                     "norberto_cosine": float(values["norberto"][i]) if "norberto" in values else None,
                     "candidate_rank": int(positions[i])} for i in chosen]
        return examples, {"policy": asdict(policy), "method": method, "view": view if "norberto" in sources else None,
                          "eligible": len(ordered), "returned": len(chosen), "requested": count,
                          "class_counts": dict(counts), "shortfall": len(chosen) < count,
                          "candidate_pool_rows": len(pool)}

    def baseline(self, target: Text, *, method: str, count: int = 3):
        """Label-blind random/length controls obey the same mandatory duplicate guards."""
        self.prepare_targets([target])
        if method == "random":
            rng = np.random.default_rng(int(digest("random42:" + target.text)[:16], 16))
            ordered = rng.permutation(len(self.records)).tolist()
        elif method == "length":
            ordered = np.argsort(np.abs(self.lengths - len(clean(target.text))), kind="stable").tolist()
        else:
            raise ValueError("Unknown simple retrieval baseline.")
        selected, used = [], set()
        for index in ordered:
            record = self.records[index]
            component = self.bank_groups[record.excel_row]
            if (component in used or component == self.query_groups.get(target.excel_row)
                    or self.normalized[index] == group(target.text)
                    or self.lexical[self.target_indices[target.excel_row], index] >= .95
                    or self._query_duplicate(target.excel_row, index)):
                continue
            if any(self._pair_duplicate(min(index, other), max(index, other)) for other in selected):
                continue
            if selected and (self.selector.matrix[index] @ self.selector.matrix[selected].T).toarray().max() >= .95:
                continue
            selected.append(index)
            used.add(component)
            if len(selected) == count:
                break
        return [{"excel_row": self.records[i].excel_row, "label": self.records[i].label,
                 "candidate_rank": position + 1, "lexical_cosine": float(self.lexical[self.target_indices[target.excel_row], i]),
                 "semantic_cosine": None, "norberto_cosine": None} for position, i in enumerate(selected)]
