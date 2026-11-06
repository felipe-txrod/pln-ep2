"""Train-only lexical/semantic retrieval with near-duplicate and diversity controls."""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import importlib.metadata
import math

import numpy as np

from gpt_classifier.data import Example, Text, canonical, clean, digest, group
from gpt_classifier.selection import Selector


def near_duplicate(left: str, right: str, *, threshold: float) -> bool:
    left, right = clean(left).lower(), clean(right).lower()
    if left == right:
        return True
    if min(len(left), len(right)) < 24:
        return SequenceMatcher(None, left, right, autojunk=False).ratio() >= threshold
    if min(len(left), len(right)) / max(len(left), len(right)) < threshold:
        return False
    a = {left[index:index + 5] for index in range(len(left) - 4)}
    b = {right[index:index + 5] for index in range(len(right) - 4)}
    return len(a & b) / max(len(a | b), 1) >= threshold


class Retriever:
    """Reuse the existing pinned local E5 encoder and train-only TF-IDF selector."""

    def __init__(self, bank: list[Example], components: dict[int, str], config, *, embeddings=None):
        if any(type(record) is not Example for record in bank) or set(components) != {r.excel_row for r in bank}:
            raise ValueError("The bank must contain only verified training examples and their groups.")
        if embeddings is None and set(config.retrieval_methods) & {"semantic", "hybrid"}:
            from pathlib import Path
            from gpt_classifier.embeddings import LocalEmbeddings
            embeddings = LocalEmbeddings(Path(config.embedding_cache))  # Local files only; never downloads.
        self.config, self.components = config, components
        bounded = [record for record in bank if len(record.text) <= config.max_example_chars]
        self.selector = Selector(bounded, max_chars=config.max_example_chars, embeddings=embeddings)
        self.selector.summary.update(original_bank_rows=len(bank), raw_character_limit=config.max_example_chars)
        self.records = self.selector.records
        self.identity = {"bank_sha256": digest(canonical([(r.excel_row, digest(r.text), r.label, components[r.excel_row])
                                                         for r in self.records])),
                         "tfidf_sha256": digest(canonical({"vocabulary": {term: int(index) for term, index in self.selector.vectorizer.vocabulary_.items()},
                                                           "idf": self.selector.vectorizer.idf_.tolist()})),
                         "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")},
                         "selection": self.selector.summary,
                         "encoder": embeddings.metadata if embeddings is not None else None}

    def prepare_targets(self, targets: list[Text]):
        if any(type(target) is not Text for target in targets):
            raise TypeError("Retrieval accepts label-free Text targets only.")
        self.selector.prepare_targets([target.text for target in targets])

    def retrieve(self, target: Text, *, method: str, count: int):
        if type(target) is not Text or method not in self.config.retrieval_methods or count not in self.config.example_counts:
            raise ValueError("Invalid retrieval target or recipe.")
        selector, config = self.selector, self.config
        lexical = (selector.matrix @ selector.vectorizer.transform([clean(target.text)]).T).toarray().ravel()
        semantic = None
        if selector.embeddings is not None:
            semantic = selector.semantic_matrix @ selector.embeddings.encode([target.text])[0]
        eligible = np.asarray([i for i, record in enumerate(self.records)
                               if group(record.text) != group(target.text)
                               and lexical[i] < config.duplicate_cosine
                               and ((lexical[i] >= config.minimum_lexical_similarity) if method == "tfidf"
                                    else semantic[i] >= config.minimum_semantic_similarity)], dtype=int)
        if not len(eligible):
            return [], {"eligible": 0, "returned": 0, "requested": count, "reason": "No relevant nonduplicate training examples"}
        relevance = np.zeros(len(self.records))
        if method == "tfidf":
            relevance = lexical
        elif method == "semantic":
            relevance = semantic
        else:
            relevance[eligible] = .5 * selector._rank_scores(lexical[eligible]) + .5 * selector._rank_scores(semantic[eligible])
        ordered = sorted(eligible.tolist(), key=lambda i: (-float(relevance[i]), self.records[i].excel_row))
        pool = []
        inspected = 0
        for index in ordered[:config.candidate_pool * 10]:
            inspected += 1
            if not near_duplicate(target.text, self.records[index].text, threshold=config.duplicate_jaccard):
                pool.append(index)
            if len(pool) == config.candidate_pool:
                break
        chosen, counts, used_components = [], Counter(), set()
        for _ in range(count):
            candidates = []
            for index in pool:
                record = self.records[index]
                if (index in chosen or self.components[record.excel_row] in used_components
                        or counts[record.label] >= math.ceil(count / 2)):
                    continue
                if any(near_duplicate(record.text, self.records[other].text, threshold=config.duplicate_jaccard) for other in chosen):
                    continue
                redundancy = 0.
                if chosen:
                    redundancy = float((selector.matrix[index] @ selector.matrix[chosen].T).toarray().max())
                    if redundancy >= config.duplicate_cosine:
                        continue
                    if method != "tfidf":
                        semantic_redundancy = float((selector.semantic_matrix[chosen] @ selector.semantic_matrix[index]).max())
                        redundancy = max(redundancy, semantic_redundancy)
                value = config.mmr_weight * float(relevance[index]) - (1 - config.mmr_weight) * redundancy
                value += config.class_bonus if not counts[record.label] else 0
                candidates.append((value, -record.excel_row, index))
            if not candidates:
                break
            _, _, best = max(candidates)
            chosen.append(best)
            counts[self.records[best].label] += 1
            used_components.add(self.components[self.records[best].excel_row])
        results = [{"excel_row": self.records[i].excel_row, "resp_text": self.records[i].text,
                    "text_sha256": digest(self.records[i].text), "label": self.records[i].label,
                    "group_id": self.components[self.records[i].excel_row], "relevance": float(relevance[i]),
                    "lexical_cosine": float(lexical[i]), "semantic_cosine": float(semantic[i]) if semantic is not None else None}
                   for i in chosen]
        return results, {"eligible": len(eligible), "returned": len(results), "requested": count,
                         "class_counts": dict(counts), "shortfall": len(results) < count,
                         "near_duplicate_candidates_inspected": inspected, "diverse_pool_rows": len(pool)}
