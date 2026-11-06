"""Explicit experiment limits; execution is always opt-in."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path

from cascade_classifier.config import CascadeConfig


@dataclass(frozen=True)
class RagConfig:
    cascade_model: str = "models/stacking-gpt-cascade/model.json"
    oof_directory: str = "results/cascade-validation-20261105"
    embedding_cache: str = ".cache/retrieval"
    response_cache: str = ".cache/cascade/responses.sqlite3"
    pilot_size: int = 48
    retrieval_methods: tuple[str, ...] = ("tfidf", "semantic", "hybrid")
    example_counts: tuple[int, ...] = (3, 5)
    max_example_chars: int = 1600
    candidate_pool: int = 120
    duplicate_cosine: float = .95
    duplicate_jaccard: float = .85
    minimum_lexical_similarity: float = .03
    minimum_semantic_similarity: float = .45
    mmr_weight: float = .8
    class_bonus: float = .04
    max_new_requests: int = 128
    transport: CascadeConfig = field(default_factory=lambda: CascadeConfig(budgets=(.05, .1)))

    def __post_init__(self):
        if (not self.retrieval_methods or len(set(self.retrieval_methods)) != len(self.retrieval_methods)
                or not set(self.retrieval_methods) <= {"tfidf", "semantic", "hybrid"}):
            raise ValueError("Unknown or repeated retrieval method.")
        if (not self.example_counts or len(set(self.example_counts)) != len(self.example_counts)
                or any(type(count) is not int or not 3 <= count <= 5 for count in self.example_counts)):
            raise ValueError("Compare three to five examples.")
        for name, minimum in (("pilot_size", 3), ("max_example_chars", 32), ("candidate_pool", 5), ("max_new_requests", 1)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"Invalid {name}.")
        for name in ("duplicate_cosine", "duplicate_jaccard", "minimum_lexical_similarity",
                     "minimum_semantic_similarity", "mmr_weight", "class_bonus"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {name}.")
        if max(self.transport.budgets) > .2:
            raise ValueError("This context study is limited to at most 20% selective routing.")

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        transport = dict(data.pop("transport", {}))
        transport.setdefault("budgets", [.05, .1])
        transport["budgets"] = tuple(transport["budgets"])
        data["transport"] = CascadeConfig(**transport)
        for name in ("retrieval_methods", "example_counts"):
            if name in data:
                data[name] = tuple(data[name])
        return cls(**data)

    @classmethod
    def load(cls, path: Path):
        value = json.loads(path.read_text())
        for name, converter in (("response_cache", str), ("embedding_cache", str),
                                ("pilot_size", int), ("max_new_requests", int)):
            if "RAG_" + name.upper() in os.environ:
                value[name] = converter(os.environ["RAG_" + name.upper()])
        for name in ("model", "reasoning_effort"):
            if "CASCADE_" + name.upper() in os.environ:
                value.setdefault("transport", {})[name] = os.environ["CASCADE_" + name.upper()]
        return cls.from_dict(value)

    def to_dict(self):
        value = asdict(self)
        value["transport"] = self.transport.to_dict()
        value["retrieval_methods"] = list(self.retrieval_methods)
        value["example_counts"] = list(self.example_counts)
        return value
