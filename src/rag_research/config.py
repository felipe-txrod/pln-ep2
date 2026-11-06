"""Predeclared offline experiment controls, independent of frozen cascade code."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ResearchConfig:
    study: str = "results/rag-context-pilot-20261106"
    e5_cache: str = ".cache/retrieval"
    norberto_cache: str = ".cache/rag-norberto-ft"
    norberto_artifact: str = "models/norberto"
    seed: int = 42
    layers: tuple[int, ...] = (11, 22)
    poolings: tuple[str, ...] = ("mean", "cls")
    batch_size: int = 16
    device: str = "gpu"
    inner_folds: int = 3
    regularization: tuple[float, ...] = (.01, .1, 1.)
    benefit_thresholds: tuple[float, ...] = (0., .05, .15)
    route_budgets: tuple[float, ...] = (.05, .1)

    def __post_init__(self):
        if self.device not in {"cpu", "gpu"} or self.batch_size < 1 or self.inner_folds < 2:
            raise ValueError("Invalid device, batch size or grouped fold count.")
        if not self.layers or not set(self.layers) <= set(range(1, 23)) or len(set(self.layers)) != len(self.layers):
            raise ValueError("NorBERTo extraction requires unique layers in 1..22.")
        if not self.poolings or not set(self.poolings) <= {"mean", "cls"}:
            raise ValueError("Unsupported pooling.")
        if any(not 0 < value <= 1 for value in self.regularization):
            raise ValueError("Regularization must be positive and at most one.")
        if any(not 0 <= value <= .5 for value in self.benefit_thresholds):
            raise ValueError("Invalid expected-benefit threshold.")
        if any(not 0 < value <= .1 for value in self.route_budgets):
            raise ValueError("Offline logged-support routing is limited to at most 10%.")

    @classmethod
    def load(cls, path: Path):
        value = json.loads(path.read_text())
        for name in ("layers", "poolings", "regularization", "benefit_thresholds", "route_budgets"):
            if name in value:
                value[name] = tuple(value[name])
        return cls(**value)

    def to_dict(self):
        return asdict(self)
