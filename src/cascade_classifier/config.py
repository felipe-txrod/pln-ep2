"""Validated experiment configuration and explicit runtime overrides."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path


@dataclass(frozen=True)
class CascadeConfig:
    stacking_model: str = "models/ensemble-stacking-qwen/model.json"
    holdout_scores: str = "results/ensemble-soft-qwen-20261105/scores"
    model: str = "gpt-6.1-sol"
    reasoning_effort: str = "low"
    budgets: tuple[float, ...] = (.05, .10, .20)
    initial_budget: float = .10
    batch_size: int = 8
    workers: int = 2
    retries: int = 1
    timeout: float = 180
    max_request_bytes: int = 1_000_000
    input_price_per_million: float = 2
    cached_input_price_per_million: float = .10
    output_price_per_million: float = 10

    def __post_init__(self):
        if not self.model.strip() or self.reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Invalid model or reasoning effort.")
        if (not self.budgets or tuple(sorted(set(self.budgets))) != self.budgets
                or any(type(x) not in (float, int) or not math.isfinite(x) or not 0 < x <= 1 for x in self.budgets)
                or self.initial_budget not in self.budgets):
            raise ValueError("Budgets must be unique, sorted fractions; initial_budget must be included.")
        for name, lower, upper in (("batch_size", 1, 8), ("workers", 1, 4), ("retries", 0, 3),
                                   ("max_request_bytes", 1024, 10_000_000)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"Invalid {name}.")
        for name in ("timeout", "input_price_per_million", "cached_input_price_per_million", "output_price_per_million"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}.")
        if self.timeout == 0 or self.cached_input_price_per_million > self.input_price_per_million:
            raise ValueError("Invalid timeout or cached input price.")

    @classmethod
    def load(cls, path: Path):
        value = json.loads(path.read_text())
        converters = {"MODEL": str, "REASONING_EFFORT": str, "BATCH_SIZE": int, "WORKERS": int,
                      "RETRIES": int, "TIMEOUT": float, "INPUT_PRICE_PER_MILLION": float,
                      "CACHED_INPUT_PRICE_PER_MILLION": float, "OUTPUT_PRICE_PER_MILLION": float}
        for name, convert in converters.items():
            if f"CASCADE_{name}" in os.environ:
                value[name.lower()] = convert(os.environ[f"CASCADE_{name}"])
        value["budgets"] = tuple(value.get("budgets", (.05, .1, .2)))
        return cls(**value)

    def to_dict(self):
        result = asdict(self)
        result["budgets"] = list(self.budgets)
        return result
