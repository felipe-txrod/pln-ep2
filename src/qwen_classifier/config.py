"""Validated experiment configuration shared by training and inference."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from gpt_classifier.data import LABELS, canonical, digest


@dataclass(frozen=True)
class Config:
    model_id: str = "Qwen/Qwen3-1.7B"
    revision: str = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    mode: str = "auto"
    max_length: int = 256
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    learning_rate: float = 0.0002
    epochs: int = 3
    batch_size: int = 1
    accumulation_steps: int = 16
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    seed: int = 42
    folds: int = 5
    fold: int = 0
    attention: str = "sdpa"
    gradient_checkpointing: bool = True
    eval_batch_size: int = 1
    group_by_length: bool = False

    def __post_init__(self) -> None:
        if self.model_id != "Qwen/Qwen3-1.7B" or not re.fullmatch(r"[a-f0-9]{40}", self.revision):
            raise ValueError("Use Qwen/Qwen3-1.7B with an immutable 40-character revision.")
        if self.mode not in {"auto", "qlora", "fp16"}:
            raise ValueError("mode must be auto, qlora, or fp16.")
        for name in ("max_length", "rank", "alpha", "epochs", "batch_size", "accumulation_steps", "eval_batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not 8 <= self.max_length <= 2048:
            raise ValueError("max_length must be between 8 and 2048 tokens.")
        if type(self.folds) is not int or type(self.fold) is not int or self.folds < 2 or not 0 <= self.fold < self.folds:
            raise ValueError("Invalid grouped cross-validation fold.")
        for name in ("dropout", "learning_rate", "warmup_ratio", "weight_decay"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}.")
        if self.dropout >= 1 or self.warmup_ratio >= 1 or self.learning_rate <= 0:
            raise ValueError("Invalid dropout, warmup ratio, or learning rate.")
        allowed = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        if not self.target_modules or len(set(self.target_modules)) != len(self.target_modules) or not set(self.target_modules) <= allowed:
            raise ValueError("Invalid LoRA target modules.")
        if self.attention not in {"sdpa", "eager"}:
            raise ValueError("attention must be sdpa or eager.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        for name in ("gradient_checkpointing", "group_by_length"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean.")

    def to_dict(self) -> dict:
        return {**asdict(self), "target_modules": list(self.target_modules)}

    @property
    def fingerprint(self) -> str:
        return digest(canonical(self.to_dict()))

    def with_mode(self, mode: str) -> Config:
        return replace(self, mode=mode)

    @classmethod
    def load(cls, path: Path) -> Config:
        values = json.loads(path.read_text(encoding="utf-8"))
        if "target_modules" in values:
            values["target_modules"] = tuple(values["target_modules"])
        return cls(**values)


LABEL2ID = {label: index for index, label in enumerate(LABELS)}
