"""Declarative member selection, independent of model implementations."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnsembleConfig:
    models: tuple[str, ...]
    members: dict[str, dict[str, str]]
    strategy: str
    gpt_config: Path | None
    tiebreaker: str = "gpt"

    @classmethod
    def load(cls, path: Path, *, models: list[str] | None = None,
             strategy: str | None = None, tiebreaker: str | None = None) -> EnsembleConfig:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) - {"models", "members", "strategy", "gpt_config", "tiebreaker"}:
            raise ValueError("Unknown ensemble configuration fields.")
        chosen = models if models is not None else data.get("models")
        if (not isinstance(chosen, list) or not chosen
                or any(not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
                       or name == "gpt" for name in chosen)
                or len(set(chosen)) != len(chosen)):
            raise ValueError("models must contain distinct non-GPT member identifiers.")
        selected_tiebreaker = tiebreaker if tiebreaker is not None else data.get("tiebreaker", "gpt")
        if (not isinstance(selected_tiebreaker, str)
                or not re.fullmatch(r"[a-z][a-z0-9_]*", selected_tiebreaker)):
            raise ValueError("tiebreaker must be gpt or a configured classifier identifier.")
        required = set(chosen) | ({selected_tiebreaker} if selected_tiebreaker != "gpt" else set())
        members = data.get("members")
        if not isinstance(members, dict) or required - set(members):
            raise ValueError("Every selected model and non-GPT tiebreaker requires a members entry.")
        for name in required:
            value = members[name]
            if (not isinstance(value, dict) or set(value) - {"run", "identity_field", "adapter"}
                    or not isinstance(value.get("run"), str) or not value["run"]
                    or any(key in value and (not isinstance(value[key], str) or not value[key])
                           for key in ("identity_field", "adapter"))):
                raise ValueError(f"Invalid run specification for {name}.")
        selected_strategy = strategy if strategy is not None else data.get("strategy", "consensus")
        if not isinstance(selected_strategy, str):
            raise ValueError("strategy must be a string.")
        selected_strategy = {"consensus_gpt": "consensus", "majority_gpt": "majority"}.get(
            selected_strategy, selected_strategy)
        if selected_strategy not in {"consensus", "majority"}:
            raise ValueError("strategy must be consensus or majority.")
        gpt_config = data.get("gpt_config")
        if selected_tiebreaker == "gpt" and (not isinstance(gpt_config, str) or not gpt_config):
            raise ValueError("gpt_config must identify a fixed GPT configuration file.")
        return cls(tuple(chosen), {name: dict(members[name]) for name in sorted(required)},
                   selected_strategy, Path(gpt_config) if selected_tiebreaker == "gpt" else None,
                   selected_tiebreaker)
