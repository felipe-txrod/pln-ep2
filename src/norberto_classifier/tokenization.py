"""Preserve the cased preprocessing and head/tail truncation used in EP1."""
from __future__ import annotations

import html
import re
from typing import Any


def clean_cased(text: str) -> str:
    """Normalize HTML entities and whitespace without lowercasing the response."""
    return re.sub(r"\s+", " ", html.unescape(str(text))).strip()


def trim_tokens(ids: list[int], budget: int, policy: str) -> list[int]:
    if budget < 1 or policy not in {"head", "head_tail"}:
        raise ValueError("Invalid token budget or truncation policy.")
    if len(ids) <= budget:
        return ids
    if policy == "head":
        return ids[:budget]
    head = (budget + 1) // 2
    tail = budget - head
    return ids[:head] + (ids[-tail:] if tail else [])


def tokenize_texts(texts: list[str], tokenizer: Any, config: dict) -> tuple[list[dict], dict]:
    normalized = [clean_cased(text) for text in texts]
    token_lists = tokenizer(normalized, add_special_tokens=False, truncation=False,
                            verbose=False)["input_ids"]
    budget = config["max_length"] - tokenizer.num_special_tokens_to_add(pair=False)
    input_names = list(getattr(tokenizer, "model_input_names", []))
    features = []
    for ids in token_lists:
        inputs = tokenizer.build_inputs_with_special_tokens(trim_tokens(ids, budget, config["truncation"]))
        feature = {"input_ids": inputs, "attention_mask": [1] * len(inputs)}
        if "token_type_ids" in input_names:
            feature["token_type_ids"] = [0] * len(inputs)
        features.append(feature)
    return features, {"rows": len(texts), "truncated_rows": sum(len(ids) > budget for ids in token_lists),
                      "input_case_preserved": True, "max_length": config["max_length"],
                      "truncation": config["truncation"]}
