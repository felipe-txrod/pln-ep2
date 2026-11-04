from __future__ import annotations

import json

from .data import LABELS, canonical, digest
from .prompt import response_object

BATCH_INSTRUCTIONS = """
Classifique cada item de items de forma independente. Cada item tem id, target
e example_ids; consulte somente as demonstrações indicadas por example_ids no
dicionário examples. Não use outros alvos como demonstrações. Responda com
predictions: uma lista de objetos {id,label}, exatamente um por id recebido.
Não omita nem repita ids. A ordem das respostas é livre; os ids preservam o vínculo.
"""


def batch_request(requests: list[dict]) -> dict:
    if not requests or any(r.get("transport") != "codex" or "batch_ids" in r for r in requests):
        raise ValueError("O lote exige requests individuais Codex.")
    first = requests[0]
    for request in requests[1:]:
        if any(request.get(k) != first.get(k) for k in ("model", "instructions", "reasoning_effort", "runtime")):
            raise ValueError("Um lote não pode misturar modelos, prompts ou runtimes.")
    examples, seen, items = {}, {}, []
    for index, request in enumerate(requests):
        payload = json.loads(request["prompt"])
        refs = []
        for example in payload["examples"]:
            key = canonical(example)
            if key not in seen:
                identifier = f"e{len(examples)}"
                seen[key] = identifier
                examples[identifier] = example
            refs.append(seen[key])
        items.append({"id": f"t{index}", "target": payload["target"], "example_ids": refs})
    identifiers = [item["id"] for item in items]
    schema = {"type": "object", "properties": {"predictions": {"type": "array",
        "items": {"type": "object", "properties": {"id": {"type": "string", "enum": identifiers},
                  "label": {"type": "string", "enum": list(LABELS)}},
                  "required": ["id", "label"], "additionalProperties": False},
        "minItems": len(items), "maxItems": len(items)}},
        "required": ["predictions"], "additionalProperties": False}
    return {**first, "instructions": first["instructions"] + BATCH_INSTRUCTIONS,
            "prompt": canonical({"examples": examples, "items": items}),
            "schema": schema, "batch_ids": identifiers}


def parse_batch_response(response: dict, expected_ids: list[str]) -> dict[str, str]:
    value = response_object(response)
    if not isinstance(value, dict) or set(value) != {"predictions"} or not isinstance(value["predictions"], list):
        raise ValueError("JSON de lote não corresponde ao schema.")
    predictions = {}
    for item in value["predictions"]:
        if not isinstance(item, dict) or set(item) != {"id", "label"}:
            raise ValueError("Item de lote inválido.")
        identifier, label = item["id"], item["label"]
        if not isinstance(identifier, str) or identifier not in expected_ids or identifier in predictions or label not in LABELS:
            raise ValueError("ID duplicado/desconhecido ou classe inválida no lote.")
        predictions[identifier] = label
    if set(predictions) != set(expected_ids) or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("O lote não retornou exatamente uma classificação por alvo.")
    return predictions


def prepare_batches(requests: list[dict], *, batch_size: int, max_bytes: int):
    """Return (payload, [(item_id, original_row_indices)]) preserving deduplication.

    Byte limit may split a batch, but never truncate an additional target or add
    records after inference. Exact duplicates share one item and one response.
    """
    unique = {}
    for index, request in enumerate(requests):
        key = digest(canonical(request))
        if key not in unique:
            unique[key] = (request, [])
        unique[key][1].append(index)
    entries = list(unique.values())
    result, current = [], []

    def pack(group):
        payload = batch_request([r for r, _ in group])
        return payload, [(identifier, indices) for identifier, (_, indices) in zip(payload["batch_ids"], group)]

    for entry in entries:
        proposed = current + [entry]
        payload, mapping = pack(proposed)
        if current and (len(proposed) > batch_size or len(canonical(payload).encode("utf-8")) > max_bytes):
            result.append(pack(current))
            current = [entry]
        else:
            current = proposed
        candidate, _ = pack(current)
        if len(canonical(candidate).encode("utf-8")) > max_bytes:
            raise ValueError("Um item com o schema do lote excede max_request_bytes.")
    if current:
        result.append(pack(current))
    return result
