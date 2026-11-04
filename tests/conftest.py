from __future__ import annotations

import json

import pytest
from openpyxl import Workbook

from gpt_classifier.data import Example


@pytest.fixture
def bank():
    return [Example(2 + i * 12 + j, f"{word} exemplo {j} texto específico informação assunto{j}", label)
            for i, (label, word) in enumerate(zip(("c1", "c234", "c5"), ("confuso", "parcial", "claro")))
            for j in range(10)]


@pytest.fixture
def make_xlsx(tmp_path):
    def create(name, rows, headers=("resp_text", "clarity")):
        path = tmp_path / name
        workbook = Workbook()
        workbook.active.append(headers)
        for row in rows:
            workbook.active.append(row)
        workbook.save(path)
        workbook.close()
        return path
    return create


def response(label="c5", **updates):
    result = {"id": "resp_mock", "object": "response", "created_at": 1, "model": "mock-model",
              "status": "completed", "output": [{"type": "message", "role": "assistant", "id": "msg_mock",
              "status": "completed", "content": [{"type": "output_text", "text": json.dumps({"label": label}), "annotations": []}]}],
              "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25,
                        "input_tokens_details": {"cached_tokens": 3}, "output_tokens_details": {"reasoning_tokens": 0}}}
    return {**result, **updates}
