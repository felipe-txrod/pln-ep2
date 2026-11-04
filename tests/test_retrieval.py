from __future__ import annotations

import json
import sqlite3
from dataclasses import replace

import numpy as np
import pytest

from conftest import response
from gpt_classifier.data import Example, Text, digest, group
from gpt_classifier.embeddings import LocalEmbeddings
from gpt_classifier.pipeline import run
from gpt_classifier.prompt import Config
from gpt_classifier.retrieval_study import retrieval_study
from gpt_classifier.selection import Selector, development_split


class FakeEmbeddings:
    metadata = {"model": "fixture", "revision": "fixed"}

    def __init__(self, *args, **kwargs):
        self.stats = {}

    def encode(self, texts):
        assert all(type(t) is str for t in texts)
        vectors = []
        for text in texts:
            vector = np.random.default_rng(int(digest(text)[:8], 16)).normal(size=384)
            vectors.append(vector / np.linalg.norm(vector))
        return np.asarray(vectors, dtype=np.float32)


def test_hybrid_balanced_deduplicated_conflict_free_and_self_excluded(bank):
    records = bank + [Example(500, "Conflito", "c1"), Example(501, " conflito ", "c5"),
                      Example(502, bank[0].text.upper(), bank[0].label)]
    selector = Selector(records, embeddings=FakeEmbeddings())
    selected = selector.select(bank[0].text, strategy="hybrid", per_class=2)
    assert [r.label for r in selected] == ["c1", "c234", "c5"] * 2
    assert len({group(r.text) for r in selected}) == 6
    assert all(group(r.text) != group(bank[0].text) for r in selected)
    assert all(r.excel_row not in (500, 501, 502) for r in selected)
    assert selected == selector.select(bank[0].text, strategy="hybrid", per_class=2)


def test_zero_semantic_weight_preserves_lexical_top1_and_ties(bank):
    selector = Selector(bank, embeddings=FakeEmbeddings(), semantic_weight=0)
    for text in ("confuso informação assunto3", "vocabulário completamente desconhecido"):
        assert selector.select(text, strategy="similar", per_class=1) == selector.select(text, strategy="hybrid", per_class=1)
    assert np.array_equal(selector._rank_scores(np.zeros(5)), np.ones(5))
    with pytest.raises(ValueError, match="encoder local"):
        Selector(bank).select("alvo", strategy="hybrid", per_class=1)


@pytest.mark.parametrize("value", [-0.1, 1.1, True, "0.5", float("nan"), float("inf")])
def test_hybrid_config_rejects_invalid_weights(value):
    for field in ("hybrid_semantic_weight", "hybrid_length_weight"):
        with pytest.raises(ValueError, match=field):
            Config(**{field: value})


def cached_encoder(tmp_path, *, namespace="model-v1"):
    encoder = object.__new__(LocalEmbeddings)
    encoder.cache_dir, encoder.batch_size, encoder.namespace = tmp_path, 2, namespace
    encoder.stats = {"cache_hits": 0, "encoded_texts": 0}
    encoder._encode = FakeEmbeddings().encode
    return encoder


def test_embedding_cache_normalizes_deduplicates_orders_and_invalidates_model(tmp_path):
    encoder = cached_encoder(tmp_path)
    first = encoder.encode(["Texto A", "Texto B", " Texto  A "])
    assert np.allclose(first[0], first[2]) and encoder.stats["encoded_texts"] == 2
    resumed = cached_encoder(tmp_path)
    resumed._encode = lambda _: pytest.fail("Cache quente não deve chamar encoder")
    assert np.array_equal(resumed.encode(["Texto B", "Texto A"]), first[[1, 0]])
    assert resumed.stats["cache_hits"] == 2
    changed = cached_encoder(tmp_path, namespace="model-v2")
    changed.encode(["Texto A"])
    assert changed.stats["encoded_texts"] == 1


def test_interrupted_embeddings_resume_committed_batches_and_reject_poison(tmp_path):
    encoder = cached_encoder(tmp_path)
    calls = []
    def interrupted(texts):
        calls.append(texts)
        if len(calls) == 2:
            raise RuntimeError("interrompido")
        return FakeEmbeddings().encode(texts)
    encoder._encode = interrupted
    with pytest.raises(RuntimeError):
        encoder.encode(["a", "b", "c"])
    resumed = cached_encoder(tmp_path)
    resumed.encode(["a", "b", "c"])
    assert resumed.stats == {"cache_hits": 2, "encoded_texts": 1}
    with sqlite3.connect(tmp_path / "embeddings.sqlite3") as db:
        db.execute("UPDATE embeddings SET vector=?", (np.zeros(384, dtype="<f4").tobytes(),))
    with pytest.raises(ValueError, match="nulo"):
        resumed.encode(["a"])


def test_masked_mean_ignores_padding_and_normalizes(monkeypatch):
    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace())
    encoder = object.__new__(LocalEmbeddings)
    a, b = np.eye(384, dtype=np.float32)[:2]
    encoder.session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")],
                                     run=lambda *args: [np.array([[a, b, a * 999]])])
    encoder.tokenizer = SimpleNamespace(encode_batch=lambda texts: [SimpleNamespace(ids=[0, 1, 1], attention_mask=[1, 1, 0])])
    actual = encoder._encode(["text"])
    assert np.allclose(actual[0], (a + b) / np.sqrt(2))


class MockBackend:
    kind = "codex"

    def __init__(self):
        self.attempts = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
        self.calls = []

    def metadata(self):
        return {"kind": "codex", "endpoint": "codex.exec", "cli_version": "fixture"}

    def create(self, request):
        self.attempts += 1
        self.calls.append(request)
        payload = json.loads(request["prompt"])
        assert all(set(item["target"]) == {"resp_text"} for item in payload["items"])
        items = [{"id": item["id"], "label": "c5"} for item in payload["items"]]
        result = response()
        result["output"][0]["content"][0]["text"] = json.dumps({"predictions": items})
        return result


def test_paired_study_train_only_budget_no_auto_selection_and_resume(tmp_path, bank, monkeypatch):
    from gpt_classifier import embeddings
    monkeypatch.setattr(embeddings, "LocalEmbeddings", FakeEmbeddings)
    config = Config(backend="codex", max_output_tokens=None, batch_size=3, strategy="hybrid")
    backend = MockBackend()
    kwargs = dict(output_dir=tmp_path / "study", retrieval_cache=tmp_path / "embeddings",
                  local_limit=9, pilot_limit=6, backend=backend, max_new_requests=4)
    preview = retrieval_study(bank, config, **kwargs)
    assert preview["status"] == "preview" and backend.attempts == 0
    result = retrieval_study(bank, config, execute=True, **kwargs)
    assert result["status"] == "completed" and backend.attempts <= 4
    assert retrieval_study(bank, config, **kwargs)["status"] == "completed"
    attempts = backend.attempts
    retrieval_study(bank, config, execute=True, **kwargs)
    assert backend.attempts == attempts
    assert not (tmp_path / "study/selected-config.json").exists()
    split = json.loads((tmp_path / "study/split.json").read_text())
    by_row = {r.excel_row: r for r in bank}
    bank_groups = {group(by_row[row].text) for row in split["bank_excel_rows"]}
    assert not bank_groups & {group(by_row[row].text) for row in split["pilot_excel_rows"]}
    with pytest.raises(ValueError, match="teto"):
        retrieval_study(bank, config, **{**kwargs, "output_dir": tmp_path / "limited", "max_new_requests": 0}, execute=True)
    assert backend.attempts == attempts


def test_pipeline_hybrid_metadata_and_model_change_refuses_resume(tmp_path, bank, monkeypatch):
    from gpt_classifier import embeddings
    monkeypatch.setattr(embeddings, "LocalEmbeddings", FakeEmbeddings)
    config = Config(backend="codex", max_output_tokens=None, batch_size=3, strategy="hybrid")
    kwargs = dict(output_dir=tmp_path / "run", cache_path=tmp_path / "responses.sqlite3", backend=MockBackend())
    records = [Text(100, "alvo A"), Text(101, "alvo B")]
    run(records, bank, config, dry_run=True, **kwargs)
    manifest = json.loads((tmp_path / "run/run.json").read_text())
    assert manifest["retrieval"]["semantic_encoder"]["revision"] == "fixed"
    monkeypatch.setattr(FakeEmbeddings, "metadata", {"model": "fixture", "revision": "changed"})
    with pytest.raises(ValueError, match="outro experimento"):
        run(records, bank, config, dry_run=True, **kwargs)


def test_explicit_larger_study_budget_and_default_remain_separate(tmp_path, bank, monkeypatch):
    from gpt_classifier import embeddings
    from gpt_classifier.cli import parser
    monkeypatch.setattr(embeddings, "LocalEmbeddings", FakeEmbeddings)
    config = Config(backend="codex", max_output_tokens=None, batch_size=3, strategy="hybrid")
    backend = MockBackend()
    kwargs = dict(output_dir=tmp_path / "expanded", retrieval_cache=tmp_path / "embeddings",
                  local_limit=9, pilot_limit=6, backend=backend)
    report = retrieval_study(bank, config, execute=True, max_new_requests=200, **kwargs)
    assert report["status"] == "completed" and backend.attempts <= 4
    assert report["max_new_requests"] == 200
    defaults = parser().parse_args(["retrieval-study", "--output-dir", str(tmp_path)])
    assert defaults.max_new_requests == 8 and defaults.pilot_limit == 12
    for budget in (-1, True, 2.5):
        with pytest.raises(ValueError, match="inteiro não negativo"):
            retrieval_study(bank, config, max_new_requests=budget, **kwargs)
