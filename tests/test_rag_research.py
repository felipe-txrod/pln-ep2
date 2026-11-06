"""Offline research integrity tests with synthetic vectors and logged decisions only."""
from dataclasses import replace
import hashlib
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import copy

import numpy as np
import pytest

from gpt_classifier.data import Example, Text
from gpt_classifier.selection import Selector
from rag_cascade.config import RagConfig
from rag_cascade.retrieval import Retriever
from rag_research.config import ResearchConfig
from rag_research.norberto_embeddings import FineTunedEmbeddings, normalize_vectors, pool_hidden
from rag_research.retrieval import ConfigurableRetriever, SelectionPolicy, ablations, fuse_ranks
from rag_research.acceptance import Policy, apply_policy, assert_disjoint, run as run_acceptance
from rag_research.interventions import acceptance_rule, post_call_features, pre_call_features, rule_can_change_prediction


class SyntheticEncoder:
    metadata = {"synthetic": True}

    def encode(self, texts):
        result = []
        for text in texts:
            rng = np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))
            vector = np.ones(384) + rng.uniform(-.2, .2, 384)
            result.append(vector / np.linalg.norm(vector))
        return np.asarray(result, dtype=np.float32)


def index_fixture():
    texts = ["Public budget information explains spending limits and authorizations.",
             "Hospital information describes patient rights and service schedules.",
             "Municipal information explains construction permits and documentation.",
             "Information about public staff gives salary totals and job categories.",
             "The university provides historical information about research funding.",
             "Information requested concerns payments and the current tax legislation.",
             "Citizens find information on health policy and annual budget decisions.",
             "Education information lists the schools and student transportation routes.",
             "The agency sent information about community projects and enrollment."]
    bank = [Example(i + 2, text, ("c1", "c234", "c5")[i % 3]) for i, text in enumerate(texts)]
    target = Text(100, "The information request concerns public budget and health service funding.")
    groups = {r.excel_row: str(r.excel_row) for r in bank}
    encoder = SyntheticEncoder()
    old = Retriever(bank, groups, RagConfig(), embeddings=encoder)
    new = ConfigurableRetriever(old.selector, groups, [target], {100: "query-group"},
                                {"e5": old.selector.semantic_matrix}, {"e5": encoder.encode([target.text])})
    return target, old, new


def test_rank_fusion_is_scale_invariant_and_tie_aware():
    left, right = np.array([.2, .1, .1]), np.array([100., 500., 200.])
    assert np.array_equal(fuse_ranks([left, right]), fuse_ranks([left * 1000, right / 17]))
    assert fuse_ranks([left])[1] == fuse_ranks([left])[2]
    with pytest.raises(ValueError):
        fuse_ranks([np.array([float("nan")])])


def test_legacy_hybrid_is_reproduced_and_diversity_cannot_change_first_neighbor():
    target, old, new = index_fixture()
    expected, _ = old.retrieve(target, method="hybrid", count=3)
    actual, _ = new.retrieve(target, method="tfidf_e5", count=3)
    assert [r["excel_row"] for r in actual] == [r["excel_row"] for r in expected]
    first = actual[0]["excel_row"]
    for policy in ablations():
        selected, _ = new.retrieve(target, method="tfidf_e5", policy=policy)
        assert selected[0]["excel_row"] == first
    with pytest.raises(TypeError):
        new.retrieve(Example(target.excel_row, target.text, "c1"), method="e5")


def test_group_exclusion_and_query_text_identity_are_enforced():
    target, _, index = index_fixture()
    index.query_groups[target.excel_row] = index.bank_groups[index.records[0].excel_row]
    selected, _ = index.retrieve(target, method="e5")
    assert index.records[0].excel_row not in {row["excel_row"] for row in selected}
    with pytest.raises(ValueError):
        index.retrieve(Text(target.excel_row, "Tampered text"), method="e5")
    index.lexical[0, :6] = .99
    for method in ("random", "length"):
        control = index.baseline(target, method=method)
        assert len(control) == 3 and all(row["lexical_cosine"] < .95 for row in control)


def test_relevance_only_can_return_three_same_class_examples():
    bank = [Example(i + 2, text, "c1" if i < 3 else "c234" if i < 5 else "c5") for i, text in enumerate([
        "Annual expenditure funding was approved by parliament.", "Hospital staff enrollment begins in January.",
        "Tax documents require a signed statement and identification.", "University scholarship payments are delayed.",
        "Public transportation schedules list accessible routes.", "Environmental permits cover construction waste disposal."])]
    selector = Selector(bank)
    groups = {r.excel_row: str(r.excel_row) for r in bank}
    target = Text(100, "Requested information is described below.")
    cosine = np.array([.99, .98, .97, .8, .7, .6])
    vectors = np.column_stack([cosine, np.sqrt(1 - cosine ** 2)]).astype(np.float32)
    index = ConfigurableRetriever(selector, groups, [target], {}, {"e5": vectors}, {"e5": np.array([[1., 0.]], dtype=np.float32)})
    pure, _ = index.retrieve(target, method="e5", policy=SelectionPolicy("relevance", 1., 0., None))
    constrained, _ = index.retrieve(target, method="e5", policy=SelectionPolicy())
    assert [r["label"] for r in pure] == ["c1", "c1", "c1"]
    assert max([r["label"] for r in constrained].count(label) for label in ("c1", "c234", "c5")) <= 2


def test_frozen_feature_cache_reuses_views_and_rejects_corruption(tmp_path):
    config = replace(ResearchConfig(), norberto_cache=str(tmp_path), device="cpu")
    calls = []
    def encode(texts):
        calls.append(list(texts))
        values = np.ones((len(texts), 768), dtype=np.float32)
        values[:, 0] = [len(t) + 1 for t in texts]
        return {f"layer{layer}_{mode}": values for layer in config.layers for mode in config.poolings}
    encoder = FineTunedEmbeddings(config, encoder=encode, metadata={"synthetic": True, "adapter": "trained"})
    targets = [Text(1, "Synthetic response"), Text(2, "Another example")]
    first, second = encoder.encode(targets), encoder.encode(targets)
    assert len(calls) == 1 and set(first) == {"layer11_mean", "layer11_cls", "layer22_mean", "layer22_cls"}
    assert all(np.array_equal(first[view], second[view]) for view in first)
    assert encoder.stats == {"cache_hits": 2, "encoded_texts": 2}
    with pytest.raises(TypeError):
        encoder.encode([Example(3, "Labeled target must not enter inference", "c1")])
    with sqlite3.connect(tmp_path / "embeddings.sqlite3") as db:
        db.execute("UPDATE embeddings SET vector=?", (b"corrupt",))
    with pytest.raises(ValueError, match="checksum"):
        encoder.encode(targets)


def test_base_and_adapter_cache_identities_never_mix(tmp_path):
    config = replace(ResearchConfig(), norberto_cache=str(tmp_path), device="cpu")
    fake = lambda texts: {f"layer{layer}_{mode}": np.ones((len(texts), 768)) for layer in config.layers for mode in config.poolings}
    base = FineTunedEmbeddings(config, encoder=fake, metadata={"adapter": None})
    trained = FineTunedEmbeddings(config, encoder=fake, metadata={"adapter": "frozen-fine-tuned"})
    base.encode([Text(1, "Same text")])
    trained.encode([Text(1, "Same text")])
    assert base.namespace != trained.namespace and trained.stats["cache_hits"] == 0


def test_embedding_validation_and_masked_pooling():
    with pytest.raises(ValueError):
        normalize_vectors(np.zeros((2, 768)), 2)
    with pytest.raises(ValueError):
        normalize_vectors(np.ones((2, 384)), 2)
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[1., 2.], [3., 4.], [999., 999.]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.equal(pool_hidden(hidden, mask, "mean"), torch.tensor([[2., 3.]]))
    assert torch.equal(pool_hidden(hidden, mask, "cls"), torch.tensor([[1., 2.]]))


def logged_fixture(n=90):
    rng = np.random.default_rng(91)
    routing, gold, logged, groups, folds = {}, {}, {}, {}, {}
    labels = ("c1", "c234", "c5")
    for i in range(n):
        p = rng.dirichlet([2, 2, 2])
        label = labels[int(p.argmax())]
        gold[i] = labels[i % 3]
        routing[i] = {"label": label, "probabilities": dict(zip(labels, p)),
                      "maximum_probability": float(p.max()), "margin": float(np.sort(p)[-1] - np.sort(p)[-2]),
                      "entropy": float(-(p * np.log(p)).sum()), "disagreement": .5,
                      "text_sha256": hashlib.sha256(str(i).encode()).hexdigest(), "uncertainty_rank": i + 1,
                      "local_predictions": {name: {"label": labels[(i + offset) % 3], "probabilities": dict(zip(labels, p))}
                                            for offset, name in enumerate(("svm", "rl", "norberto", "qwen"))}}
        groups[i], folds[i] = str(i // 2), (i // 2) % 5
        if i % 4:
            gpt = gold[i] if i % 2 else labels[(i + 1) % 3]
            examples = [{"label": labels[j], "semantic_cosine": .9 - .03 * j,
                         "lexical_cosine": .4 - .05 * j, "relevance": 1 - .1 * j} for j in range(3)]
            logged[i] = {"stage": "pilot" if i < n // 3 else "confirmation",
                         "arms": {arm: {"label": gpt, "retrieved": examples}
                                  for arm in ("text", "local", "hybrid-3-rag", "hybrid-3-local")}}
    return SimpleNamespace(routing=routing, gold=gold, logged=logged, groups=groups, folds=folds, identities={"synthetic": True})


def test_gold_never_enters_benefit_features_and_gpt_is_post_call_only():
    data = logged_fixture()
    identifier = next(iter(data.logged))
    row, log = data.routing[identifier], data.logged[identifier]["arms"]["hybrid-3-local"]
    before = pre_call_features(row, log["retrieved"])
    row["gold"] = "c1"
    row["true_label"] = "c5"
    assert pre_call_features(row, log["retrieved"]) == before
    assert not any("gpt" in name or "gold" in name or "true_label" in name for name in before)
    after = post_call_features(row, log["label"], log["retrieved"])
    assert "gpt_stack_probability" in after and not any("gold" in name for name in after)
    assert not any(name.startswith("retrieval_") for name in pre_call_features(row, include_retrieval=False))


def test_routing_does_not_see_gpt_answers_or_unlogged_outcomes():
    data = logged_fixture()
    scores = {("pre", "local", .1): {i: (i + 1) / 100 for i in data.routing}}
    policy = Policy("route_local", "route", "local", .1, 0, .1)
    _, called, _ = apply_policy(policy, sorted(data.gold), data, scores)
    assert len(called) <= 9 and called <= set(data.logged)
    modified = copy.deepcopy(data)
    for log in modified.logged.values():
        log["arms"]["hybrid-3-local"]["label"] = "c5"
    _, other_called, _ = apply_policy(policy, sorted(modified.gold), modified, scores)
    assert called == other_called
    with pytest.raises(ValueError, match="groups overlap"):
        assert_disjoint([0], [1], data.groups)


def test_nested_selection_cannot_use_outer_evaluation_labels(tmp_path, monkeypatch):
    from gpt_classifier.codex_backend import CodexBackend
    def forbidden_call(*args, **kwargs):
        raise AssertionError("Offline policy studies cannot call GPT.")
    monkeypatch.setattr(CodexBackend, "create", forbidden_call)
    config = replace(ResearchConfig(), device="cpu", inner_folds=2, regularization=(.1,),
                     benefit_thresholds=(0.,), route_budgets=(.1,))
    data = logged_fixture()
    first = run_acceptance(config, tmp_path / "first", data=data)
    changed = copy.deepcopy(data)
    for identifier in changed.gold:
        if changed.folds[identifier] == 0:
            changed.gold[identifier] = ("c1", "c234", "c5")[(identifier + 1) % 3]
    second = run_acceptance(config, tmp_path / "second", data=changed)
    import json
    selected_first = json.loads((tmp_path / "first/outer-selections.jsonl").read_text().splitlines()[0])
    selected_second = json.loads((tmp_path / "second/outer-selections.jsonl").read_text().splitlines()[0])
    assert selected_first["outer_fold"] == selected_second["outer_fold"] == 0
    assert selected_first["selected"] == selected_second["selected"]
    a = [json.loads(line) for line in (tmp_path / "first/joint_nested_selection.jsonl").read_text().splitlines()]
    b = [json.loads(line) for line in (tmp_path / "second/joint_nested_selection.jsonl").read_text().splitlines()]
    assert [r for r in a if r["outer_fold"] == 0] == [r for r in b if r["outer_fold"] == 0]
    for line in (tmp_path / "first/fit-audit.jsonl").read_text().splitlines():
        record = json.loads(line)
        assert_disjoint(record["training_ids"], record["evaluation_ids"], data.groups)
    assert first["new_gpt_calls"] == second["new_gpt_calls"] == 0


def test_protection_rule_uses_predicted_c234_not_true_class():
    data = logged_fixture()
    row = copy.deepcopy(data.routing[1])
    row["label"] = "c234"
    assert not acceptance_rule("protect_c234", row, "c5", [])
    row["gold"] = "c5"
    assert not acceptance_rule("protect_c234", row, "c5", [])


def test_pre_call_short_circuit_preserves_all_possible_post_rule_outputs():
    data = logged_fixture()
    total_skipped = 0
    for name in ("protect_c234", "two_base_votes", "three_base_votes", "two_examples", "votes_and_example", "protect_and_votes"):
        skipped = 0
        for identifier, log in data.logged.items():
            row, examples = data.routing[identifier], log["arms"]["hybrid-3-local"]["retrieved"]
            if not rule_can_change_prediction(name, row, examples):
                skipped += 1
                for answer in ("c1", "c234", "c5"):
                    final = answer if acceptance_rule(name, row, answer, examples) else row["label"]
                    assert final == row["label"]
        total_skipped += skipped
    assert total_skipped > 0


def test_paired_retrieval_interval_uses_only_audit_and_rejects_missing_ids(tmp_path):
    from rag_research.reporting import paired_retrieval_interval
    from gpt_classifier.data import write_jsonl
    data = logged_fixture()
    proposed, reference = [], []
    for identifier, label in data.gold.items():
        proposed.append({"excel_row": identifier, "selected": [{"label": label}] * 3})
        reference.append({"excel_row": identifier, "selected": [{"label": "c1" if label != "c1" else "c5"}] * 3})
    write_jsonl(tmp_path / "proposed.jsonl", proposed)
    write_jsonl(tmp_path / "reference.jsonl", reference)
    result = paired_retrieval_interval(data, tmp_path, "proposed", "reference", samples=100)
    assert result["hit_at_1_95pct"] == [1., 1.]
    assert result["purity_at_3_95pct"] == [1., 1.]
    removed = next(i for i in data.gold if data.folds[i] in {3, 4})
    write_jsonl(tmp_path / "reference.jsonl", [row for row in reference if row["excel_row"] != removed])
    with pytest.raises(ValueError, match="coverage"):
        paired_retrieval_interval(data, tmp_path, "proposed", "reference", samples=100)
