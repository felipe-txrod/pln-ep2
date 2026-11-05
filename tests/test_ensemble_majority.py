"""Exercise real majority routing, provenance, and exploratory scoring boundaries."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ensemble_classifier.cli import main
from ensemble_classifier.comparison import compare_majority, majority_predictions
from ensemble_classifier.config import EnsembleConfig
from ensemble_classifier.files import read_json, read_predictions
from ensemble_classifier.pipeline import combine
from gpt_classifier.data import LABELS, Text, digest, file_digest, write_json, write_jsonl


@pytest.fixture
def voting_study(make_xlsx, tmp_path):
    texts = [f"response {index}" for index in range(4)]
    source = make_xlsx("source.xlsx", [(text, None) for text in texts])
    gold = make_xlsx("gold.xlsx", list(zip(texts, ("c1", "c234", "c5", "c5"))))
    records = [Text(index + 2, text) for index, text in enumerate(texts)]
    labels = {"norberto": ["c1", "c1", "c1", "c234"],
              "svm": ["c1", "c234", "c234", "c5"],
              "rl": ["c5", "c234", "c5", "c5"]}
    votes = {name: [{"excel_row": row.excel_row, "text_sha256": digest(row.text), "label": label}
                    for row, label in zip(records, choices)] for name, choices in labels.items()}
    members = {}
    for index, (name, rows) in enumerate(votes.items()):
        directory = tmp_path / name
        directory.mkdir()
        artifact = digest(name)
        write_json(directory / "run.json", {"status": "completed", "labels": list(LABELS),
                   "input": {"sha256": file_digest(source)}, "artifact_sha256": artifact})
        write_jsonl(directory / "predictions.jsonl", rows)
        manifest = directory / "frozen_manifest.json"
        write_json(manifest, {"artifact_sha256": artifact})
        members[name] = {"run": str(directory), "manifest": str(manifest), "identity_field": "artifact_sha256"}
    config = EnsembleConfig(tuple(members), members, "majority", None, "rl")
    return config, source, gold, records, votes


@pytest.mark.parametrize("tiebreaker,third", [("rl", "c5"), ("svm", "c234"), ("norberto", "c1")])
def test_strict_majority_wins_and_only_three_way_tie_uses_arbitrator(voting_study, tiebreaker, third):
    config, _, _, records, votes = voting_study
    rows = majority_predictions(records, votes, config.models, tiebreaker)
    assert [row["label"] for row in rows] == ["c1", "c234", third, "c5"]
    assert [row["arbitrated"] for row in rows] == [False, False, True, False]
    assert rows[2]["decision_source"] == tiebreaker
    assert all("gold" not in row for row in rows)


def test_two_member_disagreement_uses_fixed_external_arbitrator(voting_study):
    _, _, _, records, votes = voting_study
    rows = majority_predictions(records, votes, ("norberto", "svm"), "rl")
    assert [row["label"] for row in rows] == ["c1", "c234", "c5", "c5"]
    assert rows[2]["label"] not in rows[2]["member_labels"].values()


def test_voting_rejects_missing_members_and_reordered_predictions(voting_study):
    config, _, _, records, votes = voting_study
    with pytest.raises(ValueError, match="Missing predictions"):
        majority_predictions(records, votes, config.models, "unknown")
    with pytest.raises(ValueError, match="distinct"):
        majority_predictions(records, votes, ("svm", "svm"), "svm")
    votes["svm"].reverse()
    with pytest.raises(ValueError, match="alignment"):
        majority_predictions(records, votes, config.models, "rl")


def test_comparison_persists_all_candidates_before_gold_and_keeps_defaults(voting_study, tmp_path, monkeypatch):
    import ensemble_classifier.comparison as comparison

    config, source, gold, _, _ = voting_study
    output = tmp_path / "comparison"
    original_reader = comparison.read_xlsx
    gold_reads = []

    def guarded_reader(path, **kwargs):
        if kwargs.get("labeled"):
            gold_reads.append(path)
            assert len(list(output.glob("*/predictions.jsonl"))) == 15
            assert read_json(output / "run.json")["status"] == "predictions_completed"
            for candidate in output.glob("*/predictions.jsonl"):
                assert all("gold" not in row for row in read_predictions(candidate))
        return original_reader(path, **kwargs)

    monkeypatch.setattr(comparison, "read_xlsx", guarded_reader)
    report = compare_majority(config, source, gold, output)
    assert gold_reads == [gold]
    assert report["status"] == "completed" and report["phase"] == "exploratory"
    assert report["default_configuration_changed"] is False
    assert len(report["candidates"]) == 15
    assert report["best_full_ensemble_by_accuracy"] == ["norberto-svm-rl__tie-rl"]
    winner = report["candidates"][0]
    assert winner["correct"] == 4 and winner["accuracy"] == 1.0
    full = next(row for row in report["candidates"] if row["id"] == "norberto-svm-rl__tie-rl")
    assert full["routed_rows"] == 1
    assert EnsembleConfig.load(output / full["id"] / "config.json").models == config.models
    with pytest.raises(FileExistsError):
        compare_majority(config, source, gold, output)


def test_misaligned_gold_cannot_complete_comparison(voting_study, make_xlsx, tmp_path):
    config, source, _, _, _ = voting_study
    bad_gold = make_xlsx("misaligned.xlsx", [("different", "c1") for _ in range(4)])
    output = tmp_path / "bad-comparison"
    with pytest.raises(ValueError, match="exactly match"):
        compare_majority(config, source, bad_gold, output)
    assert read_json(output / "run.json")["status"] == "predictions_completed"
    assert not (output / "summary.json").exists()


def test_comparison_rejects_nonmajority_and_cli_runs_offline(voting_study, tmp_path, capsys):
    config, source, gold, _, _ = voting_study
    with pytest.raises(ValueError, match="majority"):
        compare_majority(replace(config, strategy="consensus"), source, gold, tmp_path / "invalid")
    path = tmp_path / "config.json"
    write_json(path, {"models": list(config.models), "members": config.members,
                     "strategy": "majority", "tiebreaker": "rl"})
    assert main(["compare-majority", "--config", str(path), "--input", str(source),
                 "--gold", str(gold), "--output-dir", str(tmp_path / "cli-comparison")]) == 0
    assert '"phase": "exploratory"' in capsys.readouterr().out


def test_official_combine_still_requires_test2_before_test1(voting_study, tmp_path, monkeypatch):
    import ensemble_classifier.pipeline as pipeline

    config, source, _, _, _ = voting_study
    real_digest = pipeline.file_digest
    official = Path(pipeline.__file__).resolve().parents[2] / "data/test1.xlsx"
    monkeypatch.setattr(pipeline, "file_digest", lambda path: file_digest(source) if path == official else real_digest(path))
    with pytest.raises(ValueError, match="test2 run"):
        combine(config, source, tmp_path / "official-output")
    assert not (tmp_path / "official-output").exists()
