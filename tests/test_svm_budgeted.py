"""Verify reduced budgets, exact reuse, sparse-pipeline parity and hard timeouts."""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np

from ensemble_classifier.files import read_json
from gpt_classifier.data import LABELS, Example, write_json
from svm_research.search import baseline_candidate, broad_candidates, build_pipeline, candidate_id, fit_fold, grouped_folds
from svm_budgeted.cli import import_folds
from svm_budgeted.engine import (aggregate, bounded_fit, candidates_a, candidates_b, feature_cache,
                                  identity, representation_fold, search, validate_fold)


def blocking_fit(*args):
    time.sleep(30)


class BudgetedSvmTests(unittest.TestCase):
    def samples(self):
        records = [Example(i + 2, f"uniqueitem{i} common response topic{i % 7} {LABELS[i % 3]} words", LABELS[i % 3])
                   for i in range(90)]
        return (records, np.asarray([r.text for r in records], dtype=object),
                np.asarray([r.label for r in records]), np.asarray([r.excel_row for r in records]),
                np.asarray([str(i // 3) for i in range(90)]))

    def test_reduced_design_is_subset_and_keeps_all_families(self):
        candidates = candidates_a()
        original = {candidate_id(c) for c in broad_candidates()}
        self.assertEqual(len(candidates), 96)
        self.assertTrue({candidate_id(c) for c in candidates} <= original)
        self.assertIn(baseline_candidate(), candidates)
        self.assertEqual({c["family"] for c in candidates}, {"word", "char", "char_wb", "word_char_wb"})
        self.assertFalse(any(v["norm"] is None for c in candidates for v in c["vectors"].values()))
        self.assertEqual({v["norm"] for c in candidates for v in c["vectors"].values()}, {"l1", "l2"})
        self.assertEqual({v["binary"] for c in candidates for v in c["vectors"].values()}, {True, False})

    def test_refinement_respects_smaller_budget_and_two_representations(self):
        trials = [{"candidate": c, "candidate_id": candidate_id(c), "eligible": True, "mean_f1_macro": .5}
                  for c in candidates_a()]
        candidates = candidates_b(trials)
        self.assertLessEqual(len(candidates), 25)
        self.assertIn(baseline_candidate(), candidates)
        self.assertIn("hinge", {c["loss"] for c in candidates})

    def test_shared_features_match_original_pipeline(self):
        _, texts, labels, rows, groups = self.samples()
        train, valid = grouped_folds(labels, groups, 3, 42)[0]
        candidate = baseline_candidate()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, meta = feature_cache(candidate, 0, train, valid, texts, labels, rows, groups, root / "cache", 42)
            matrices = joblib.load(path, mmap_mode="r")
            original = build_pipeline(candidate).fit(texts[train], labels[train])
            independent = build_pipeline(candidate).named_steps["classifier"].fit(matrices["training"], labels[train])
            np.testing.assert_allclose(original.decision_function(texts[valid]),
                                       independent.decision_function(matrices["validation"]), atol=1e-12, rtol=1e-12)
            np.testing.assert_array_equal(original.predict(texts[valid]), independent.predict(matrices["validation"]))
            with patch("svm_budgeted.engine.build_pipeline", side_effect=AssertionError("Unexpected TF-IDF refit")):
                self.assertEqual((path, meta), feature_cache(candidate, 0, train, valid, texts, labels, rows, groups, root / "cache", 42))

    def test_features_never_fit_validation_text(self):
        _, texts, labels, rows, groups = self.samples()
        train, valid = grouped_folds(labels, groups, 3, 42)[0]
        candidate = next(c for c in candidates_a() if c["family"] == "word" and c["vectors"]["word"]["min_df"] == 1)
        model = build_pipeline(candidate)
        vectorizer = dict(model.named_steps["features"].transformer_list)["word"]
        with tempfile.TemporaryDirectory() as directory, patch("svm_budgeted.engine.build_pipeline", return_value=model):
            feature_cache(candidate, 0, train, valid, texts, labels, rows, groups, Path(directory), 42)
            for i in valid:
                self.assertNotIn(f"uniqueitem{i}", vectorizer.vocabulary_)

    def test_hard_timeout_is_terminal_without_predictions(self):
        _, texts, labels, rows, groups = self.samples()
        train, valid = grouped_folds(labels, groups, 3, 42)[0]
        expected = identity(baseline_candidate(), 0, train, valid, rows, 42)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            started = time.monotonic()
            result = bounded_fit(baseline_candidate(), expected, root / "unused", labels[train], labels[valid], rows[valid],
                                 root, {}, 1, target=blocking_fit)
            self.assertEqual(result["status"], "timeout")
            self.assertLess(time.monotonic() - started, 8)
            self.assertIsNone(result["metrics"])
            self.assertFalse((root / "fold-0.npz").exists())
            self.assertEqual(result, validate_fold(root, expected))
            self.assertFalse(aggregate(baseline_candidate(), [result, result, result])["eligible"])

    def test_actual_subprocess_and_resume(self):
        _, texts, labels, rows, groups = self.samples()
        train, valid = grouped_folds(labels, groups, 3, 42)[0]
        candidate = baseline_candidate()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, meta = feature_cache(candidate, 0, train, valid, texts, labels, rows, groups, root / "cache", 42)
            expected = identity(candidate, 0, train, valid, rows, 42)
            result = bounded_fit(candidate, expected, path, labels[train], labels[valid], rows[valid], root / "trials", meta, 15)
            self.assertEqual(result["status"], "completed")
            with patch("svm_budgeted.engine.mp.get_context", side_effect=AssertionError("Unexpected process restart")):
                self.assertEqual(result, bounded_fit(candidate, expected, path, labels[train], labels[valid], rows[valid], root / "trials", meta, 15))

    def test_source_reuse_reconstructs_metrics_and_rejects_tampering(self):
        records, texts, labels, rows, groups = self.samples()
        folds = grouped_folds(labels, groups, 3, 42)
        candidate = baseline_candidate()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "original", root / "amended"
            write_json(source / "phase-a/plan.json", {"seed": 42, "folds": 3, "candidates": [candidate]})
            train, valid = folds[0]
            old_root = source / "phase-a/trials" / candidate_id(candidate)
            fit_fold(candidate, 0, train, valid, texts, labels, rows, groups, old_root, root / "old-cache", 42)
            data = SimpleNamespace(training=records, groups=groups)
            result = import_folds(source, output, data)
            self.assertEqual(len(result["imported_folds"]), 1)
            self.assertEqual((old_root / "fold-0.json").read_bytes(),
                             (output / "phase-a/trials" / candidate_id(candidate) / "fold-0.json").read_bytes())
            old = read_json(old_root / "fold-0.json")
            old["metrics"]["f1_macro"] = .99
            write_json(old_root / "fold-0.json", old)
            with self.assertRaisesRegex(ValueError, "metrics do not match"):
                import_folds(source, root / "rejected", data)

    def test_parallel_engine_runs_and_resumes(self):
        records, _, _, _, groups = self.samples()
        data = SimpleNamespace(training=records, groups=groups)
        config = {"seed": 42, "phase_a_folds": 3, "jobs": 4, "fit_timeout_seconds": 15}
        word = next(c for c in candidates_a() if c["family"] == "word")
        candidates = [baseline_candidate(), {**baseline_candidate(), "C": .005}, word, {**word, "C": .05}]
        with tempfile.TemporaryDirectory() as directory, patch("svm_budgeted.engine.candidates_a", return_value=candidates):
            root = Path(directory)
            result = search(data, config, root, "a")
            self.assertEqual(result["status"], "completed")
            self.assertTrue(all(t["eligible"] for t in result["trials"]))
            self.assertEqual(result, search(data, config, root, "a"))
            self.assertEqual(read_json(root / "phase-a/progress.json")["terminal_folds"], 12)


if __name__ == "__main__":
    unittest.main()
