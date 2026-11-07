"""Synthetic isolation, search-budget, caching and reproducibility checks."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gpt_classifier.data import Example, LABELS, digest
from svm_research.data import partition_indices
from svm_research.search import (baseline_candidate, broad_candidates, build_pipeline, candidate_id,
                                 fit_fold, grouped_folds, refined_candidates, select_one_se)
from svm_research.search import search_phase
from ensemble_classifier.files import read_json


class SvmResearchTests(unittest.TestCase):
    def samples(self):
        records = [Example(2 + i, f"sampleunique{i} common words response {LABELS[i % 3]} topic{i % 7}", LABELS[i % 3])
                   for i in range(90)]
        return (np.asarray([r.text for r in records], dtype=object), np.asarray([r.label for r in records]),
                np.asarray([r.excel_row for r in records]), np.asarray([str(i) for i in range(90)]))

    def test_budget_and_coverage(self):
        candidates = broad_candidates()
        self.assertEqual(len(candidates), 256)
        self.assertEqual(len({candidate_id(c) for c in candidates}), 256)
        self.assertEqual({c["family"] for c in candidates}, {"word", "char", "char_wb", "word_char_wb"})
        self.assertIn(baseline_candidate(), candidates)
        for family in {c["family"] for c in candidates}:
            selected = [c for c in candidates if c["family"] == family]
            self.assertEqual(len(selected), 64)
            self.assertEqual({c["C"] for c in selected}, {.005, .05, .5, 5.})
            self.assertEqual({c["class_weight"] for c in selected}, {None, "balanced"})
            self.assertEqual({v["norm"] for c in selected for v in c["vectors"].values()}, {"l1", "l2", None})

    def test_original_recipe(self):
        model = build_pipeline(baseline_candidate())
        features = dict(model.named_steps["features"].transformer_list)
        self.assertEqual(features["word"].ngram_range, (1, 2))
        self.assertEqual(features["char"].analyzer, "char_wb")
        self.assertEqual(features["char"].ngram_range, (3, 5))
        self.assertIsNone(model.named_steps["features"].transformer_weights)
        self.assertEqual(model.named_steps["classifier"].C, .05)

    def test_vocabulary_is_fit_on_training_only(self):
        texts, labels, _, groups = self.samples()
        train, valid = grouped_folds(labels, groups, 3, 42)[0]
        candidate = next(c for c in broad_candidates() if c["family"] == "word" and c["vectors"]["word"]["min_df"] == 1)
        model = build_pipeline(candidate).fit(texts[train], labels[train])
        vocabulary = dict(model.named_steps["features"].transformer_list)["word"].vocabulary_
        for i in valid:
            self.assertNotIn(f"sampleunique{i}", vocabulary)
        self.assertTrue(any(f"sampleunique{i}" in vocabulary for i in train))

    def test_grouped_folds_preserve_duplicate_components(self):
        _, y, _, _ = self.samples()
        groups = np.asarray([str(i // 3) for i in range(len(y))])
        folds = grouped_folds(y, groups, 5, 42)
        covered = []
        for train, valid in folds:
            self.assertFalse(set(groups[train]) & set(groups[valid]))
            covered.extend(valid.tolist())
        self.assertEqual(sorted(covered), list(range(len(y))))

    def test_partition_rejects_duplicates_crossing_roles(self):
        records = [Example(2, "Same text", "c1"), Example(3, " same  TEXT ", "c5")]
        partition = [{"excel_row": r.excel_row, "text_sha256": digest(r.text), "group_id": str(i),
                      "role": role} for i, (r, role) in enumerate(zip(records, ("calibration", "weight_selection")))]
        with self.assertRaisesRegex(ValueError, "crosses holdout roles"):
            partition_indices(records, np.asarray(["0", "1"]), partition)

    def test_partition_rejects_missing_and_changed_ids(self):
        with self.assertRaisesRegex(ValueError, "missing holdout"):
            partition_indices([Example(2, "text", "c1")], ["0"], [])
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            partition_indices([Example(2, "text", "c1")], ["0"],
                              [{"excel_row": 3, "text_sha256": digest("text"), "group_id": "0", "role": "calibration"}])

    def make_trial(self, c, mean, std=.01, eligible=True):
        candidate = {**baseline_candidate(), "C": c}
        return {"candidate": candidate, "candidate_id": candidate_id(candidate), "mean_f1_macro": mean,
                "std_f1_macro": std, "folds": [{}, {}, {}, {}, {}], "mean_feature_count": 100,
                "eligible": eligible}

    def test_one_se_prefers_regularization_and_excludes_nonconvergence(self):
        trials = [self.make_trial(.05, .500), self.make_trial(.005, .499), self.make_trial(.001, .7, eligible=False)]
        self.assertEqual(select_one_se(trials)["candidate"]["C"], .005)
        self.assertEqual(refined_candidates(trials)[0], baseline_candidate())
        self.assertLessEqual(len(refined_candidates(trials)), 61)

    def test_hinge_uses_compatible_solver(self):
        model = build_pipeline({**baseline_candidate(), "loss": "hinge"})
        self.assertTrue(model.named_steps["classifier"].dual)

    def test_fold_resume_checks_hash_and_skips_fitting(self):
        x, y, rows, g = self.samples()
        train, valid = grouped_folds(y, g, 3, 42)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = (baseline_candidate(), 0, train, valid, x, y, rows, g, root / "fold", root / "cache", 42)
            first = fit_fold(*args)
            self.assertEqual(first["shared_groups"], 0)
            with patch("svm_research.search.build_pipeline", side_effect=AssertionError("Unexpected refit")):
                self.assertEqual(first, fit_fold(*args))
            (root / "fold/fold-0.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "modified"):
                fit_fold(*args)

    def test_fold_refuses_group_leakage(self):
        x, y, rows, groups = self.samples()
        groups[1] = groups[0]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "crosses training"):
                fit_fold(baseline_candidate(), 0, np.asarray([0]), np.asarray([1]), x, y, rows, groups,
                         Path(directory) / "fold", Path(directory) / "cache", 42)

    def test_parallel_phase_records_and_resumes_all_folds(self):
        x, y, rows, groups = self.samples()
        data = SimpleNamespace(training=[Example(int(r), t, label) for r, t, label in zip(rows, x, y)], groups=groups)
        config = {"seed": 42, "phase_a_folds": 3, "jobs": 2, "cache_gib": 1}
        candidates = [baseline_candidate(), {**baseline_candidate(), "C": .005}]
        with tempfile.TemporaryDirectory() as directory, patch("svm_research.search.broad_candidates", return_value=candidates):
            result = search_phase(data, config, Path(directory), "a")
            self.assertEqual(len(result["trials"]), 2)
            self.assertEqual(sum(len(t["folds"]) for t in result["trials"]), 6)
            self.assertFalse(result["test1_accessed"])
            self.assertEqual(result, search_phase(data, config, Path(directory), "a"))
            self.assertEqual(read_json(Path(directory) / "phase-a/progress.json")["completed"], 2)


if __name__ == "__main__":
    unittest.main()
