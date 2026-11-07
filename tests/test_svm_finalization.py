"""Test grouped calibration/weight isolation, paired inference and late final-gold access."""
from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from ensemble_classifier.soft_voting import FrozenScores
from gpt_classifier.data import LABELS, Example, Text, digest, file_digest, write_json, write_jsonl
from svm_finalization import methods, pipeline
from svm_research.search import baseline_candidate, grouped_folds


class FixedMargins:
    def __init__(self, margins):
        self.margins = margins
        self.classes_ = np.asarray(LABELS)

    def decision_function(self, texts):
        if len(texts) != len(self.margins):
            raise ValueError("Unexpected target coverage.")
        return self.margins


class FinalizationTests(unittest.TestCase):
    def sample(self, n=120):
        generator = np.random.default_rng(42)
        labels = np.asarray([LABELS[i % 3] for i in range(n)])
        groups = np.asarray([str(i // 3) for i in range(n)])
        margins = generator.normal(size=(n, 3))
        return margins, labels, groups

    def test_portable_calibration_matches_sklearn_on_held_scores(self):
        scores, labels, _ = self.sample()
        held, _, _ = self.sample(30)
        held = held * 20
        for method in ("sigmoid", "isotonic"):
            fitted = methods.fit_calibration(scores, labels, method)
            expected = CalibratedClassifierCV(FrozenEstimator(FrozenScores()), method=method).fit(scores, labels)
            np.testing.assert_allclose(methods.apply_calibrator(held, fitted), expected.predict_proba(held), atol=1e-12, rtol=0)

    def test_isotonic_zero_sum_and_invalid_mapping(self):
        mapping = {"method": "isotonic_ovr", "x": [[0., 1.]] * 3, "y": [[0., 0.]] * 3}
        np.testing.assert_allclose(methods.apply_calibrator(np.zeros((2, 3)), mapping), 1 / 3)
        mapping["x"] = [[1., 0.]] * 3
        with self.assertRaisesRegex(ValueError, "Invalid monotonic"):
            methods.apply_calibrator(np.zeros((2, 3)), mapping)

    def test_calibration_policy_has_nested_group_isolation(self):
        scores, labels, groups = self.sample()
        _, report, oof = methods.nested_calibration(scores, labels, groups)
        seen = []
        for outer in report["outer_folds"]:
            train, valid = np.asarray(outer["training_indices"]), np.asarray(outer["validation_indices"])
            self.assertFalse(set(groups[train]) & set(groups[valid]))
            assignments = np.asarray(outer["inner_selection"]["fold_assignments"])
            for fold in range(3):
                self.assertFalse(set(groups[train[assignments == fold]]) & set(groups[train[assignments != fold]]))
            seen.extend(valid.tolist())
        self.assertEqual(sorted(seen), list(range(len(labels))))
        for probabilities in oof.values():
            self.assertTrue(np.isfinite(probabilities).all())
            np.testing.assert_allclose(probabilities.sum(axis=1), 1)

    def test_sigmoid_is_default_without_mean_improvement(self):
        scores, labels, groups = self.sample()
        with patch("svm_finalization.methods.fit_calibration", return_value={}), \
             patch("svm_finalization.methods.apply_calibrator", side_effect=lambda x, _: np.full((len(x), 3), 1 / 3)):
            method, check = methods.choose_calibration(scores, labels, groups)
        self.assertEqual(method, "sigmoid")
        self.assertAlmostEqual(check["paired_log_loss_gain"], 0)

    def test_nested_weights_are_chosen_only_on_outer_training_rows(self):
        scores, labels, groups = self.sample()
        svm = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
        norberto = np.full_like(svm, 1 / 3)
        original = methods.choose_weight
        with patch("svm_finalization.methods.choose_weight", wraps=original) as choices:
            _, report, oof = methods.nested_weights(svm, norberto, labels, groups)
        self.assertEqual(len(choices.call_args_list), 6)
        for call, outer in zip(choices.call_args_list[:5], report["outer_folds"]):
            train, valid = outer["training_indices"], outer["validation_indices"]
            np.testing.assert_array_equal(call.args[2], labels[train])
            np.testing.assert_array_equal(call.args[3], groups[train])
            self.assertFalse(set(call.args[3]) & set(groups[valid]))
        np.testing.assert_allclose(oof.sum(axis=1), 1)

    def test_weight_rule_prefers_original_anchor_for_equal_models(self):
        _, labels, groups = self.sample()
        p = np.full((len(labels), 3), 1 / 3)
        weight, _ = methods.choose_weight(p, p, labels, groups)
        self.assertEqual(weight, .75)

    def test_paired_group_bootstrap_identical_models_has_zero_delta(self):
        _, labels, groups = self.sample()
        result = methods.paired_statistics(labels, {"A": labels.copy(), "B": labels.copy()}, groups, repeats=100)
        pair = result["pairs"]["B_minus_A"]
        self.assertEqual(pair["corrected"], 0)
        self.assertEqual(pair["harmed"], 0)
        self.assertEqual(pair["delta_95_interval"]["f1_macro"], [0., 0.])
        self.assertEqual(pair["mcnemar_exact_p_descriptive"], 1.)

    def test_paired_corrected_and_harmed_counts(self):
        labels = np.asarray(["c1", "c234", "c5", "c1", "c234", "c5"])
        left = np.asarray(["c1", "c234", "c1", "c5", "c234", "c5"])
        right = np.asarray(["c234", "c234", "c5", "c1", "c234", "c5"])
        pair = methods.paired_statistics(labels, {"A": left, "B": right}, np.arange(6), repeats=100)["pairs"]["B_minus_A"]
        self.assertEqual((pair["corrected"], pair["harmed"], pair["net_correct"]), (2, 1, 1))
        self.assertAlmostEqual(pair["delta"]["accuracy"], 1 / 6)

    def test_identical_ensemble_cannot_pass_adoption_screen(self):
        _, labels, groups = self.sample()
        p = np.full((len(labels), 3), 1 / 3)
        assignments = np.full(len(labels), -1)
        for fold, (_, valid) in enumerate(grouped_folds(labels, groups, 5, 42)):
            assignments[valid] = fold
        screen = methods.adoption_screen(labels, p, p, groups, assignments)
        self.assertFalse(screen["passed"])
        self.assertFalse(screen["checks"]["positive_bootstrap_lower_bound"])

    def test_final_refit_and_resume_validate_training_membership(self):
        records = [Example(i + 2, f"common response {LABELS[i % 3]} uniqueitem{i} topic{i % 5}", LABELS[i % 3]) for i in range(90)]
        holdout = [Example(i + 200, f"reserved response topic{i % 5}", LABELS[i % 3]) for i in range(30)]
        data = SimpleNamespace(training=records, holdout=holdout)
        plan = {"selected_candidate": {"candidate": baseline_candidate()}, "refit_wall_limit_seconds": 15}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            margins, _ = pipeline.ensure_refit(plan, data, root)
            self.assertEqual(margins.shape, (30, 3))
            with patch("svm_finalization.pipeline.mp.get_context", side_effect=AssertionError("Unexpected refit")):
                again, _ = pipeline.ensure_refit(plan, data, root)
                np.testing.assert_array_equal(again, margins)
            data.training[0] = Example(2, "changed", "c1")
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                pipeline.ensure_refit(plan, data, root)

    def test_final_inputs_cannot_open_before_freeze(self):
        with patch("svm_finalization.pipeline.load_frozen", side_effect=ValueError("not frozen")), \
             patch("svm_finalization.pipeline.read_xlsx", side_effect=AssertionError("Final input opened")) as reader:
            with self.assertRaisesRegex(ValueError, "not frozen"):
                pipeline.predict_test1(Path("unused"))
            reader.assert_not_called()

    def test_gold_opens_only_after_all_eight_saved_predictions(self):
        matrix = [[139, 93, 68], [74, 135, 91], [50, 77, 173]]
        records = [Text(i + 2, f"synthetic target {i}") for i in range(900)]
        gold = [Example(r.excel_row, r.text, LABELS[i // 300]) for i, r in enumerate(records)]
        labels = [label for row in matrix for label, count in zip(LABELS, row) for _ in range(count)]
        p = np.full((900, 3), .1)
        for i, label in enumerate(labels):
            p[i, LABELS.index(label)] = .8
        margins = np.log(p / (1 - p))
        calibrator = {"method": "sigmoid_ovr", "a": [-1., -1., -1.], "b": [0., 0., 0.]}
        source = {"sha256": digest("synthetic input"), "rows": 900}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, bundle = root / "historical", root / "baseline"
            write_json(bundle / "model.json", {})
            baseline_hash = file_digest(bundle / "model.json")
            ensemble_rows = [{"excel_row": r.excel_row, "text_sha256": digest(r.text), "label": label,
                              "probabilities": dict(zip(LABELS, values.tolist()))} for r, label, values in zip(records, labels, p)]
            write_jsonl(old / "predictions.jsonl", ensemble_rows)
            write_json(old / "run.json", {"artifact_sha256": baseline_hash, "input": source, "gold_accessed": False,
                                         "predictions_sha256": file_digest(old / "predictions.jsonl")})
            for name, scores, field in (("svm", margins, "decision_scores"), ("norberto", p, "probabilities")):
                member = old / "members" / name
                write_jsonl(member / "predictions.jsonl", [{"excel_row": r.excel_row, "text_sha256": digest(r.text), "label": label,
                            field: dict(zip(LABELS, values.tolist()))} for r, label, values in zip(records, labels, scores)])
                write_json(member / "run.json", {"status": "completed", "gold_accessed": False, "training_performed": False,
                           "input": source, "artifact_sha256": digest(name), "labels": list(LABELS),
                           "predictions_sha256": file_digest(member / "predictions.jsonl")})
            write_json(root / "plan.json", {"test1_input": str(root / "input.xlsx"), "test1_gold": str(root / "gold.xlsx"),
                       "historical_test1_run": str(old), "search_config_value": {"baseline": str(bundle)}})
            write_json(root / "frozen.json", {"test_fixture": True})
            (root / "gold.xlsx").write_text("synthetic gold fixture")
            saved_plan = json.loads((root / "plan.json").read_text())
            saved_plan["test1_expected_gold_sha256"] = file_digest(root / "gold.xlsx")
            write_json(root / "plan.json", saved_plan)
            joblib.dump(FixedMargins(margins), root / "model.joblib")
            frozen = {"recommended_configuration": "A", "weights": {name: {"svm": .75, "norberto": .25} for name in ("A", "B", "C")},
                      "svm_calibrator": calibrator,
                      "original_baseline": {"model_identities": {name: {"artifact_sha256": digest(name)} for name in ("svm", "norberto")},
                                            "calibration": {"svm": calibrator, "norberto": {"method": "identity"}}}}
            events = []
            def frozen_gate(_):
                events.append("frozen")
                return frozen
            def reader(path, *, labeled=False):
                self.assertTrue(events and events[0] == "frozen")
                if not labeled:
                    return records, source
                for name in ("A", "B", "C", "svm_original_calibrated", "svm_optimized_calibrated", "norberto_unchanged",
                             "svm_original_native", "svm_optimized_native"):
                    self.assertTrue((root / "test1" / f"{name}.jsonl").is_file())
                self.assertTrue((root / "test1/prediction-manifest.json").is_file())
                events.append("gold")
                return gold, {"sha256": file_digest(path), "rows": 900}
            with patch("svm_finalization.pipeline.load_frozen", side_effect=frozen_gate), \
                 patch("svm_finalization.pipeline.read_xlsx", side_effect=reader), \
                 patch("svm_finalization.reporting.report"):
                result = pipeline.evaluate_test1(root)
            self.assertIn("gold", events)
            self.assertTrue(result["gold_read_after_all_predictions"])
            self.assertFalse(result["recommendation_changed_after_test1"])
            self.assertAlmostEqual(result["metrics"]["A"]["f1_macro"], .49584421663197825)


if __name__ == "__main__":
    unittest.main()
