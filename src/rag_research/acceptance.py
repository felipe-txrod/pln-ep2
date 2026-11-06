"""Nested grouped evaluation of benefit routing and post-call acceptance on logged support."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gpt_classifier.data import LABELS, canonical, digest, file_digest, write_json, write_jsonl
from .data import load_data
from .interventions import acceptance_rule, historical_analysis, intervention_utility, post_call_features, pre_call_features, rule_can_change_prediction


@dataclass(frozen=True)
class Policy:
    family: str
    phase: str
    features: str = "retrieval"
    regularization: float = .1
    threshold: float = 0.
    budget: float = .1
    rule: str | None = None

    @property
    def name(self):
        return canonical(asdict(self))


def policy_families(config):
    none, all_answers = Policy("baseline", "none"), Policy("unconditional", "all")
    families = {"rules": [none, all_answers] + [Policy("rules", "rule", rule=name) for name in
                ("protect_c234", "two_base_votes", "three_base_votes", "two_examples", "votes_and_example", "protect_and_votes")]}
    for features in ("local", "retrieval"):
        for phase in ("accept", "route"):
            name = phase + "_" + features
            families[name] = [none, all_answers] + [Policy(name, phase, features, c, threshold, budget)
                for c in config.regularization for threshold in config.benefit_thresholds
                for budget in (config.route_budgets if phase == "route" else (.1,))]
    families["cascade"] = [none, all_answers] + [Policy("cascade", "cascade", "retrieval", c, threshold, budget)
        for c in config.regularization for threshold in config.benefit_thresholds for budget in config.route_budgets]
    return families


def assert_disjoint(train, test, groups):
    if set(train) & set(test) or {groups[i] for i in train} & {groups[i] for i in test}:
        raise ValueError("Training and evaluation similarity groups overlap.")


def metrics(gold, predictions):
    if set(gold) != set(predictions) or not set(predictions.values()) <= set(LABELS):
        raise ValueError("Metric IDs or predicted classes are invalid.")
    matrix = np.zeros((3, 3), dtype=int)
    for identifier in gold:
        matrix[LABELS.index(gold[identifier]), LABELS.index(predictions[identifier])] += 1
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2 * np.diag(matrix), denominator, out=np.zeros(3), where=denominator != 0)
    return {"rows": len(gold), "correct": int(matrix.trace()), "accuracy": float(matrix.trace() / len(gold)),
            "f1_macro": float(f1.mean()), "f1_per_class": dict(zip(LABELS, f1.tolist())),
            "confusion_matrix": matrix.tolist()}


class UtilityEstimator:
    """Estimate P(corrected)-P(harmed), without class balancing that changes priors."""

    def __init__(self, regularization, seed):
        self.regularization, self.seed = regularization, seed
        self.pipeline, self.constant = None, 0.
        self.fitted_ids = []

    def fit(self, features, utilities, identifiers):
        self.fitted_ids = list(identifiers)
        y = np.array([utilities[i] for i in identifiers], dtype=int)
        if len(y) == 0 or len(set(y)) < 2:
            self.constant = float(y.mean()) if len(y) else 0.
            return self
        if not all(np.isfinite(list(features[i].values())).all() for i in identifiers):
            raise ValueError("Nonfinite benefit features.")
        self.pipeline = make_pipeline(DictVectorizer(sparse=False), StandardScaler(),
                                      LogisticRegression(C=self.regularization, max_iter=1000, tol=1e-5, random_state=self.seed))
        self.pipeline.fit([features[i] for i in identifiers], y)
        if max(self.pipeline[-1].n_iter_) >= self.pipeline[-1].max_iter:
            raise RuntimeError("Benefit model did not converge.")
        return self

    def predict(self, features, identifiers):
        if not identifiers:
            return {}
        if self.pipeline is None:
            return {i: self.constant for i in identifiers}
        probabilities = self.pipeline.predict_proba([features[i] for i in identifiers])
        classes = self.pipeline[-1].classes_
        utility = np.zeros(len(identifiers))
        for index, label in enumerate(classes):
            utility += label * probabilities[:, index]
        if not np.isfinite(utility).all():
            raise ValueError("Nonfinite expected-benefit scores.")
        return dict(zip(identifiers, utility.tolist()))


def feature_tables(data):
    result = {}
    for mode in ("pre", "post"):
        for feature_set in ("local", "retrieval"):
            features = {}
            for identifier, log in data.logged.items():
                context = log["arms"]["hybrid-3-local"]
                args = (data.routing[identifier], context["retrieved"])
                features[identifier] = pre_call_features(*args, include_retrieval=feature_set == "retrieval") if mode == "pre" else post_call_features(
                    data.routing[identifier], context["label"], context["retrieved"], include_retrieval=feature_set == "retrieval")
            result[mode, feature_set] = features
    return result


def fitted_scores(data, tables, train, test, config, fit_audit):
    assert_disjoint(train, test, data.groups)
    training = sorted(set(train) & set(data.logged))
    evaluation = sorted(set(test) & set(data.logged))
    utilities = {i: intervention_utility(data.routing[i]["label"], data.logged[i]["arms"]["hybrid-3-local"]["label"], data.gold[i]) for i in training}
    result = {}
    for mode in ("pre", "post"):
        ids = training if mode == "pre" else [i for i in training if data.routing[i]["label"] != data.logged[i]["arms"]["hybrid-3-local"]["label"]]
        for feature_set in ("local", "retrieval"):
            for c in config.regularization:
                estimator = UtilityEstimator(c, config.seed).fit(tables[mode, feature_set], utilities, ids)
                if set(estimator.fitted_ids) & set(test):
                    raise ValueError("Benefit fitting accessed held-out targets.")
                result[mode, feature_set, c] = estimator.predict(tables[mode, feature_set], evaluation)
                fit_audit.append({"mode": mode, "features": feature_set, "regularization": c,
                                  "training_ids": ids, "evaluation_ids": evaluation,
                                  "training_groups": len({data.groups[i] for i in ids}), "shared_groups": 0})
    return result


def apply_policy(policy, identifiers, data, scores):
    """Replay only logged counterfactuals; unlogged targets remain stacking by design."""
    candidates = sorted(set(identifiers) & set(data.logged))
    if policy.phase == "none":
        called = []
    elif policy.phase in {"all", "accept", "rule"}:
        called = candidates
    elif policy.phase in {"route", "cascade"}:
        expected = scores["pre", policy.features, policy.regularization]
        threshold = policy.threshold if policy.phase == "route" else 0.
        called = sorted((i for i in candidates if expected[i] > threshold),
                        key=lambda i: (-expected[i], data.routing[i]["text_sha256"], i))[:int(len(identifiers) * policy.budget)]
    else:
        raise ValueError("Unknown intervention policy phase.")
    accepted = set()
    for identifier in called:
        row = data.routing[identifier]
        context = data.logged[identifier]["arms"]["hybrid-3-local"]
        gpt = context["label"]
        if row["label"] == gpt:
            accepted.add(identifier)
        elif policy.phase in {"all", "route"}:
            accepted.add(identifier)
        elif policy.phase == "rule":
            if acceptance_rule(policy.rule, row, gpt, context["retrieved"]):
                accepted.add(identifier)
        elif scores["post", policy.features, policy.regularization][identifier] > policy.threshold:
            accepted.add(identifier)
    predictions = {i: data.logged[i]["arms"]["hybrid-3-local"]["label"] if i in accepted else data.routing[i]["label"] for i in identifiers}
    changed = {i for i in accepted if predictions[i] != data.routing[i]["label"]}
    return predictions, set(called), changed


def choose(candidates, predictions, calls, changes, gold):
    return max(candidates, key=lambda policy: (metrics(gold, predictions[policy])["f1_macro"],
                                              -calls[policy], -changes[policy], policy.phase == "none"))


def paired_bootstrap(data, baseline, predictions, seed, samples=1000):
    groups = sorted(set(data.groups.values()))
    group_index = {group: index for index, group in enumerate(groups)}
    old, new = np.zeros((len(groups), 9)), np.zeros((len(groups), 9))
    for identifier, true in data.gold.items():
        index = group_index[data.groups[identifier]]
        old[index, LABELS.index(true) * 3 + LABELS.index(baseline[identifier])] += 1
        new[index, LABELS.index(true) * 3 + LABELS.index(predictions[identifier])] += 1
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(samples):
        weights = rng.multinomial(len(groups), np.ones(len(groups)) / len(groups))
        before, after = [(weights @ array).reshape(3, 3) for array in (old, new)]
        def f1(matrix):
            denominator = matrix.sum(0) + matrix.sum(1)
            return np.divide(2 * np.diag(matrix), denominator, out=np.zeros(3), where=denominator != 0).mean()
        differences.append((after.trace() / after.sum() - before.trace() / before.sum(), f1(after) - f1(before)))
    intervals = np.quantile(differences, [.025, .975], axis=0)
    return {"accuracy_difference_95pct": intervals[:, 0].tolist(), "macro_f1_difference_95pct": intervals[:, 1].tolist(),
            "samples": samples, "note": "Descriptive group bootstrap of fixed OOF predictions; selection is not rerun."}


def reference_policies(data, baseline, matched_calls):
    """Descriptive controls, excluded from the nested candidate-selection search."""
    result = {}
    identifiers = sorted(data.gold)
    for name in ("protect_c234", "two_base_votes", "three_base_votes", "two_examples", "votes_and_example", "protect_and_votes"):
        predictions, called, changed = apply_policy(Policy("reference", "rule", rule=name), identifiers, data, {})
        necessary = {i for i in called if rule_can_change_prediction(name, data.routing[i], data.logged[i]["arms"]["hybrid-3-local"]["retrieved"])}
        skipped = called - necessary
        if any(predictions[i] != baseline[i] for i in skipped):
            raise ValueError("Pre-call short circuit changed a post-rule prediction.")
        result[name] = {"metrics": metrics(data.gold, predictions), "requested_targets": len(called),
                        "equivalent_pre_call_requested_targets": len(necessary),
                        "safe_pre_call_skips": len(skipped), "short_circuit_predictions_identical": True,
                        "corrected": sum(predictions[i] == data.gold[i] and baseline[i] != data.gold[i] for i in predictions),
                        "harmed": sum(predictions[i] != data.gold[i] and baseline[i] == data.gold[i] for i in predictions)}
    for model_name, requested in matched_calls.items():
        if not model_name.startswith("route_"):
            continue
        chosen = set()
        for fold in sorted(set(data.folds.values())):
            count = sum(data.folds[i] == fold for i in requested)
            available = [i for i in data.logged if data.folds[i] == fold]
            chosen.update(sorted(available, key=lambda i: (data.routing[i]["uncertainty_rank"], i))[:count])
        predictions = {i: data.logged[i]["arms"]["hybrid-3-local"]["label"] if i in chosen else baseline[i] for i in identifiers}
        result["uncertainty_matched_to_" + model_name] = {"metrics": metrics(data.gold, predictions), "requested_targets": len(chosen),
            "corrected": sum(predictions[i] == data.gold[i] and baseline[i] != data.gold[i] for i in predictions),
            "harmed": sum(predictions[i] != data.gold[i] and baseline[i] == data.gold[i] for i in predictions)}
    return result


def stage_holdout(data, tables, candidates, config, output, audit):
    """Choose only on the 48 pilot queries, then replay the separate confirmation cohort."""
    train = sorted(i for i, log in data.logged.items() if log["stage"] == "pilot")
    excluded_groups = {data.groups[i] for i in train}
    test = sorted(i for i in data.gold if data.groups[i] not in excluded_groups)
    assert_disjoint(train, test, data.groups)
    predictions = {policy: {} for policy in candidates}
    calls, changes = {policy: 0 for policy in candidates}, {policy: 0 for policy in candidates}
    splitter = StratifiedGroupKFold(config.inner_folds, shuffle=True, random_state=config.seed)
    for left, right in splitter.split(np.zeros(len(train)), [data.gold[i] for i in train], [data.groups[i] for i in train]):
        fit, validation = [train[i] for i in left], [train[i] for i in right]
        scores = fitted_scores(data, tables, fit, validation, config, audit)
        for policy in candidates:
            pred, requested, accepted = apply_policy(policy, validation, data, scores)
            predictions[policy].update(pred)
            calls[policy] += len(requested)
            changes[policy] += len(accepted)
    winner = choose(candidates, predictions, calls, changes, {i: data.gold[i] for i in train})
    scores = fitted_scores(data, tables, train, test, config, audit)
    predictions, called, accepted = apply_policy(winner, test, data, scores)
    baseline = {i: data.routing[i]["label"] for i in test}
    write_jsonl(output / "pilot-trained-confirmation-predictions.jsonl", [{"excel_row": i, "label": predictions[i],
        "called_in_replay": i in called, "accepted_change": i in accepted} for i in test])
    return {"selected_using_pilot_only": asdict(winner), "training_queries": len(train), "evaluation_rows": len(test),
            "shared_groups": 0, "metrics": metrics({i: data.gold[i] for i in test}, predictions),
            "baseline": metrics({i: data.gold[i] for i in test}, baseline), "requested_targets": len(called),
            "corrected": sum(predictions[i] == data.gold[i] and baseline[i] != data.gold[i] for i in test),
            "harmed": sum(predictions[i] != data.gold[i] and baseline[i] == data.gold[i] for i in test)}


def run(config, output: Path, *, data=None):
    if output.exists():
        raise FileExistsError("Use a fresh logged-policy study directory.")
    data = data or load_data(config)
    families = policy_families(config)
    candidates = list(dict.fromkeys(policy for values in families.values() for policy in values))
    tables = feature_tables(data)
    output.mkdir(parents=True)
    protocol = {"config": config.to_dict(), "source": data.identities,
                "implementation": {name: file_digest(Path(__file__).with_name(name)) for name in ("config.py", "data.py", "interventions.py", "acceptance.py")},
                "candidates": [asdict(policy) for policy in candidates], "outer_folds": sorted(set(data.folds.values())),
                "inner_grouped_folds": config.inner_folds, "features": {str(key): sorted(next(iter(value.values()))) for key, value in tables.items()},
                "criterion": "Inner grouped OOF macro-F1, then fewer requested targets and accepted changes; zero calls eligible.",
                "supported_queries": len(data.logged), "population_rows": len(data.gold),
                "support_note": "All routing comparisons restrict candidate calls to historical logged support; no GPT outcomes are imputed for unqueried targets.",
                "new_gpt_calls": 0, "test_sets_accessed": False}
    write_json(output / "protocol.json", protocol)
    analysis, events = historical_analysis(data)
    write_json(output / "historical-analysis.json", analysis)
    write_jsonl(output / "historical-events.jsonl", events)
    baseline = {i: row["label"] for i, row in data.routing.items()}
    predictions = {name: {} for name in [*families, "joint_nested_selection"]}
    called = {name: set() for name in predictions}
    changed = {name: set() for name in predictions}
    selections, audit, started = [], [], time.monotonic()
    for outer in sorted(set(data.folds.values())):
        train = sorted(i for i, fold in data.folds.items() if fold != outer)
        test = sorted(set(data.folds) - set(train))
        assert_disjoint(train, test, data.groups)
        splitter = StratifiedGroupKFold(config.inner_folds, shuffle=True, random_state=config.seed + outer)
        inner_predictions = {policy: {} for policy in candidates}
        inner_calls = {policy: 0 for policy in candidates}
        inner_changes = {policy: 0 for policy in candidates}
        for inner_train, inner_test in splitter.split(np.zeros(len(train)), [data.gold[i] for i in train], [data.groups[i] for i in train]):
            fit_ids, validation_ids = [train[i] for i in inner_train], [train[i] for i in inner_test]
            scores = fitted_scores(data, tables, fit_ids, validation_ids, config, audit)
            for policy in candidates:
                pred, requested, accepted = apply_policy(policy, validation_ids, data, scores)
                inner_predictions[policy].update(pred)
                inner_calls[policy] += len(requested)
                inner_changes[policy] += len(accepted)
        inner_gold = {i: data.gold[i] for i in train}
        winners = {name: choose(values, inner_predictions, inner_calls, inner_changes, inner_gold) for name, values in families.items()}
        winners["joint_nested_selection"] = choose(candidates, inner_predictions, inner_calls, inner_changes, inner_gold)
        scores = fitted_scores(data, tables, train, test, config, audit)
        for name, policy in winners.items():
            pred, requested, accepted = apply_policy(policy, test, data, scores)
            predictions[name].update(pred)
            called[name].update(requested)
            changed[name].update(accepted)
        selections.append({"outer_fold": outer, "training_ids": train, "evaluation_ids": test,
                           "shared_groups": 0, "selected": {name: asdict(policy) for name, policy in winners.items()}})
        print(f"Logged-policy nested CV: outer {outer} complete; {len(candidates)} fixed candidates; {time.monotonic() - started:.1f}s", file=sys.stderr, flush=True)
    # Persist every held-out output before deriving overall metrics or selecting report rows.
    for name, pred in predictions.items():
        write_jsonl(output / (name + ".jsonl"), [{"excel_row": i, "outer_fold": data.folds[i], "group_id": data.groups[i],
            "label": pred[i], "stacking_label": baseline[i], "logged_support": i in data.logged,
            "called_in_replay": i in called[name], "accepted_change": i in changed[name]} for i in sorted(pred)])
    write_jsonl(output / "outer-selections.jsonl", selections)
    holdout = stage_holdout(data, tables, candidates, config, output, audit)
    write_jsonl(output / "fit-audit.jsonl", audit)
    unconditional = {i: data.logged[i]["arms"]["hybrid-3-local"]["label"] if i in data.logged else baseline[i] for i in baseline}
    summary = {}
    for name, pred in {"baseline": baseline, "unconditional_logged": unconditional, **predictions}.items():
        corrected = sum(pred[i] == data.gold[i] and baseline[i] != data.gold[i] for i in pred)
        harmed = sum(pred[i] != data.gold[i] and baseline[i] == data.gold[i] for i in pred)
        summary[name] = {"metrics": metrics(data.gold, pred), "corrected": corrected, "harmed": harmed,
                         "requested_targets_in_replay": len(called.get(name, data.logged if name == "unconditional_logged" else {})),
                         "accepted_changes": sum(pred[i] != baseline[i] for i in pred)}
    report = {"status": "completed", "protocol_sha256": file_digest(output / "protocol.json"),
              "historical_analysis": analysis, "results": summary,
              "joint_selection_bootstrap": paired_bootstrap(data, baseline, predictions["joint_nested_selection"], config.seed),
              "reference_policies_descriptive_only": reference_policies(data, baseline, called),
              "pilot_trained_confirmation_holdout": holdout,
              "outer_selections": selections, "fits": len(audit), "elapsed_seconds": time.monotonic() - started,
              "new_gpt_calls": 0, "base_model_updates": 0, "support_queries": len(data.logged),
              "files": {path.name: file_digest(path) for path in output.glob("*.jsonl")},
              "unqueried_counterfactuals_unavailable": len(data.gold) - len(data.logged),
              "primary_estimate": "joint_nested_selection includes family, regularization, threshold and budget selection inside each outer fold.",
              "limitations": ["Routing is conditional on logged support; quality outside the historical queried pool is unidentified.",
                              "Current labels informed retrospective hypotheses before this study; grouped CV does not erase that exposure.",
                              "All scores reuse the original hybrid-3-local GPT context, never synthetic replies for new retrieved examples.",
                              "No final test set or production configuration has been changed."]}
    write_json(output / "acceptance-report.json", report)
    return {key: value for key, value in report.items() if key not in {"historical_analysis", "outer_selections"}}
