"""Reconstruct paired cluster intervals using independent row-weighted confusion counts."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ensemble_classifier.files import read_json, read_predictions
from gpt_classifier.data import LABELS, file_digest, group, read_xlsx, write_json
from svm_budgeted.cli import base_config
from svm_finalization.pipeline import load_frozen
from svm_research.data import load_data, location


def verify_intervals(labels, groups, predictions, saved):
    truth = np.asarray([LABELS.index(label) for label in labels])
    _, inverse = np.unique(groups, return_inverse=True)
    count = int(inverse.max() + 1)
    if saved["groups"] != count or saved["seed"] != 42 or saved["repeats"] != 2000:
        raise ValueError("Unexpected bootstrap protocol.")
    names = list(predictions)
    draws = np.empty((2000, len(names), 2))
    generator = np.random.default_rng(42)
    codes = [truth * 3 + np.asarray([LABELS.index(label) for label in predictions[name]]) for name in names]
    for start in range(0, 2000, 128):
        size = min(128, 2000 - start)
        group_weights = generator.multinomial(count, np.full(count, 1 / count), size=size)
        row_weights = group_weights[:, inverse]
        for model, code in enumerate(codes):
            matrices = np.stack([np.bincount(code, weights=weights, minlength=9).reshape(3, 3) for weights in row_weights])
            true, predicted = matrices.sum(axis=2), matrices.sum(axis=1)
            tp = np.diagonal(matrices, axis1=1, axis2=2)
            f1 = np.divide(2 * tp, true + predicted, out=np.zeros_like(tp), where=true + predicted != 0).mean(axis=1)
            draws[start:start + size, model, 0] = tp.sum(axis=1) / matrices.sum(axis=(1, 2))
            draws[start:start + size, model, 1] = f1
    checked = 0
    for i, name in enumerate(names):
        for j, metric in enumerate(("accuracy", "f1_macro")):
            if not np.allclose(np.quantile(draws[:, i, j], [.025, .975]), saved["absolute_95_intervals"][name][metric], atol=1e-12, rtol=0):
                raise ValueError("Absolute confidence interval mismatch.")
            checked += 1
    for pair in saved["pairs"].values():
        delta = draws[:, names.index(pair["right"])] - draws[:, names.index(pair["left"])]
        for j, metric in enumerate(("accuracy", "f1_macro")):
            if not np.allclose(np.quantile(delta[:, j], [.025, .975]), pair["delta_95_interval"][metric], atol=1e-12, rtol=0):
                raise ValueError("Paired confidence interval mismatch.")
            checked += 1
    return {"groups": count, "models": len(names), "intervals_verified": checked, "draws": 2000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new verification output file.")
    frozen = load_frozen(args.run)
    plan = read_json(args.run / "plan.json")
    data = load_data(base_config(plan["search_config_value"]))
    dev = read_json(args.run / "development.json")
    selected = data.selection
    with np.load(args.run / "meta-validation.npz", allow_pickle=False) as arrays:
        names = list(dev["paired_diagnostics"]["absolute_95_intervals"])
        predictions = {name: np.asarray(LABELS)[arrays[name].argmax(axis=1)] for name in names}
    development = verify_intervals([data.holdout[i].label for i in selected], data.holdout_groups[selected], predictions, dev["paired_diagnostics"])
    final = read_json(args.run / "test1/metrics.json")
    manifest = read_json(args.run / "test1/prediction-manifest.json")
    gold, source = read_xlsx(location(plan["test1_gold"]), labeled=True)
    if final["frozen_sha256"] != file_digest(args.run / "frozen.json") or source["sha256"] != plan["test1_expected_gold_sha256"]:
        raise ValueError("Final evaluation provenance changed.")
    predictions = {}
    for name in final["paired_statistics"]["absolute_95_intervals"]:
        path = args.run / "test1" / f"{name}.jsonl"
        if file_digest(path) != manifest["files"][path.name]:
            raise ValueError("Final predictions changed.")
        predictions[name] = np.asarray([r["label"] for r in read_predictions(path)])
    test1 = verify_intervals([r.label for r in gold], [group(r.text) for r in gold], predictions, final["paired_statistics"])
    result = {"status": "passed", "method": "Independent row-weighted bincount confusion matrices using the same paired cluster draws",
              "development": development, "test1": test1, "frozen_sha256": file_digest(args.run / "frozen.json"),
              "recommendation": frozen["recommended_configuration"], "verifier_sha256": file_digest(Path(__file__)), "test2_accessed": False}
    write_json(args.output, result, replace=False)
    print(result)


if __name__ == "__main__":
    main()
