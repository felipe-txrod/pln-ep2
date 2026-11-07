"""Export exact explanations of the frozen development logistic regression."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gpt_classifier.data import read_xlsx
from rl_classifier import RLClassifier


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model_path = Path("models/rl-development/model.joblib")
    before = checksum(model_path)
    model = RLClassifier().load()._model
    features = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    names = features.get_feature_names_out()
    word_indices = [i for i, name in enumerate(names) if name.startswith("word__")]
    global_terms = {}
    for label, coefficients in zip(classifier.classes_, classifier.coef_, strict=True):
        global_terms[label] = [
            {"term": str(names[i]).removeprefix("word__"), "coefficient": float(coefficients[i])}
            for i in sorted(word_indices, key=lambda i: -coefficients[i])[:15]
        ]
    gold_path = Path("data/test1-rotulado-para-teste-estatistico.xlsx")
    prediction_path = Path("results/svm-finalization-20261106/test1/A.jsonl")
    gold, _ = read_xlsx(gold_path, labeled=True)
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    if len(gold) != 900 or len(predictions) != 900:
        raise ValueError("Expected all 900 original observations.")
    local = []
    selected_rows = {25, 77, 137, 693}
    for record, prediction in zip(gold, predictions, strict=True):
        if prediction["excel_row"] != record.excel_row:
            raise ValueError("Prediction row mismatch.")
        if prediction["text_sha256"] != hashlib.sha256(record.text.encode()).hexdigest():
            raise ValueError("Prediction text checksum mismatch.")
        if record.excel_row not in selected_rows:
            continue
        vector = features.transform([record.text])
        scores = classifier.decision_function(vector)[0]
        label_index = int(np.argmax(scores))
        contributions = vector.data * classifier.coef_[label_index, vector.indices]
        intercept = float(classifier.intercept_[label_index])
        if not np.isclose(intercept + contributions.sum(), scores[label_index], atol=1e-12):
            raise ValueError("The additive explanation failed to reconstruct its logit.")
        local.append({
            "excel_row": record.excel_row,
            "text": record.text.strip(),
            "gold": record.label,
            "soft_prediction": prediction["label"],
            "soft_probabilities": prediction["probabilities"],
            "logistic_prediction": str(classifier.classes_[label_index]),
            "logistic_intercept": intercept,
            "logistic_score": float(scores[label_index]),
            "all_contributions_sum": float(contributions.sum()),
            "top_absolute_contributions": [
                {"feature": str(names[vector.indices[i]]), "contribution": float(contributions[i])}
                for i in np.argsort(-np.abs(contributions))[:15]
            ],
        })
    if len(local) != len(selected_rows) or checksum(model_path) != before:
        raise ValueError("Missing selected cases or modified model.")
    result = {
        "method": "Exact additive logits of the frozen word/character logistic regression",
        "training_performed": False,
        "remote_calls": 0,
        "model_sha256": before,
        "gold_sha256": checksum(gold_path),
        "predictions_sha256": checksum(prediction_path),
        "labels": list(classifier.classes_),
        "global_word_terms": global_terms,
        "local": local,
        "limitations": "Explains the logistic classifier only; coefficients are associations, not causal clarity criteria or explanations of neural or ensemble internals.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {len(local)} checked local explanations; no model was fitted.")


if __name__ == "__main__":
    main()
