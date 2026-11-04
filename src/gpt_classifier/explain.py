from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import LABELS, Example, clean, write_json
from .evaluation import align


def explain(train: list[Example], gold: list[Example], predictions: list[dict], output: Path, *, count: int = 12):
    """Independent lexical diagnostic model, never an explanation of GPT internals."""
    labels = align(gold, predictions)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000, sublinear_tf=True)
    x = vectorizer.fit_transform([clean(r.text) for r in train])
    model = LogisticRegression(C=0.25, max_iter=1000, random_state=42).fit(x, [r.label for r in train])
    names = vectorizer.get_feature_names_out()
    global_features = {}
    for label, coefficients in zip(model.classes_, model.coef_):
        global_features[label] = {"highest_coefficients": [{"term": names[i], "weight": float(coefficients[i])}
                                                          for i in np.argsort(-coefficients)[:20]],
                                  "lowest_coefficients": [{"term": names[i], "weight": float(coefficients[i])}
                                                         for i in np.argsort(coefficients)[:20]]}
    # Prioritize GPT errors, then include correct examples. No remote call here.
    indices = sorted(range(len(gold)), key=lambda i: (gold[i].label == labels[i], gold[i].excel_row))[:count]
    local = []
    for i in indices:
        record = gold[i]
        features = vectorizer.transform([clean(record.text)])
        scores = model.decision_function(features)[0]
        diagnostic_label = model.classes_[int(np.argmax(scores))]
        cls = list(model.classes_).index(diagnostic_label)
        contributions = features.data * model.coef_[cls, features.indices]
        ranking = np.argsort(-np.abs(contributions))[:20]
        local.append({"excel_row": record.excel_row, "resp_text": record.text, "gold": record.label,
                      "gpt_prediction": labels[i], "diagnostic_prediction": diagnostic_label,
                      "diagnostic_intercept": float(model.intercept_[cls]),
                      "diagnostic_score": float(scores[cls]),
                      "all_terms_contribution_sum": float(contributions.sum()),
                      "top_contributions": [{"term": names[features.indices[k]], "contribution": float(contributions[k])}
                                            for k in ranking]})
    result = {"method": "TF-IDF unigramas/bigramas + regressão logística C=0.25, ajuste só no treino",
              "interpretation_limit": "Coeficientes e contribuições explicam apenas o modelo lexical auxiliar, não decisões internas ou causalidade do GPT. Associação com classe não é prova de clareza.",
              "labels": list(LABELS), "global": global_features, "local": local}
    write_json(output, result, replace=False)
    return result
