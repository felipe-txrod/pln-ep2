from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import LABELS, Example, digest


def align(records: list[Example], predictions: list[dict]) -> list[str]:
    if len(records) != len(predictions):
        raise ValueError("Quantidade de previsões diferente da quantidade de registros.")
    expected_rows = [r.excel_row for r in records]
    observed_rows = [p.get("excel_row") for p in predictions]
    if observed_rows != expected_rows or len(set(observed_rows)) != len(observed_rows):
        raise ValueError("Previsões fora da ordem original ou IDs duplicados.")
    for record, prediction in zip(records, predictions):
        if prediction.get("text_sha256") != digest(record.text):
            raise ValueError(f"Texto desalinhado na linha {record.excel_row}.")
        if prediction.get("label") not in LABELS:
            raise ValueError("Previsão com rótulo inválido.")
    return [p["label"] for p in predictions]


def metrics(records: list[Example], predictions: list[dict]) -> dict:
    predicted = align(records, predictions)
    gold = [r.label for r in records]
    return {"rows": len(records), "accuracy": accuracy_score(gold, predicted),
            "f1_macro": f1_score(gold, predicted, labels=list(LABELS), average="macro", zero_division=0),
            "label_order": list(LABELS), "confusion_matrix": confusion_matrix(gold, predicted, labels=list(LABELS)).tolist(),
            "classification_report": classification_report(gold, predicted, labels=list(LABELS), output_dict=True, zero_division=0),
            "prediction_counts": dict(Counter(predicted))}


def compare(gold: list[Example], predictions: list[dict], baseline: list[Example], train: list[Example]) -> dict:
    if len(gold) != 900 or len(baseline) != 900:
        raise ValueError("A comparação oficial exige as 900 linhas de teste1.")
    for truth, old in zip(gold, baseline):
        if truth.excel_row != old.excel_row or truth.text != old.text:
            raise ValueError("A entrega EP1 não está alinhada ao gabarito de teste1.")
    pred = np.array(align(gold, predictions))
    y = np.array([r.label for r in gold])
    old = np.array([r.label for r in baseline])
    previous, current = old == y, pred == y
    a = int(np.sum(previous & current))
    b = int(np.sum(previous & ~current))  # Only EP1 correct.
    c = int(np.sum(~previous & current))  # Only GPT correct.
    d = int(np.sum(~previous & ~current))
    discordant = b + c
    exact_p = float(binomtest(b, discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    uncorrected = (b - c) ** 2 / discordant if discordant else 0.0
    corrected = max(abs(b - c) - 1, 0) ** 2 / discordant if discordant else 0.0
    majority = max(LABELS, key=lambda label: sum(r.label == label for r in train))
    return {"ep2": metrics(gold, predictions), "ep1": {"accuracy": float(previous.mean()), "correct": int(previous.sum())},
            "accuracy_delta": float(current.mean() - previous.mean()),
            "training_majority_baseline": {"label": majority, "test1_accuracy": float(np.mean(y == majority))},
            "mcnemar": {"contingency_rows_ep1_correct_incorrect_columns_ep2_correct_incorrect": [[a, b], [c, d]],
                        "only_ep1_correct": b, "only_ep2_correct": c,
                        "chi2_uncorrected": uncorrected, "chi2_continuity_corrected": corrected,
                        "p_chi2_corrected": float(chi2.sf(corrected, 1)),
                        "p_exact_two_sided": exact_p, "primary_test": "exact two-sided binomial McNemar",
                        "alpha": 0.05, "significantly_superior": c > b and exact_p < 0.05},
            "caveat": "Teste1 reservado à comparação final. p<0,05 sozinho não demonstra superioridade se o delta for negativo; linhas repetidas também limitam a hipótese de independência entre pares."}


def error_cases(gold: list[Example], predictions: list[dict]) -> list[dict]:
    labels = align(gold, predictions)
    return [{"excel_row": r.excel_row, "resp_text": r.text, "gold": r.label, "prediction": p,
             "manual_analysis": "Preencher após leitura humana; não inferir intenção/anexos ausentes."}
            for r, p in zip(gold, labels) if r.label != p]
