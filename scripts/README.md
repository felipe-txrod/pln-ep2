# Script entry points

[Project documentation](../docs/README.md)

Run commands from the EP2 root. Scripts are grouped by model or purpose.
Dataset, model, cache, and result locations remain relative to the project root.

## NorBERTo

The active model is the unchanged original EP1 delivery adapter in
`models/norberto/`. Prediction verifies its pinned adapter/base and uses the
original FP32/SDPA, 512-token head/tail representation.

| Entry point | Purpose |
|---|---|
| `norberto/test1.sh predict` | Predict with the original EP1 adapter on GPU |
| `norberto/test1.sh compare` | Score saved original-model predictions on CPU |
| `norberto/evaluate-test1.py` | Python implementation of these two stages |

```bash
bash scripts/norberto/test1.sh predict \
  --output-dir results/norberto-ep1-test1-new

bash scripts/norberto/test1.sh compare \
  --prediction-dir results/norberto-ep1-test1-new \
  --output-dir results/norberto-ep1-test1-metrics-new
```

Use the [test1 guide](../docs/guides/norberto-test1.md) and
[original inference package](../src/norberto_classifier/README.md).
The active package exposes inference only. Experimental loss corrections,
training launchers, diagnostics, candidate-08 evaluators, and the frozen
logistic variant were retired on 2026-11-05. Their measurements remain in the
[archive](../docs/archive/README.md#norberto-training-investigations).

## SVM

Evaluate a newly trained EP1 artifact without replacing the frozen model:

```bash
bash scripts/svm/test1.sh \
  --artifact results/svm-full/model.joblib \
  --input data/test1.xlsx \
  --gold data/test1-rotulado-para-teste-estatistico.xlsx \
  --output-dir results/svm-full-test1
```

This offline CPU-only command validates the saved recipe and adjacent metadata,
persists text-only predictions, then reads gold labels to produce `metrics.json`.
The output includes accuracy, macro-F1, precision, recall, and the confusion
matrix. Use a new output directory. `SVM_IMAGE` overrides the existing image.

## Logistic regression

```bash
bash scripts/rl/train.sh --dry-run
bash scripts/rl/train.sh
```

The CPU-only offline launcher fits the fixed EP1 logistic recipe on 16,093
frozen development rows and excludes all 3,999 holdout rows. It creates
`models/rl-development/` for the selected RL default. The historical full-corpus
RL model is preserved and never selected as a fallback. See the
[RL guide](../src/rl_classifier/README.md).

## Qwen

| Entry point | Purpose |
|---|---|
| `qwen/container.sh` | Run the Qwen CLI in the configured ROCm image |
| `qwen/evaluate-test1.py` | Evaluate an exported trained adapter on test1 |

[Qwen workflow and compatibility](../docs/guides/qwen-finetuning.md).

For retraining on the same development/holdout rows as the other local models,
pass `--ep1-split data/ep1-development-holdout.csv` to `train` or `prepare`.
The [exact command](../docs/guides/qwen-finetuning.md#retrain-with-the-exact-ep1-partition)
uses a fresh output and automatic compatibility preflight. The selected Qwen
now participates in [soft voting and stacking](../docs/guides/ensemble-qwen.md);
majority presets retain three voters.

## Model artifacts

| Entry point | Purpose |
|---|---|
| `models/prepare-base.py` | Restore or verify the pinned public NorBERTo base |
| `models/freeze.py` | Export existing inference artifacts without training |

```bash
python3 scripts/models/prepare-base.py --verify-only
python3 scripts/models/freeze.py --help
```

[Artifact formats and restoration](../models/README.md).

## Diagnostics and experiments

`diagnostics/audit-dataset-overlap.py` compares the six current EP1/EP2
workbooks on CPU, without model loading. It reports file/content identity,
exact and normalized text overlap, and conflicting labels. See the
[dataset audit](../docs/results/dataset-overlap.md) for results and reproduction.

Former NorBERTo training diagnostics are historical records, with local
source snapshots in
`results/norberto-rollback-20261105/sources-before-rollback.tar.gz`.

`experiments/gpt/pilot-efficiency.py` contains the earlier paired GPT/Codex
pilot. It prepares a dry run by default; `--execute` enables remote requests.
See the [historical GPT study](../docs/archive/gpt/first-result.md).

## Compatibility commands

`prepare-model-base.py` forwards to `models/prepare-base.py`, preserving
restoration instructions emitted by frozen model packages. The removed
NorBERTo training wrappers are available only in the historical source snapshot.

Choose a fresh output destination for every prediction or comparison run.

## Weighted soft voting versus the EP1 delivery

RAG context experiments use `python -m rag_cascade`. This workflow is currently
**prepared, not executed** at the user's request. Classification requires
`--execute`; see the [RAG guide](../docs/guides/rag-stacking.md) for previews,
train-only example retrieval and the test2-before-test1 guards.

Selective stacking + GPT uses `python -m cascade_classifier`; no additional
wrapper is required. The [cascade guide](../docs/guides/stacking-gpt-cascade.md)
documents preparation, initial 10% prompt ablation, cached extension, frozen
budget selection, prediction and separate final evaluation.

```bash
PYTHONPATH=src .venv/bin/python scripts/ensemble/compare-ep1.py \
  --run results/ensemble-soft-test1-20261105 \
  --baseline ../EP1/deliveries/test1.xlsx \
  --output-dir results/ensemble-soft-vs-ep1-new
```

This offline CPU comparison verifies the frozen artifact and saved predictions,
aligns the actual EP1 workbook with all 900 gold rows, and reports accuracy,
macro-F1, confusion matrices and paired McNemar. It performs no inference or
parameter tuning. It is an isolated comparison; it does not certify the
assignment's test2-before-test1 order. See the
[results and protocol limitations](../docs/results/ensemble-soft.md#conferência-contra-a-entrega-efetiva-do-ep1).

The current four-model commands and frozen artifacts, including Qwen, are
documented in the [Qwen ensemble guide](../docs/guides/ensemble-qwen.md).

For stacking, use `python -m ensemble_classifier fit-stacking`,
`predict-stacking` and `evaluate-stacking`. No additional launcher is required.
The [guide](../docs/guides/ensemble-stacking.md) gives the exact commands and
cached-score workflow; the [results](../docs/results/ensemble-stacking.md)
include per-class F1, confusion matrices and corrected/harmed Excel rows.
