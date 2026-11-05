# Predict test1 with a trained NorBERTo adapter and score saved predictions

[Documentation index](../README.md)

Run these commands from the EP2 root. They use the existing `pln-rocm:6.4.1`
image, sibling EP1 sources/cache, and the completed recipe-3 development adapter.
`NORBERTO_IMAGE` and `NORBERTO_EP1_ROOT` can override the local image and EP1 root.

First, predict using only the texts in the unlabeled workbook:

```bash
bash scripts/norberto/test1.sh predict \
  --training-run results/norberto-development-recipe3 \
  --input data/test1.xlsx \
  --output-dir results/norberto-recipe3-test1
```

Prediction uses the GPU with ROCm override `10.3.0`, FP32, and batch 8. It checks
training completion, the frozen development split, row IDs, recipe hashes, and
exported adapter hashes before loading the model. No test1 gold labels are read.
All 900 original rows are saved to `predictions.jsonl`; `run.json` records the
adapter identity, input hash, prediction hash, and `gold_accessed=false`.

Then compare the saved predictions with the labeled workbook:

```bash
bash scripts/norberto/test1.sh compare \
  --prediction-dir results/norberto-recipe3-test1 \
  --input data/test1.xlsx \
  --gold data/test1-rotulado-para-teste-estatistico.xlsx \
  --output-dir results/norberto-recipe3-test1-metrics
```

Comparison runs in a container without GPU device exposure. It loads no model
and performs no new inference. It verifies the prediction file hash, original
test1 input hash, row IDs/order, and exact text alignment with the gold workbook.
It writes `metrics.json` with accuracy, macro-F1, macro-precision, macro-recall,
per-class metrics, and the confusion matrix in `c1/c234/c5` order, plus provenance
in `run.json`. This compares predictions with gold; it does not run a paired
comparison against the original EP1 submission.

Both stages require new output directories and run offline. Neither performs
training or changes the frozen original adapter. Existing full-corpus runs
remain supported by `evaluate-norberto-test1.py`; its default `--stage all`
retains the previous predict-then-score workflow.

Three CPU regression tests verify scoring without training/model access and
rejection of changed predictions and misaligned gold texts. A read-only
manifest check accepted all 16,093 recorded training rows of the actual
recipe-3 adapter without importing model libraries or performing inference.
Preparing these commands did not run the real model on test1.
