# Predict and score test1 with the original EP1 NorBERTo

[Documentation index](../README.md)

The selected model is the unchanged EP1 delivery adapter in `models/norberto/`:
LoRA rank 4/alpha 8, pinned public base revision, 512-token head/tail input,
FP32/SDPA, and batch 8. Its original final training used two epochs on 16,093
development responses. No training or loss correction runs in this workflow.
Sibling EP1 sources and Hugging Face caches are not required after the model
bundle has been restored. `NORBERTO_IMAGE` selects another compatible local image.

Run from EP2 using the existing offline `pln-rocm:6.4.1` environment:

```bash
bash scripts/norberto/test1.sh predict \
  --input data/test1.xlsx \
  --output-dir results/norberto-ep1-test1-new
```

This predicts all 900 text-only responses, saves row IDs, text hashes, labels
and probabilities, and records the original adapter and artifact-manifest
identities. Gold labels are not read. The package validates immutable base,
adapter, tokenizer, configuration, and training metadata before loading.

Score the persisted predictions separately, on CPU:

```bash
bash scripts/norberto/test1.sh compare \
  --prediction-dir results/norberto-ep1-test1-new \
  --input data/test1.xlsx \
  --gold data/test1-rotulado-para-teste-estatistico.xlsx \
  --output-dir results/norberto-ep1-test1-metrics-new
```

Comparison requires the original EP1 adapter identity, an intact prediction
file, and exact row/text alignment. It loads no model and writes accuracy,
macro-F1, precision, recall, class reports, and a confusion matrix. Both stages
require new output directories. The removed `--training-run` interface no
longer accepts experimental adapters.

The restored inference path was verified on all 900 rows against the original
EP1 delivery: all labels match, with 432 correct predictions (48.00%) and
47.86% macro-F1. Verification artifacts are under
`results/norberto-rollback-20261105/`. The ensemble configuration identifies
the original model through `artifact_manifest_sha256` and reads its selected
predictions from `results/norberto-ep1-test1` (test1) or
`results/norberto-ep1-test2` (test2). Test2 inference is a separate operation:

```bash
PYTHONPATH=src python3 -m norberto_classifier predict \
  --input data/test2.xlsx --device gpu \
  --output-dir results/norberto-ep1-test2
```

No test2 prediction or delivery was performed during restoration.
