# NorBERTo evaluation after correcting loss normalization

The recipe-version-2 adapter achieved **45.33% accuracy (408/900)** and
**43.83% macro-F1** on labeled test1. This is 67 more correct predictions than
the legacy full-corpus refit, but 24 fewer than the original EP1 submission.
No ensemble configuration or model replacement was performed.

## Training completion and provenance

`results/norberto-full-normalized/` completed **two epochs and 2,512 updates**
on all **20,092 original EP1 training responses**, including the historical
holdout. Training took **2,334.80 seconds (38.91 minutes)** without reaching
the time budget. The 272,643 saved trainable parameters are finite; all 22
LoRA B tensors contain nonzero updates. Exported file hashes passed verification.

The manifest records recipe version 2 and
`loss_normalization=sample_count_across_accumulation`. The training wrapper and
loss-implementation SHA-256 values match the corrected source files. Optimizer
hyperparameters remain the original two-epoch recipe: rank 4, alpha 8, dropout
0.05, `Wqkv`, context 512 with head/tail truncation, FP32, SDPA, learning rate
0.001, batch 4, accumulation 4, weight decay 0.01, warmup ratio 0.1, seed 42.

Adapter weights SHA-256:

```text
f43ebe2b9cc1670057bdaa5c7d43b6235d002837bd8d68dbac50b243a1485d71
```

All saved training row IDs, text hashes, and labels were checked against the
original workbook. The pretrained base revision remains
`db73446f89c96044863ea05a39f680524b84bccb`.

## Test1 metrics

Inference used only texts from `data/test1.xlsx`. Predictions were persisted
before the labeled workbook was opened. All 900 original rows were retained.

| Metric | Result |
|---|---:|
| Accuracy | 45.33% |
| Correct / incorrect | 408 / 492 |
| Macro-F1 | 43.83% |
| Macro-precision | 44.95% |
| Macro-recall | 45.33% |

| Class | Precision | Recall | F1 | Correct / support |
|---|---:|---:|---:|---:|
| `c1` | 45.34% | 47.00% | 46.15% | 141 / 300 |
| `c234` | 43.35% | 25.00% | 31.71% | 75 / 300 |
| `c5` | 46.15% | 64.00% | 53.63% | 192 / 300 |

Confusion matrix: rows are true classes, columns are predictions.

| True / predicted | `c1` | `c234` | `c5` |
|---|---:|---:|---:|
| `c1` | 141 | 59 | 100 |
| `c234` | 101 | 75 | 124 |
| `c5` | 69 | 39 | 192 |

Predicted class counts were 311 `c1`, 173 `c234`, and 416 `c5`.

## Comparison of the three NorBERTo adapters

| Adapter | Training rows | Loss recipe | Correct / 900 | Accuracy |
|---|---:|---|---:|---:|
| Original EP1 submission | 16,093 | Legacy | 432 | 48.00% |
| Full-corpus legacy refit | 20,092 | Legacy | 341 | 37.89% |
| Full-corpus normalized refit | 20,092 | Sample-normalized | 408 | 45.33% |

Compared with the legacy full refit, correct predictions by class changed by +85 in
`c1`, −72 in `c234`, and +54 in `c5`, totaling +67. Compared with the original
EP1 submission, changes were +23, −80, and +33. The current main weakness is
recovery of the intermediate class, with recall of 25.00%.

Paired comparisons reuse saved predictions from the other adapters; neither
was executed again:

| Baseline | Only baseline correct | Only normalized correct | Exact two-sided McNemar p |
|---|---:|---:|---:|
| Original EP1 submission | 145 | 121 | 0.158346 |
| Legacy full-corpus refit | 154 | 221 | 0.000634651 |

The first comparison does not establish a significant difference at 0.05 or
equivalence. The second nominally favors the corrected adapter. Prior test1
inspection and repeated texts limit confirmatory interpretation. No additional
hyperparameters or ensemble rules were selected using these results.

## Training behavior after correction

| Epoch interval | Mean sample-normalized loss | Mean pre-clipping gradient norm |
|---|---:|---:|
| 0–0.5 | 1.12765 | 3.15449 |
| 0.5–1 | 1.09751 | 1.39068 |
| 1–1.5 | 1.07218 | 1.35278 |
| 1.5–2 | 1.05983 | 1.44972 |

The maximum logged norm was 10.0386. This run did not show the escalation to
hundreds recorded in the legacy full refit. Historical raw loss/gradient scales
also differ because normalization changed, so raw ratios alone are not causal
proof. The observed stability and recovered predictions are consistent with
the correction helping this execution. A single run does not isolate all causes
or establish a general guarantee of improved accuracy.

The [investigation](investigacao-norberto-full.md) predates this follow-up and
documents the bug, original instability, tests, and causal limits.

## Execution, independent verification, and scope

The exported adapter was evaluated once on the RX 6650 XT, in FP32 with batch
8 and ROCm override `10.3.0`, using the existing
[`evaluate-norberto-test1.py`](../scripts/evaluate-norberto-test1.py) script.
The container had no network access; project files, original sources, and base
cache were read-only. Loading, tokenization, and inference took **34.59 seconds**.

An independent standard-library ZIP/XML reader recomputed metrics and verified
all 900 gold labels and row IDs, confusion counts, probability distributions,
finite logits, argmax labels, and input/prediction hashes. No training or
additional inference was performed during this verification.

This is an isolated test1 comparison without a preceding test2 run for this
adapter. The former holdout is training data and is not an unseen validation
set for this model. Original adapters and ensemble configuration remain unchanged.

For reproduction in the validated environment with EP1 and its cache mounted:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=src \
python3 scripts/evaluate-norberto-test1.py \
  --ep1-root /ep1 \
  --training-run results/norberto-full-normalized \
  --output-dir results/norberto-normalized-test1-another-new-run
```

Raw outputs are in `results/norberto-normalized-test1-20261105/`.
The [public result aggregate](norberto-normalized-test1-results.json) contains
full metrics, provenance, training checks, and paired comparisons.
