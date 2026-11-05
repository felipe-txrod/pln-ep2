# NorBERTo recipe 3: completed training and holdout evaluation

[Documentation index](../README.md)

The optimizer-compatible adapter achieved **44.81% accuracy (1,792/3,999)**
and **44.81% macro-F1** on the frozen EP1 holdout. This recovers 366 correct
predictions over recipe 2 and remains 20 below the original EP1 adapter on
the same holdout. The prior gradient escalation did not recur in this run.

## Training verification and stability

`results/norberto-development-recipe3/` completed **two epochs and all 2,012
updates** on **16,093 development rows**, taking **1,879.41 seconds (31.32
minutes)**. It reserved the original 3,999 holdout rows with zero row/group
overlap. Training did not use validation or test data and did not reach the
time budget. All 272,643 exported trainable parameters are finite, all 22 LoRA
B tensors contain updates, and exported hashes match the training manifest.
The recorded implementation hashes match the current sources.

Recipe 3 retains example-normalized loss reporting and explicitly scales
backpropagated gradients by accumulation count 4, preserving EP1's optimizer
units for clipping and AdamW. The numerical hyperparameters and fixed split
remain unchanged. The [correction](../guides/norberto-optimizer-units.md) explains
the difference from recipe 2 and the handling of unequal final microbatches.

| Epoch interval | Mean logged loss | Mean optimizer gradient norm | Mean sample gradient norm | Maximum optimizer norm |
|---|---:|---:|---:|---:|
| 0–0.5 | 1.12420 | 11.50810 | 2.87702 | 22.58407 |
| 0.5–1 | 1.08497 | 8.09402 | 2.02351 | 17.90670 |
| 1–1.5 | 1.04843 | 7.43696 | 1.85924 | 12.93741 |
| 1.5–2 | 1.02875 | 8.12732 | 2.03183 | 11.67706 |

These are summaries of 25 logged samples per interval. Optimizer-unit norms
are before clipping; sample-unit norms divide them by 4. The maximum logged
sample-unit norm was **5.64602**, compared with **275.30652** in recipe 2.
The last logged optimizer-unit norm was 6.21800, or 1.55450 in sample units.
Peak allocated/reserved training VRAM was **3,259.04/3,800 MiB**.

All **50 logged gradient norms through update 1,000** exactly match the
original EP1 training logs. The separate 600-update diagnostic also matched
all 30 logged norms and learning rates. At update 540, recipe 3's sample-unit
norm was 1.94616; recipe 2 had 253.14911. The original and recipe-3 optimizer
norms were both 7.78463. This supports the optimizer-unit mismatch as an
explanation for the early divergence. It does not establish bitwise equality
of all intermediate tensors or guarantee identical results after the corrected
tail weighting at the first epoch boundary.

Adapter weights SHA-256:

```text
641760227c2d5862f56fa3251d7e8722422a37d4de30a76d0623dbe52325e04c
```

The base remains `Itau-Unibanco/NorBERTo-base`, revision
`db73446f89c96044863ea05a39f680524b84bccb`.

## Frozen holdout metrics

| Metric | Result |
|---|---:|
| Accuracy | 44.81% |
| Correct / incorrect | 1,792 / 2,207 |
| Macro-F1 | 44.81% |
| Macro-precision | 44.96% |
| Macro-recall | 44.81% |

| Class | Precision | Recall | F1 | Correct / support |
|---|---:|---:|---:|---:|
| `c1` | 46.46% | 41.29% | 43.72% | 519 / 1,257 |
| `c234` | 41.10% | 42.34% | 41.71% | 600 / 1,417 |
| `c5` | 47.33% | 50.79% | 49.00% | 673 / 1,325 |

Confusion matrix: rows are true classes, columns are predictions.

| True / predicted | `c1` | `c234` | `c5` |
|---|---:|---:|---:|
| `c1` | 519 | 440 | 298 |
| `c234` | 366 | 600 | 451 |
| `c5` | 232 | 420 | 673 |

Predicted class counts were 1,117 `c1`, 1,460 `c234`, and 1,422 `c5`.

## Comparison on identical held-out rows

| Adapter | Correct / 3,999 | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Original EP1 | 1,812 | 45.31% | 45.31% |
| Recipe 2, corrected mean-loss gradients | 1,426 | 35.66% | 35.59% |
| Recipe 3, corrected weighting and optimizer units | 1,792 | 44.81% | 44.81% |

Recipe 3 gains **366 correct predictions (+9.15 percentage points)** over
recipe 2, and loses **20 (-0.50 points)** versus the original. Saved baseline
predictions were reused; neither older adapter was executed again.

| Baseline | Both correct | Only baseline correct | Only recipe 3 correct | Both incorrect | Exact two-sided McNemar p |
|---|---:|---:|---:|---:|---:|
| Original EP1 | 1,723 | 89 | 69 | 2,118 | 0.130386 |
| Recipe 2 | 753 | 673 | 1,039 | 1,534 | 8.36777e-19 |

The first comparison does not establish a significant difference at 0.05 or
equivalence. The second nominally favors recipe 3. This holdout has been used
historically, and related/repeated texts limit independent-pair interpretation.
The measured recovery and stability support this correction in the observed
run, without establishing a general accuracy guarantee.

## Evaluation integrity and reproduction

The final exported adapter was evaluated once, in FP32 with batch 8 on the
RX 6650 XT and ROCm override `10.3.0`. Loading, tokenization, and inference
took **156.13 seconds**. Inputs, project sources, base cache, and trained
weights were mounted read-only in an offline container. All original holdout
rows were preserved; predictions were saved before reading their gold labels.

An independent standard-library ZIP/XML parser verified every gold label,
text hash, and original row ID, and recomputed the confusion matrix and class
metrics. Finite logits, softmax probabilities, argmax labels, input hashes,
and prediction/script hashes were checked. This verification did not perform
another model inference or any training.

Test1/test2 were not accessed, and the original frozen model and ensemble
configuration were preserved. These are holdout metrics; the original 48.00%
test1 accuracy concerns a different dataset.

Inside the validated ROCm image with EP1/cache mounts available, use a new
destination to reproduce:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=src \
python3 scripts/norberto/evaluate-holdout.py \
  --ep1-root /ep1 \
  --training-run results/norberto-development-recipe3 \
  --output-dir results/norberto-recipe3-holdout-another-new-run
```

Raw results are in `results/norberto-recipe3-holdout-20261105/`.
The [public aggregate](data/norberto-recipe3-holdout-results.json) includes full
metrics, fingerprints, training audits, regression tests, and paired comparisons.
