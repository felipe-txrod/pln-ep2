# NorBERTo development-only refit: frozen holdout validation

The corrected-loss adapter trained on **16,093 development responses** achieved
**35.66% accuracy (1,426/3,999)** and **35.59% macro-F1** on the original EP1
holdout. The original EP1 adapter achieved **45.31% (1,812/3,999)** on these
same rows. This is a decrease of **386 correct predictions, or 9.65 percentage
points**. Training completed, but the resulting classifier underperformed the
original adapter on this validation set.

## Training completion and partition integrity

`results/norberto-development-normalized/` completed two fixed epochs and all
**2,012 optimizer updates**, taking **1,916.67 seconds (31.94 minutes)**. There
was no time-budget interruption, validation, or checkpoint selection during
training. The adapter was initialized from the pinned public base, independently
of the original trained adapter.

The frozen split contains 16,093 training rows in 12,956 groups and 3,999 holdout
rows in 3,281 groups, with **zero row or group overlap**. All saved training row
IDs, text hashes, and labels match the source workbook. Source implementations
and exported file hashes passed verification. All 272,643 saved trainable
parameters are finite, and all 22 LoRA B tensors contain nonzero updates.

Recipe version 2 uses sample-normalized cross entropy across accumulation.
The numerical hyperparameters remain the original effective two-epoch recipe:
LoRA rank 4, alpha 8, dropout 0.05, `Wqkv`, context 512 with head/tail truncation,
FP32, SDPA, learning rate 0.001, batch 4, accumulation 4, weight decay 0.01,
warmup ratio 0.1, and seed 42. The public base revision is
`db73446f89c96044863ea05a39f680524b84bccb`.

Adapter weights SHA-256:

```text
2af7484a44ce6a09a9fec3d2ad36216b0749ffa7e95d03a76bddecccd70d0aec
```

## Holdout metrics

| Metric | Result |
|---|---:|
| Accuracy | 35.66% |
| Correct / incorrect | 1,426 / 2,573 |
| Macro-F1 | 35.59% |
| Macro-precision | 35.69% |
| Macro-recall | 35.58% |

| Class | Precision | Recall | F1 | Correct / support |
|---|---:|---:|---:|---:|
| `c1` | 32.10% | 34.45% | 33.23% | 433 / 1,257 |
| `c234` | 37.40% | 38.32% | 37.85% | 543 / 1,417 |
| `c5` | 37.56% | 33.96% | 35.67% | 450 / 1,325 |

Confusion matrix: rows are true classes, columns are predicted classes.

| True / predicted | `c1` | `c234` | `c5` |
|---|---:|---:|---:|
| `c1` | 433 | 450 | 374 |
| `c234` | 500 | 543 | 374 |
| `c5` | 416 | 459 | 450 |

Predicted class counts were 1,349 `c1`, 1,452 `c234`, and 1,198 `c5`.

## Matched comparison with the original EP1 adapter

The original adapter's saved holdout predictions were reused and checked
against the same row IDs, gold labels, source hash, and split hash. No new
inference was performed for that adapter.

| Adapter | Training rows | Loss recipe | Correct / 3,999 | Accuracy | Macro-F1 |
|---|---:|---|---:|---:|---:|
| Original EP1 adapter | 16,093 | Legacy | 1,812 | 45.31% | 45.31% |
| Development-only refit | 16,093 | Sample-normalized | 1,426 | 35.66% | 35.59% |

Both adapters were correct on 756 rows. Only the original was correct on
1,056 rows; only the refit was correct on 670 rows; both were incorrect on
1,517 rows. The exact two-sided McNemar p-value is **1.33114e-20**, nominally
favoring the original. This is a historically reused, grouped holdout; repeated
or related responses limit independent-pair interpretation.

These are holdout results. The original adapter's 48.00% test1 accuracy and
the full-corpus refits' test1 scores concern a different dataset and are not
direct comparators here. Full-corpus adapters trained on these holdout rows,
so those adapters cannot use this set as unseen validation.

## Training behavior and interpretation

| Epoch interval | Mean logged loss | Mean pre-clipping gradient norm | Maximum norm |
|---|---:|---:|---:|
| 0–0.5 | 1.11957 | 2.46840 | 4.64416 |
| 0.5–1 | 1.17576 | 52.90237 | 275.30652 |
| 1–1.5 | 1.13091 | 90.69721 | 186.92047 |
| 1.5–2 | 1.13462 | 114.22054 | 191.36064 |

These summaries use the 100 logged update samples, 25 per interval. Gradient
clipping remained enabled. The gradients escalated despite corrected loss
normalization, and the loss did not consistently decrease. The correction
addresses the confirmed accumulation bug; it does not guarantee stable
optimization or recover the original classification quality for every run.
The precise trigger of this regression remains unisolated. This validation
does not establish that the normalization change alone caused the decrease.
See the [earlier investigation](investigacao-norberto-full.md) for the inherited
bug and the limits of causal attribution.

The subsequent [optimizer and backend audit](norberto-instability-audit.md)
checks the complete corrected Trainer loop in CPU and GPU execution, confirms
large derivatives on CPU, and localizes the sensitivity to the first attention
LoRA. It also explains why the changed loss scale alters the clipping regime.

## Execution and independent verification

The [evaluation script](../scripts/evaluate-norberto-holdout.py) loaded the new
adapter and predicted all 3,999 held-out texts once in FP32, batch 8, on the
RX 6650 XT with ROCm override `10.3.0`. Loading, tokenization, and inference
took **163.31 seconds**. Peak allocated/reserved VRAM was **951.81/1,270 MiB**.
The container was offline, with project files, EP1 sources, and the base cache
mounted read-only. Predictions were saved before reading gold labels.

An independent standard-library ZIP/XML parser verified all gold labels and
original row IDs. Metrics, confusion counts, finite logits, normalized
probabilities, argmax predictions, and source/split/prediction/script hashes
were reconciled without repeating model inference or training.

Test1 and test2 were not accessed. The original frozen adapter and ensemble
configuration were preserved; this result did not select a replacement model.

To reproduce inside the validated ROCm image with EP1 at `/ep1` and its Hugging
Face cache mounted at the configured cache location, choose a new output path:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=src \
python3 scripts/evaluate-norberto-holdout.py \
  --ep1-root /ep1 \
  --training-run results/norberto-development-normalized \
  --output-dir results/norberto-development-holdout-another-new-run
```

Raw results are under `results/norberto-development-holdout-20261105/`.
The [public result aggregate](norberto-development-holdout-results.json)
contains full metrics, provenance, training audits, gradient summaries, and
the paired comparison.
