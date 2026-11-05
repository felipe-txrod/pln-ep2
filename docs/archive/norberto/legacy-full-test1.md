# Full-corpus NorBERTo evaluation on labeled test1

[Documentation index](../../README.md) · Historical experiment record; use the guides for current commands.

The subsequent [training-only investigation](regression-investigation.md)
documents optimization instability, a reproduced loss-normalization bug,
and the tested correction for future training. Historical results below remain unchanged.

The NorBERTo adapter refitted on all **20,092 EP1 training responses** achieved
**37.89% accuracy (341/900)** and **36.47% macro-F1** on test1. The original EP1
submission achieved 48.00% (432/900). Full-corpus refitting completed technically,
but its observed test1 quality decreased by **91 correct predictions**, or
**10.11 percentage points**. The original adapter remains unchanged.

## Training and artifact verification

The run in `results/norberto-full-training/` completed both configured epochs
and all **2,512 optimizer updates** in **2,389.49 seconds (39.82 minutes)**.
It did not stop because of the 75-minute budget. The effective configuration
matches the original two-epoch EP1 recipe: LoRA rank 4/alpha 8/dropout 0.05,
`Wqkv`, context 512 with head/tail truncation, FP32, SDPA, learning rate 0.001,
batch 4, accumulation 4, weight decay 0.01, warmup 0.1, and seed 42.

All 20,092 saved training row IDs, text hashes, and labels were reconciled
against the unchanged EP1 training workbook. The former holdout was included;
test1/test2 were not training inputs, and no internal validation was performed.
The inherited configuration retains historical split descriptors; the full-data
runner does not apply that split.

The new adapter has **272,643 finite parameters**, 22 nonzero LoRA B tensors,
and the trained classification weight and bias. Its SHA-256 is:

```text
ea0bd1ef02516e0de768dfb8d76313c3a8cacd25d43d9f4431c5c35a4b2c86be
```

All exported file hashes were verified. The public NorBERTo base checksum
matches the pinned revision `db73446f89c96044863ea05a39f680524b84bccb`.
A separate CPU-only loading check compared all **46 loaded adapter tensors**
against the safetensors file using exact equality. Every tensor, including the
classification head, matched; the class mapping was also correct. This check
performed no inference or training.

## Test1 results

Inference read only the texts from `data/test1.xlsx`. Predictions were saved
before opening the gold workbook `test1-rotulado-para-teste-estatistico.xlsx`.
All 900 rows, texts, and original IDs were preserved and aligned.

| Metric | Result |
|---|---:|
| Accuracy | 37.89% |
| Correct / incorrect | 341 / 559 |
| Macro-F1 | 36.47% |
| Macro-precision | 37.27% |
| Macro-recall | 37.89% |

| Class | Precision | Recall | F1 | Correct / support |
|---|---:|---:|---:|---:|
| `c1` | 34.57% | 18.67% | 24.24% | 56 / 300 |
| `c234` | 38.48% | 49.00% | 43.11% | 147 / 300 |
| `c5` | 38.76% | 46.00% | 42.07% | 138 / 300 |

Confusion matrix: rows are true labels and columns are predictions.

| True / predicted | `c1` | `c234` | `c5` |
|---|---:|---:|---:|
| `c1` | 56 | 129 | 115 |
| `c234` | 50 | 147 | 103 |
| `c5` | 56 | 106 | 138 |

The model predicted `c1` 162 times, `c234` 382 times, and `c5` 356 times.
The largest loss relative to the original submission was in `c1`:

| True class | Original EP1 correct | Full refit correct | Difference |
|---|---:|---:|---:|
| `c1` | 118 | 56 | −62 |
| `c234` | 155 | 147 | −8 |
| `c5` | 159 | 138 | −21 |

## Paired comparison and interpretation

The original EP1 predictions were reused without executing the original model:
both models were correct on 192 rows, only the original on 240, only the full
refit on 149, and neither on 319. Exact two-sided McNemar yielded
**p = 0.000004585**, in the direction favoring the original model. Repeated
texts and prior inspection of test1 limit interpretation as an independent
confirmatory comparison.

| Saved model | Correct / 900 | Accuracy |
|---|---:|---:|
| SVM | 445 | 49.44% |
| Logistic regression | 444 | 49.33% |
| Original EP1 NorBERTo | 432 | 48.00% |
| Qwen3-1.7B QLoRA | 420 | 46.67% |
| NorBERTo full-corpus refit | 341 | 37.89% |

The checks establish that the requested corpus and recipe were used and the
new weights were loaded correctly. They do not establish the cause of the
performance decrease. More training responses change the number of updates,
sampling order, and optimization trajectory even when hyperparameters remain
fixed. No causal attribution to a particular change is supported by this run.
No retraining, checkpoint search, threshold adjustment, model replacement, or
ensemble modification was performed during evaluation.

This is an isolated comparison without a preceding test2 run for this adapter.
The historical holdout is now part of this model's training data and cannot
serve as its unseen validation set. Test1 metrics must not guide a new rule
while the same test is presented as an untouched final evaluation.

## Execution and reproduction

The adapter was evaluated once on the RX 6650 XT, using the original EP1
inference construction and batch size 8. The ROCm override was `10.3.0`.
The container had no network access; project files, EP1 sources, and the base
cache were mounted read-only. Loading, tokenization, and inference took
**34.47 seconds**. Peak PyTorch allocated/reserved VRAM was
**951.81 / 1,136 MiB**.

The reusable entry point is
[`scripts/norberto/evaluate-test1.py`](../../../scripts/norberto/evaluate-test1.py).
It requires the original EP1 source and pinned base cache. Inside the validated
ROCm container, with EP1 mounted at `/ep1` and its Hugging Face cache mounted
at `/workspace/.cache/huggingface`:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=src \
python3 scripts/norberto/evaluate-test1.py \
  --ep1-root /ep1 \
  --training-run results/norberto-full-training \
  --output-dir results/norberto-full-test1-another-new-run
```

An independent standard-library ZIP/XML reader reconciled all gold labels,
confusion counts, per-class scores, probability sums, logits/argmax predictions,
and source/prediction hashes. Raw reports remain in
`results/norberto-full-test1-20261105-003353/`; the full public aggregate is
[`norberto-full-test1-results.json`](../../results/data/norberto-full-test1-results.json).
