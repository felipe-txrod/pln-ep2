# Qwen3-1.7B evaluation on labeled test1

[Documentation index](../README.md)

The completed QLoRA NF4 adapter achieved **46.67% accuracy (420/900)** and
**46.44% macro-F1** on all original test1 rows. The workbook, row IDs, and text
hashes were preserved. Inference consumed only texts from `data/test1.xlsx`;
gold labels were read from `test1-rotulado-para-teste-estatistico.xlsx` after
predictions had been persisted. No training or parameter selection occurred.

## Training completion and adapter integrity

The run in `results/qwen-fast-training/` completed three epochs and all 3,003
updates in 11,222.19 seconds, approximately 3 h 7 min. No FP16 overflow steps
were skipped. All 3,217,408 saved trainable parameters were finite; all 112 LoRA
B tensors contained nonzero updates. The trained classification head was present.

| Epoch | Internal validation accuracy | Internal validation macro-F1 |
|---|---:|---:|
| 1 | 43.53% | 34.96% |
| 2 | 42.28% | 39.94% |
| 3 | **45.84%** | **45.52%** |

The selected adapter was `checkpoint-3003`, chosen using internal validation
accuracy. Its weights match the exported adapter byte-for-byte, with SHA-256:

```text
099fa94ccd7e9d40c7a0e0110154e64c5b5bc7430d71c1f0063bee589e935b50
```

The base revision is `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` of
`Qwen/Qwen3-1.7B`. Training used 16,010 responses and reserved 4,082 for grouped
internal validation. It did not refit on all 20,092 training responses.

## Test1 metrics

| Metric | Result |
|---|---:|
| Accuracy | 46.67% |
| Correct / incorrect | 420 / 480 |
| Macro-F1 | 46.44% |
| Macro-precision | 46.43% |
| Macro-recall | 46.67% |

| Class | Precision | Recall | F1 | Correct / support |
|---|---:|---:|---:|---:|
| `c1` | 50.00% | 49.00% | 49.49% | 147 / 300 |
| `c234` | 41.20% | 36.67% | 38.80% | 110 / 300 |
| `c5` | 48.08% | 54.33% | 51.02% | 163 / 300 |

Confusion matrix: rows are true labels and columns are predictions.

| True / predicted | `c1` | `c234` | `c5` |
|---|---:|---:|---:|
| `c1` | 147 | 85 | 68 |
| `c234` | 82 | 110 | 108 |
| `c5` | 65 | 72 | 163 |

## Descriptive comparison with saved models

| Model | Correct / 900 | Accuracy |
|---|---:|---:|
| SVM | 445 | 49.44% |
| Logistic regression | 444 | 49.33% |
| Original EP1 NorBERTo submission | 432 | 48.00% |
| Qwen3-1.7B QLoRA | 420 | 46.67% |

Qwen had 12 fewer correct predictions than the original NorBERTo submission,
a difference of −1.33 percentage points. Both were correct on 310 rows;
only NorBERTo was correct on 122 and only Qwen on 110. Both were incorrect on
358. The exact two-sided McNemar p-value was **0.470258**, which does not show
a statistically significant difference at 0.05 and does not establish equivalence.
Repeated texts limit independence assumptions in this paired comparison.

The original NorBERTo predictions were reused; NorBERTo was not executed again.
The later full-corpus NorBERTo refit is a different model and is not represented
by this baseline. SVM and logistic regression results come from the
[saved CPU evaluation](lexical-test1.md).

This evaluation is an isolated comparison. No test2 run or delivery preceded
it for this adapter, so it does not establish completion of the prescribed
test2-then-test1 submission sequence. Test1 has also been inspected previously.
Do not use these results to tune ensemble members, thresholds, prompts, or
arbitration rules while describing the same test as an untouched final evaluation.

## Execution, verification, and reproduction

Inference reloaded the exported adapter on the RX 6650 XT with the validated
ROCm 6.4 stack and `HSA_OVERRIDE_GFX_VERSION=10.3.0`. A fresh NF4 kernel probe
passed. The container had no network access, the project was mounted read-only,
and only a new evaluation directory was writable. The model used its existing
one-response inference batches. Loading and inference took **235.75 seconds**;
peak PyTorch allocated/reserved VRAM was 1,640.87 / 1,746 MiB.

An independent standard-library ZIP/XML reader checked all 900 gold labels.
Row IDs, confusion counts, per-class metrics, probability sums, finite logits,
argmax predictions, and source/prediction hashes matched the saved report.
The original training run and adapter were unchanged.

The reusable comparison entry point is
[`scripts/qwen/evaluate-test1.py`](../../scripts/qwen/evaluate-test1.py). Inside the
validated ROCm environment, with the original base cache available:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
PYTHONPATH=.cache/qwen-runtime:src \
python3 scripts/qwen/evaluate-test1.py \
  --training-run results/qwen-fast-training \
  --output-dir results/qwen-test1-another-new-run
```

Use a new output directory. Rerunning performs new inference; the saved results
are sufficient for further analysis without executing a model again.

Complete public aggregate metrics, configuration, runtime versions, provenance
hashes, and the paired comparison are in
[`qwen-test1-results.json`](data/qwen-test1-results.json). Raw predictions and reports
remain in `results/qwen-test1-evaluation-20261104-225455-verified/qwen/`.
