# Measured results

[Documentation index](../README.md)

Compare results only on the same dataset and preserve the training recipe
and partition recorded by each report. The original EP1 NorBERTo reference
has 48.00% accuracy on test1 and 45.31% on its frozen holdout.

## Test1: 900 labeled responses

| Model | Accuracy | Report |
|---|---:|---|
| SVM, full-corpus refit | 50.11% | [Verified full-corpus evaluations](data/full-test1-verification.json) |
| Frozen EP1 SVM | 49.44% | [CPU classifier evaluation](lexical-test1.md) |
| Frozen EP1 logistic regression | 49.33% | [CPU classifier evaluation](lexical-test1.md) |
| Original EP1 NorBERTo | 48.00% | Reference documented in the [Qwen comparison](qwen-test1.md) |
| NorBERTo recipe 3, development-trained | 46.89% | [Saved test1 metrics](data/norberto-recipe3-test1-summary.json) |
| Qwen3-1.7B QLoRA | 46.67% | [Qwen evaluation](qwen-test1.md) |
| NorBERTo recipe 3, full-corpus refit | 38.56% | [Verified full-corpus evaluations](data/full-test1-verification.json) |

The development-trained recipe-3 row records 422/900 correct predictions (46.8889%, rounded
to 46.89%), read from the existing completed comparison. Its macro-F1 is
46.85%. No new inference was performed to assemble this index.

The table lists results documented with versioned aggregates. New NorBERTo
adapters are distinct models. Their separate test1 prediction/scoring
commands are in the [test1 guide](../guides/norberto-test1.md); a holdout score
must not be substituted for a test1 result.

## Full-corpus refits: verified test1 results

Both models were trained on all 20,092 training rows. Metrics below were
independently recomputed on CPU from their existing saved predictions, with
all 900 gold/source response texts and row IDs aligned. Artifact, training,
prediction, and workbook hashes were checked. No model inference or retraining
was performed during verification.

| Model | Correct | Accuracy | Macro-F1 | Macro-precision | Macro-recall |
|---|---:|---:|---:|---:|---:|
| SVM, fixed C=0.05 | 451/900 | 50.11% | 49.87% | 49.93% | 50.11% |
| NorBERTo recipe 3 | 347/900 | 38.56% | 33.29% | 40.83% | 38.56% |

The full SVM gains six correct predictions (+0.67 percentage points) relative
to the development-trained frozen SVM (445/900). This descriptive difference
does not establish statistical significance.

The full NorBERTo loses 75 correct predictions (-8.33 percentage points)
relative to recipe 3 trained on the development partition (422/900). Its
prediction counts are `c1=216`, `c234=62`, and `c5=622`; gold has 300 rows per
class. Recall is 30.33% for c1, 9.00% for c234, and 76.33% for c5.

NorBERTo reached all 2,512 updates and two epochs in 40.20 minutes without a
budget stop. Completion did not produce improved classification quality.
The largest logged pre-clipping gradient norm was 1,196.86 at update 1,120,
compared with 22.58 in the development run; both use optimizer scale 4.
This spike and the prediction imbalance are consistent with unstable
optimization, but these checks do not establish its cause. No training
configuration or ensemble membership was changed based on test1 labels.

The [verification aggregate](data/full-test1-verification.json) includes full
metrics, confusion matrices, per-class scores, and model identities.

## NorBERTo frozen holdout: 3,999 responses

| Adapter | Accuracy | Macro-F1 | Report |
|---|---:|---:|---|
| Original EP1 | 45.31% | 45.31% | Reference in the recipe-3 comparison |
| Recipe 3, optimizer-compatible | 44.81% | 44.81% | [Completed validation](norberto-recipe3-holdout.md) |
| Recipe 2, mean-loss gradient units | 35.66% | 35.59% | [Historical validation](../archive/norberto/recipe2-development-holdout.md) |

Full-corpus adapters include the former holdout in training and cannot use
these rows as unseen validation. Older full-corpus test1 evaluations and
training investigations are listed in the [archive](../archive/README.md).

## Technical aggregates

[Dataset overlap audit](dataset-overlap.md) verifies the reused training corpus,
test1 identity, and exact/normalized text overlap across training and both tests.

JSON files in [`data/`](data/) contain the recorded metrics, hashes, execution
checks, and paired comparisons. They retain their original filenames and bytes.
Each detailed report links its own aggregate. Raw predictions, checkpoints,
and caches are local run artifacts and are not duplicated here.
