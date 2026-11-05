# Measured results

[Documentation index](../README.md)

Compare results only on the same dataset and preserve the training recipe
and partition recorded by each report. The original EP1 NorBERTo reference
has 48.00% accuracy on test1 and 45.31% on its frozen holdout.

## Test1: 900 labeled responses

| Model | Accuracy | Report |
|---|---:|---|
| Frozen EP1 SVM | 49.44% | [CPU classifier evaluation](lexical-test1.md) |
| Frozen EP1 logistic regression | 49.33% | [CPU classifier evaluation](lexical-test1.md) |
| Original EP1 NorBERTo | 48.00% | Reference documented in the [Qwen comparison](qwen-test1.md) |
| NorBERTo recipe 3, development-trained | 46.89% | [Saved test1 metrics](data/norberto-recipe3-test1-summary.json) |
| Qwen3-1.7B QLoRA | 46.67% | [Qwen evaluation](qwen-test1.md) |

The recipe-3 test1 row records 422/900 correct predictions (46.8889%, rounded
to 46.89%), read from the existing completed comparison. Its macro-F1 is
46.85%. No new inference was performed to assemble this index.

The table lists results documented with versioned aggregates. New NorBERTo
adapters are distinct models. Their separate test1 prediction/scoring
commands are in the [test1 guide](../guides/norberto-test1.md); a holdout score
must not be substituted for a test1 result.

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

JSON files in [`data/`](data/) contain the recorded metrics, hashes, execution
checks, and paired comparisons. They retain their original filenames and bytes.
Each detailed report links its own aggregate. Raw predictions, checkpoints,
and caches are local run artifacts and are not duplicated here.
