# Measured results

[Documentation index](../README.md)

Compare results only on the same dataset and preserve the training recipe
and partition recorded by each report. The original EP1 NorBERTo reference
has 48.00% accuracy on test1 and 45.31% on its frozen holdout.

## Selected Qwen: frozen EP1 partition

The completed 16,093/3,999 run is now the default Qwen model in
`models/qwen/adapter/`. It completed three epochs and 3,018 updates; validation
scored **1,791/3,999 (44.79%)**, macro-F1 **44.44%**. All source rows, selected
checkpoint, package checksums and 225 finite tensors were checked, followed
by real GPU reload on synthetic texts. The original per-row validation export
had a length-grouping order bug; exact checkpoint-RNG recovery reproduced
the aggregate metrics and saved aligned records separately. Qwen remains
excluded from the default ensemble. See the [model and audit](qwen-ep1-model.md).

## Stacking versus weighted soft voting

The [stacking comparison](ensemble-stacking.md) reuses the same frozen models,
calibrators and 1,972 development rows as soft-weight selection. Five grouped
stratified folds selected meta-LR C=0.1 from four fixed candidates using a
one-standard-error rule. It scored **442/900 (49.11%)**, macro-F1 **49.03%**,
below weighted soft voting at 447/900 (49.67%), macro-F1 49.58%. It corrected
40 soft errors and harmed 45 correct cases; exact McNemar p=0.664645.
The report includes per-class F1, confusion matrices and affected Excel rows.
No base model was refitted, no test-label tuning occurred and earlier artifacts
were preserved. This remains an exploratory comparison with historical exposure.

## Local majority ensemble: test1 comparison

The [weighted soft-voting comparison](ensemble-soft.md) is a separate frozen
method: 2,027 holdout rows calibrate probabilities and 1,972 select weights by
macro-F1 from 16 candidates. It scores 447/900 (49.67%) and 49.58% macro-F1,
using SVM 0.75, LR 0.00 and NorBERTo 0.25. The report includes all individual
confusion matrices, calibration effects and inherited cross-dataset text overlap.
The [direct comparison with the actual EP1 delivery](ensemble-soft.md#conferência-contra-a-entrega-efetiva-do-ep1)
confirms all 900 original labels match the saved NorBERTo predictions. Exact
two-sided McNemar gives p=0.303327: the numerical gain is not statistically
significant at 0.05. The official test2-before-test1 order was not established.

The user subsequently excluded Qwen; active defaults are now NorBERTo + SVM +
RL, strict majority, with RL arbitration. The [holdout isolation audit and
results](ensemble-holdout.md) confirm all 3,999 holdout rows and their groups
were excluded from the selected models' fitting. NorBERTo scores 1,812/3,999
(45.31%); RL and the best trio each score 1,803/3,999 (45.09%); SVM scores
1,792/3,999 (44.81%).

The best NorBERTo + SVM + RL majority used RL for three-way ties: **443/900
(49.22%)**, below RL alone at 447/900 (49.67%). All three use development-only
artifacts with holdout reserved. Fifteen fixed voting configurations were
compared as an explicitly exploratory study, yielding six distinct prediction
sequences. See the [results and reproduction command](ensemble-majority.md)
and [independently verified aggregate](data/ensemble-majority-test1.json).

## Selected NorBERTo: unchanged EP1 delivery

The selected artifact is `models/norberto/final_model/`, SHA-256
`d693f6e2995d117ad04a5692817527764fcccee6bfb2dde99120baf9fdd2d385`.
It retains the original adapter, trained head, tokenizer, base revision,
512-token head/tail representation, FP32/SDPA, and original settings. Its
final training used two epochs on 16,093 development rows. No loss correction,
new adapter training, or convex-head replacement is active.

Restoration was verified by predicting all 900 test1 texts again: every label
matches the EP1 delivery, with 432 correct (48.00%) and 47.86% macro-F1.
The maximum probability difference from the original saved CSV is 8.60e-8.
An independent ZIP/XML reader confirmed all texts, gold labels, row IDs,
metrics, and the original adapter/tokenizer/configuration identity. Historical
holdout accuracy remains 45.31%; no new holdout inference was needed.

The source cleanup retired 34 experimental files, preserving a local source
snapshot and all previous result/model/cache directories. Test1 commands and
the ensemble now select the original model explicitly. The available suite
passed 115 tests; two optional test modules require the unavailable `openai`
SDK. No model was retrained and no test2 prediction or delivery was generated.
See the [restoration record](data/norberto-ep1-restoration.json) and
[current workflow](../guides/norberto-test1.md).

## Ensemble defaults: development models with holdout reserved

NorBERTo and SVM select the original 16,093-row development artifacts. RL now
defaults to a development-only bundle under `models/rl-development/`. The user
completed training: fixed C=0.25, 38 iterations, 25.08 seconds, with 3,999 holdout
rows excluded. Subsequent verification checked all 16,093 training identities,
model/source hashes, zero group overlap, default loading, and synthetic
inference. See the [RL verification](data/rl-development-verification.json).
The historical 20,092-row RL remains archived in `models/rl/` and is not a
fallback. The subsequent RL evaluation is recorded below.

Both ensemble configurations pin the selected NorBERTo/SVM/RL manifests and
reject saved predictions from different artifacts. Historical Qwen used a
16,010/4,082 split; its newly selected package uses the same EP1 16,093/3,999
split and remains excluded. The user subsequently
selected NorBERTo + SVM + RL majority voting with RL arbitration as the default. The
available suite passed 121 tests, including a synthetic logistic fit verifying
that holdout-only vocabulary is excluded and the exported bundle reloads.
SVM test1 predictions were regenerated only to record the current runtime:
all 900 labels match the previous development model. No new scoring, test2
inference, or remote request occurred.

See the [configuration record](data/ensemble-development-defaults.json) and
[RL command guide](../../src/rl_classifier/README.md).

## Development-only RL: holdout and test1

The saved C=0.25 logistic model was evaluated without refitting or parameter
selection. Training used 16,093 development rows; all 3,999 holdout rows and
their similarity components remained outside fitting.

| Dataset | Correct | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Frozen EP1 holdout | 1,803 / 3,999 | 45.09% | 44.78% |
| Test1 against the labeled workbook | 447 / 900 | 49.67% | 49.32% |

Both datasets' text-only predictions were saved before loading gold for
scoring. An independent ZIP/XML source reader verified all 4,899 row IDs,
texts, and labels, manually recomputed the confusion matrices and metrics,
and reconstructed probabilities from decision scores (maximum error 2.23e-16).
Every holdout label matches the historical EP1 lexical logistic comparison.
All model-file hashes remained unchanged. Test1 gains three correct predictions
over the historical full-corpus RL (444/900); this descriptive difference does
not establish a general accuracy advantage for less training data.

Raw results, predictions, per-row comparisons, and evaluator/verifier snapshots
are under `results/rl-development-evaluation-20261105/`. The selected ensemble
test1 run is now `results/rl-development-test1`, with the development-model
identity verified. No test2 prediction, remote call, or new training occurred.
The holdout and test1 have been consulted historically; this remains a
descriptive evaluation. See the [verified aggregate](data/rl-development-evaluation.json).

## Test1: 900 labeled responses

| Model | Accuracy | Report |
|---|---:|---|
| SVM, full-corpus refit | 50.11% | [Verified full-corpus evaluations](data/full-test1-verification.json) |
| RL, development-only with holdout reserved | 49.67% | [Verified RL evaluation](data/rl-development-evaluation.json) |
| Frozen EP1 SVM | 49.44% | [CPU classifier evaluation](lexical-test1.md) |
| Historical EP1 logistic regression, full corpus | 49.33% | [CPU classifier evaluation](lexical-test1.md) |
| Original EP1 NorBERTo | 48.00% | Reference documented in the [Qwen comparison](qwen-test1.md) |
| NorBERTo recipe 3, development-trained | 46.89% | [Saved test1 metrics](data/norberto-recipe3-test1-summary.json) |
| Qwen3-1.7B QLoRA | 46.67% | [Qwen evaluation](qwen-test1.md) |
| NorBERTo recipe 3, full-corpus refit | 38.56% | [Verified full-corpus evaluations](data/full-test1-verification.json) |
| EP1 CV winner candidate-08, development final adapter | 34.78% | [Reloaded adapter evaluation](data/norberto-ep1-candidate08-evaluation.json) |

The development-trained recipe-3 row records 422/900 correct predictions (46.8889%, rounded
to 46.89%), read from the existing completed comparison. Its macro-F1 is
46.85%. No new inference was performed to assemble this index.

The table lists results documented with versioned aggregates. New NorBERTo
adapters are distinct models. Their separate test1 prediction/scoring
commands are in the [test1 guide](../guides/norberto-test1.md); a holdout score
must not be substituted for a test1 result.

## Historical NorBERTo experiments

The following experiments are retained as evidence. Their corrections, refits,
and logistic-head workflows have been retired and are not used by the selected
EP1 model.

### Frozen NorBERTo logistic: EP1 cross-validation

The public encoder was frozen in evaluation mode and its 768-dimensional
masked-mean document vectors were L2-normalized. Only the logistic head was
fitted. All 16,093 development rows use **exactly the original NorBERTo EP1
outer and inner row partitions**; the 3,999 holdout rows were excluded before
tokenization. This protocol has three outer folds and the first inner holdout
from five grouped splits, without an outer-training refit.

| Model on the same development outer folds | Correct / 16,093 | Pooled accuracy | Pooled macro-F1 |
|---|---:|---:|---:|
| Frozen public NorBERTo + logistic regression | 6,850 | 42.57% | 42.20% |
| Original EP1 NorBERTo LoRA | 6,995 | 43.47% | 43.22% |
| EP1 lexical logistic, matched inner training rows | 7,078 | 43.98% | 43.51% |
| EP1 lexical logistic, complete outer training rows | 7,080 | 43.99% | 43.54% |

Historical controls reuse the saved EP1 OOF predictions after checking all
row IDs, labels, groups, and folds. They were not rerun. The new candidate
loses 145 correct predictions (-0.90 percentage points) relative to LoRA.
Its descriptive paired similarity-group bootstrap interval for that delta is
[-1.84, +0.01] points; this does not establish equivalence. The fixed folds and
historically reused holdout make the experiment exploratory.

| Fold | Inner-selected C | Outer accuracy | Outer macro-F1 |
|---|---:|---:|---:|
| 1 | 1 | 42.97% | 42.24% |
| 2 | 10 | 42.81% | 42.64% |
| 3 | 10 | 41.95% | 41.48% |

All nine prespecified L-BFGS fits converged in 46–372 iterations. The final
development classifier uses the predeclared median C=10 and converged in
134 iterations. Outer scores did not select that parameter. The encoder had
zero trainable parameters, zero optimizer updates, and an identical loaded
state SHA-256 before and after extraction. This removes encoder optimization
from this pipeline; it does not guarantee monotonic accuracy with more data.

Extraction encoded 14,469 unique texts in 252.05 seconds, with peak allocated
VRAM 1,300.62 MiB. Head CV took 34.67 seconds and final fitting 5.38 seconds
on CPU. The final safe NPZ head and manifest are in
`results/norberto-frozen-logistic-ep1-20261105/model/`; the original models and
ensemble configuration were preserved.

An independent ZIP/XML reader verified all OOF source texts, labels, row IDs,
and metrics. Direct reconstruction of probabilities from all three exported
heads matched saved OOF probabilities exactly. Cold inference, safe reload,
cache ordering, and the prediction CLI passed on three synthetic responses.
The subsequent holdout evaluation is recorded below. The new classifier has
not been scored on test1 or test2.

The available regression suite passed 153 tests, including the new pipeline
checks; two unrelated modules could not be collected because the ROCm image
lacks the optional `openai` SDK. An existing reference test separately checked
the historical EP1 test1 predictions; those labels did not enter this model's
selection, fitting, or evaluation.

See the [technical aggregate](data/norberto-frozen-logistic-ep1.json) and the
[reproduction guide](../archive/norberto/frozen-logistic.md).

### Frozen NorBERTo logistic: unchanged classifier on EP1 holdout

The saved C=10 development classifier was evaluated on all 3,999 reserved
holdout rows, without refitting, parameter selection, or adapter loading.
The 16,093 development rows and 12,956 development similarity components are
disjoint from the 3,999 holdout rows and 3,281 holdout components. Source and
split hashes match training. Every prediction was saved before constructing
gold-bearing records for scoring. Classifier, manifest, and encoder state
hashes remained unchanged.

| Model on the same holdout rows | Correct / 3,999 | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Frozen public NorBERTo + logistic regression | 1,694 | 42.36% | 42.08% |
| Original EP1 NorBERTo LoRA | 1,812 | 45.31% | 45.31% |
| EP1 lexical logistic regression | 1,803 | 45.09% | 44.78% |
| Frozen EP1 lexical SVM | 1,792 | 44.81% | 44.47% |
| NorBERTo LoRA recipe 3, development-trained | 1,792 | 44.81% | 44.81% |

Historical controls reuse saved predictions after checking all row IDs and
groups/gold labels, or source text hashes for the recipe-3 JSONL. No baseline
was retrained or rerun. The new candidate loses 118 correct predictions
(-2.95 percentage points) to original LoRA; the descriptive paired
similarity-component bootstrap interval is [-5.14, -1.00] points (2,000 draws,
seed 42). This historically consulted holdout remains exploratory.

Holdout accuracy is 0.20 percentage points below the 42.57% development OOF
score. That proximity does not establish stability when training size changes:
the final head was not refitted during this evaluation. The fixed encoder
avoids adapter optimization, but the representation has lower observed
classification quality than the controls.

The confusion matrix, with true classes in rows and predicted classes in
columns ordered `c1, c234, c5`, is `[[497,394,366], [396,477,544], [253,352,720]]`.
Recall is 39.54%, 33.66%, and 54.34%, respectively. Extracting 3,705 unique
holdout texts took 72.90 seconds with peak allocated VRAM 1,300.62 MiB; the
complete evaluation took 75.00 seconds, offline. Encoder trainable parameters
and optimizer updates were both zero.

An independent ZIP/XML workbook reader verified every source text, gold
label, row ID, and frozen group. All 3,999 probability vectors were recomputed
directly from the exported safe NPZ head and saved embeddings, with maximum
absolute error zero. Manual confusion-matrix, accuracy, F1, recall, and paired
control counts matched the report. Artifacts are preserved separately in
`results/norberto-frozen-logistic-holdout-20261105/`; the original training
manifest remains an immutable record of the earlier development-only run.
No new official test evaluation or full-corpus fit was performed.

See the [holdout aggregate](data/norberto-frozen-logistic-holdout.json) and
[reproduction command](../archive/norberto/frozen-logistic.md#holdout-evaluation).

### EP1 CV winner: candidate-08 reloaded

EP1's saved three-finalist CV ranked `candidate-08` first, with mean accuracy
43.73% (LoRA rank 16, alpha 32, learning rate 0.0003). At the user's request,
the preserved final adapter was reloaded and evaluated on holdout and test1.
This is the adapter trained on all 16,093 development rows for three epochs,
selected by the median inner-selected epochs `2, 3, 3`; it is not a fold
checkpoint. No retraining, parameter search, or model promotion occurred.

| Evaluation | Candidate-08 correct | Candidate-08 accuracy | Candidate-08 macro-F1 | Original EP1 accuracy |
|---|---:|---:|---:|---:|
| Frozen holdout, 3,999 rows | 1,439 | 35.98% | 35.31% | 45.31% (1,812 correct) |
| Test1, 900 rows | 313 | 34.78% | 34.30% | 48.00% (432 correct) |

Test1 was scored against `data/test1-rotulado-para-teste-estatistico.xlsx`,
after checking every row ID and response text against the unlabeled input.
All predictions for both datasets were saved before constructing gold-bearing
records. The final adapter files and complete loaded model state were
unchanged, with zero trainable parameters, parameter gradients, or optimizer
updates during inference. Training and holdout similarity groups were disjoint.

Independent ZIP/XML source parsing, manual metrics, and probability
reconstruction from saved logits verified all 4,899 rows. Every holdout
prediction matched the historical candidate-08 evaluation (maximum probability
difference 2.97e-8). The original delivered-model controls were recomputed
from their saved predictions. Candidate-08 loses 373 correct holdout predictions
(-9.33 percentage points) and 119 test1 predictions (-13.22 points) relative
to that original model. Its CV advantage did not generalize to the final
adapter. The historically consulted holdout remains exploratory.

The run took 200.64 seconds offline, with 152.12 seconds for holdout and 33.74
seconds for test1, including dataset tokenization. Peak allocated GPU memory
was 954.91 MiB. Six relevant regression checks passed. Predictions, metrics,
and per-row gold comparisons are under
`results/norberto-ep1-candidate08-evaluation-20261105/evaluation/`.

See the [verified aggregate](data/norberto-ep1-candidate08-evaluation.json) and
[reproduction guide](../archive/norberto/candidate08-evaluation.md).

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

### Training-scale audit: recipe 3

This retired experiment is preserved in the
[historical audit](../archive/norberto/recipe3-scale-audit.md). The measured
300-row training-sample accuracy of 52% was not held-out validation accuracy.
The original EP1 delivery model is selected; none of these training changes
are active.

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
