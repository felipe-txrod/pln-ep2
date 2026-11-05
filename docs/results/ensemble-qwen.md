# Qwen integration into soft voting and stacking

[Results index](README.md) · [Commands](../guides/ensemble-qwen.md)
· [Verified aggregate](data/ensemble-qwen.json)

The selected EP1-split Qwen was added to both ensemble families, with its
base weights unchanged. The integration preserves all original three-model
artifacts and uses new packages under `models/ensemble-soft-qwen/` and
`models/ensemble-stacking-qwen/`.

## Test1 comparison

All 900 original texts, row IDs and positions were retained. Predictions
were saved before reading the labeled workbook, after calibration, weights
and regularization were frozen.

| Method | Correct / 900 | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Soft voting, three members | 447 | 49.67% | 49.58% |
| Soft voting, four candidates including Qwen | 447 | 49.67% | 49.58% |
| Stacking, three members | 442 | 49.11% | 49.03% |
| Stacking, four members including Qwen | 440 | 48.89% | 48.80% |
| Selected Qwen alone | 414 | 46.00% | 45.53% |

The soft search selected **SVM 75%, RL 0%, NorBERTo 25%, Qwen 0%**. Qwen was
scored, calibrated, and included in all relevant candidates, but the selected
weighted sum gives it no contribution. All 900 final labels equal the
three-member soft result. Inclusion did not improve its measured quality.

The stacking model uses all 12 calibrated probabilities and selected C=0.1.
Qwen's learned coefficients are nonzero. Relative to the original stacking,
it corrects 25 errors and harms 27 correct answers, losing two net correct
answers, or 0.22 percentage points. Relative to soft voting, it corrects 47
errors and harms 54 correct answers; 30 additional changes remain wrong.

Confusion matrices use true classes in rows and predicted classes in columns,
ordered `c1, c234, c5`:

```text
Soft with four candidates     Stacking with four members
[[139, 93, 68],              [[116, 125, 59],
 [ 74,135, 91],               [ 55, 158, 87],
 [ 50, 77,173]]               [ 36, 98,166]]
```

Against the actual NorBERTo submission from EP1, soft voting gains 15 correct
answers with exact McNemar p=0.303327; stacking gains eight with p=0.615381.
Neither establishes significant superiority. These comparisons remain
exploratory because test1 was already consulted elsewhere in the project.
No parameters were changed in response to these scores.

## Frozen fitting protocol

All four base models were fitted on 16,093 development rows, excluding the
3,999 holdout rows from gradient or classifier fitting. Qwen is the selected
adapter with identity
`a045c5726a85c466914c55f9cee9fb0d412af4d66387e14b590e53b4782beed5`.
The older Qwen adapter with a different split was excluded.

The holdout split is unchanged:

| Role | Rows | Similarity groups |
|---|---:|---:|
| Probability calibration | 2,027 | 1,627 |
| Soft weight selection / stacking meta-training | 1,972 | 1,654 |

Both partition files are byte-identical to the corresponding original
three-member files, including the stacking CV assignments. No calibration
rows enter the meta-model, and groups do not cross roles.

The SVM sigmoid and existing temperature recipes were retained. Qwen's
temperature was chosen only from the calibration subset, using three grouped
folds: native log loss 1.05293 versus temperature OOF log loss 1.04078, with
final temperature 1.515480709210816. Its class argmax remains unchanged.

Soft voting compared the fixed 35 quarter-simplex candidates by selection
macro-F1. Stacking used five grouped folds, fixed C candidates
`[0.01, 0.1, 1, 10]`, and the existing one-standard-error rule. The selected
stacking OOF macro-F1 is 47.30%; this participated in selection and is not
an independent performance estimate.

Qwen's checkpoint was selected using the full original holdout. These rows
were excluded from its weight updates but already influenced checkpoint
selection. Combined with historical model research, this inherited exposure
must be disclosed when interpreting calibration and meta-model validation.

## Inference, auditing and preserved artifacts

Fresh Qwen probabilities were generated in sequential batches of 16 for all
3,999 holdout rows and 900 test1 texts, without passing gold labels to the
model. This took 427.40 and 98.69 seconds, respectively. Every output retains
its source row ID and text checksum. No hard-label export was converted into
artificial probabilities, and the previously recovered validation export
was not used as a probability source.

Fresh holdout scoring again obtains 1,791/3,999 correct (44.79%). Its macro-F1
is 44.4371%; a few individual decisions differ from the original grouped
training evaluation, whose macro-F1 was 44.4429%. The four-member fit uses
the newly generated inference distributions consistently.

An independent standard-library ZIP/XML workbook reader verified all source
IDs, texts and labels. Independent arithmetic reconstructed all calibrated
test probabilities, all 35 weight-candidate scores, all soft sums and all
12-feature meta-model outputs. Maximum probability difference was
1.11e-16. Confusion matrices, accuracy and macro-F1 reproduce the saved
reports; original three-member artifacts remain loadable and reproduce their
447/900 and 442/900 controls.

Raw evidence is under `results/ensemble-qwen-verification-20261105/`.
Qwen scores are under `results/qwen-ensemble-scores-20261105/`, with the
selected test1 run also saved in `results/qwen-ep1-test1/`. Fit and prediction
runs use separate `ensemble-soft-qwen-*` and `ensemble-stacking-qwen-*`
directories. All original weights, models and historical result files remain
unchanged. No base model was retrained and no test2 submission was generated.

The available offline suite passed 171 tests in 30.15 seconds; two unrelated
optional OpenAI SDK modules were excluded in this container. Both documented
default CLI commands were also executed on CPU with saved scores and reproduced
the audited predictions byte for byte, without reading gold.
