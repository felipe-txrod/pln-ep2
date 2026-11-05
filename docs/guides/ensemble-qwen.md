# Soft voting and stacking with the selected Qwen

Both ensemble families now support the frozen SVM, development-only RL,
original EP1 NorBERTo, and the selected `models/qwen/adapter/`. They use the
same 16,093-row supervised training partition and 3,999-row holdout.
The existing majority presets remain separate and retain three voters.

## Selected artifacts and data roles

| Method | Default artifact | Inputs |
|---|---|---|
| Calibrated soft voting | `models/ensemble-soft-qwen/model.json` | Four distributions with explicit weights |
| Probability stacking | `models/ensemble-stacking-qwen/model.json` | Twelve calibrated probabilities |

The holdout roles are byte-identical to the original three-model ensembles:
2,027 rows calibrate probabilities, and 1,972 select the soft weights or train
the stacking meta-model. Rows and similarity components are disjoint.

SVM uses sigmoid calibration. RL, NorBERTo, and Qwen use scalar temperature
scaling only when a three-fold grouped check improves log loss on calibration
rows. The weight grid contains the 35 distinct four-member quarter-simplex
points, including uniform 25% weights. Macro-F1 chooses the mixture, with
accuracy and distance to uniform weights breaking ties. Zero weights are
allowed and must be reported. The selected Qwen weight is zero.

Stacking uses all 12 probabilities, including models with zero soft weight.
Five grouped folds on the 1,972 meta rows select from `C=[0.01, 0.1, 1, 10]`
using the one-standard-error rule; the chosen C is 0.1. The exported JSON
coefficients reproduce logistic regression probabilities without pickle or
fitting during prediction.

Qwen received no gradient updates on the holdout, but the holdout selected
its checkpoint. Together with earlier research exposure, this prevents
treating these inputs as untouched final validation. Test1 labels are not
used for calibration, weight selection, or meta-model fitting.

## Prediction

With the recorded test1 score files available, combining the saved member
outputs is CPU-only. Use new directories:

```bash
PYTHONPATH=src python3 -m ensemble_classifier predict-soft \
  --input data/test1.xlsx --output-dir results/soft-qwen-test1-new

PYTHONPATH=src python3 -m ensemble_classifier predict-stacking \
  --input data/test1.xlsx --soft-run results/soft-qwen-test1-new \
  --output-dir results/stacking-qwen-test1-new
```

For fresh data or a clone without the ignored raw score files, use GPU
inference and the selected public base caches. Neural models execute
sequentially, releasing GPU allocations between models:

```bash
PYTHONPATH=src python3 -m ensemble_classifier predict-soft \
  --input data/test2.xlsx --output-dir results/soft-qwen-test2-new \
  --device gpu --live-norberto --live-qwen

PYTHONPATH=src python3 -m ensemble_classifier predict-stacking \
  --input data/test2.xlsx --soft-run results/soft-qwen-test2-new \
  --output-dir results/stacking-qwen-test2-new
```

Run live prediction in the existing ROCm runtime with its bitsandbytes cache
and GPU device mappings, as described in the [Qwen guide](qwen-finetuning.md).
These commands do not train the base models, refit calibrators, or read gold.
The test2 examples were not executed in this integration.

## Reproducing the fit

The default configurations identify four members and pin the Qwen score file
and run checksums. Reproducing a fit requires those local recorded scores:

```bash
PYTHONPATH=src python3 -m ensemble_classifier fit-soft \
  --config configs/ensemble-soft.json --output-dir results/soft-qwen-fit-new
```

To fit another stacking artifact, copy `configs/ensemble-stacking.json` and
point its `soft_model`, `soft_partition`, and `holdout_scores` fields at the
intended immutable fit. Do not mix calibrators, partitions or scores from
different model packages.

The three-member models remain under `models/ensemble-soft/` and
`models/ensemble-stacking/`. Use `--model` to select them explicitly.
Their original configurations are preserved as
`configs/ensemble-soft-legacy.json` and `configs/ensemble-stacking-legacy.json`.
Only the two audited original artifact checksums receive legacy compatibility;
other artifacts still require their exact implementation identity.

The [integration report](../results/ensemble-qwen.md) gives the measured
test1 comparison and independent verification.
