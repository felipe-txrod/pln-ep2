# Four-member calibrated soft voting

This is the selected soft-voting package for SVM, development-only RL,
original EP1 NorBERTo, and the selected EP1-split Qwen. All base weights remain
frozen. Qwen uses `models/qwen/adapter/`; the historical Qwen adapter is excluded.

The original 3,999-row holdout is split into the same group-disjoint roles as
the three-member method: 2,027 calibration rows and 1,972 weight-selection rows.
SVM uses sigmoid calibration; RL, NorBERTo, and Qwen use scalar temperatures
selected by three-fold grouped checks on calibration rows only. Qwen's
temperature is 1.515480709210816.

Thirty-five distinct quarter-simplex candidates were compared by macro-F1.
The selected weights are **SVM 0.75, RL 0, NorBERTo 0.25, Qwen 0**. Qwen was
scored, calibrated, and included in the search, but does not contribute to
the selected weighted sum. Its zero weight is an outcome of holdout selection.

Test1 scored **447/900 (49.67%)**, macro-F1 **49.58%**. All 900 labels equal
the preserved three-member soft-voting result. No parameter was updated using
test1 gold labels. The frozen model, calibration parameters, membership and
code identity are in `model.json`; `frozen_manifest.json` pins its checksum.

CLI `predict-soft` defaults to this package. The original three-member package
remains in `models/ensemble-soft/`. See the [integration report](../../docs/results/ensemble-qwen.md)
and [usage guide](../../docs/guides/ensemble-qwen.md).

The holdout has historical exposure and selected the Qwen checkpoint. It is
development validation, not an untouched final test. This task did not produce
a test2 submission or establish the official test2-before-test1 order.
