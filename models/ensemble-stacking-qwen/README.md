# Four-member probability stacking

This is the selected stacking package for frozen SVM, development-only RL,
original EP1 NorBERTo, and EP1-split Qwen. It references the calibrators and
group-disjoint partition in `models/ensemble-soft-qwen/`.

The multinomial L2 logistic meta-model receives 12 calibrated probabilities,
ordered `svm, rl, norberto, qwen` by `c1, c234, c5`. It is fitted only on the
1,972 meta-training rows; 2,027 calibration rows are excluded. Five grouped
folds compare the fixed grid `C=[0.01, 0.1, 1, 10]`, with the one-standard-error
rule selecting **C=0.1**. Qwen has nonzero learned coefficients in the meta-model.

Test1 scored **440/900 (48.89%)**, macro-F1 **48.80%**. The original
three-member stacking scored 442/900 (49.11%). Adding Qwen corrected 25 of
its errors and harmed 27 correct answers, a net loss of two correct answers.
No model, calibration parameter or regularization setting was adjusted using
this test result.

CLI `predict-stacking` defaults here. The previous package remains available
under `models/ensemble-stacking/`; its weights and provenance are unchanged.
The JSON model is checked against its manifest, shared soft model, partition,
class order, feature order and numerical probability contract before use.

See the [results](../../docs/results/ensemble-qwen.md) and
[commands](../../docs/guides/ensemble-qwen.md). Qwen's checkpoint selection used
the original holdout, so these are historically exposed development inputs,
not an independent validation sample for the full modeling process.
