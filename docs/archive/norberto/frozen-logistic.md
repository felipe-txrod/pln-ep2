# Frozen NorBERTo with logistic regression

Historical workflow retired on 2026-11-05. The unchanged original EP1 delivery
adapter is the selected model. Commands below describe the former experiment;
its active training/evaluation implementation has been removed. Local source
snapshots are preserved in `results/norberto-rollback-20261105/sources-before-rollback.tar.gz`.

[Documentation index](../../README.md)

This candidate uses the pinned public `Itau-Unibanco/NorBERTo-base` revision
`db73446f89c96044863ea05a39f680524b84bccb` as an immutable feature extractor.
No LoRA adapter or classification head is loaded. Extraction uses evaluation
mode, inference-only execution, FP32, and zero trainable encoder parameters.
Only an L2-regularized multinomial logistic regression is fitted.

## Representation

Input cleaning preserves case and uses the existing EP1-compatible head/tail
tokenization with a 512-token context. Attention-mask-weighted mean pooling of
the last hidden state includes attended special tokens and excludes padding.
Each 768-dimensional vector is L2-normalized independently. There is no
corpus-fitted scaler or learned pooling, so the fixed vectors may be cached
and reused across validation folds without fitting on held-out observations.

The encoder verifies the public base/tokenizer SHA-256 values. Its loaded state
hash must remain identical before and after extraction. The SQLite cache stores
per-vector checksums and keys covering text, encoder identity, precision,
runtime versions, device selection, batch size, and extraction version.
Completed batches persist immediately; retry a failed extraction with a new
output directory and the same cache to reuse them.

## Exact EP1 NorBERTo validation partitions

The implementation reproduces the original NorBERTo-512 evaluation protocol,
not the different nested grid-search protocol used by some lexical studies:

- The unchanged EP1 workbook and frozen similarity-component split are hash-pinned.
- Only 16,093 development rows enter extraction, fitting, and OOF scoring.
  The 3,999 holdout rows are excluded before tokenization.
- Three stratified grouped outer folds use seed 42. Within each outer training
  partition, the first split of five stratified grouped folds uses seed
  `42 + fold`. Every reconstructed row ID is checked against the saved EP1
  `split-1.json` through `split-3.json`.
- The prespecified `C` grid is `1, 10, 100`, from EP1's frozen semantic-logistic
  experiments. Inner accuracy selects `C`; ties retain grid order. The selected
  classifier is evaluated on the outer fold without an outer-training refit,
  preserving the number of training examples used by EP1's encoder.
- A final classifier uses all development rows and the median of the three
  inner-selected `C` values. Outer scores never select final parameters.

| Fold | Inner training | Inner validation | Outer validation |
|---|---:|---:|---:|
| 1 | 8,572 | 2,199 | 5,322 |
| 2 | 8,769 | 2,176 | 5,148 |
| 3 | 8,151 | 2,319 | 5,623 |

The solver matches the EP1 logistic family: L-BFGS, L2, maximum 1,500 iterations,
tolerance 0.0001, no class weighting. Nonconvergence raises an error rather
than exporting a successful classifier. Configuration lives in
``configs/norberto-frozen-logistic.json``.

## Commands

Run from the EP2 repository root. The launcher uses the existing offline
`pln-rocm:6.4.1` image and mounts sibling EP1 read-only. `NORBERTO_IMAGE` and
`NORBERTO_EP1_ROOT` override those locations. Extraction/inference expose the
GPU; `--device gpu` explicitly selects it. Planning, CV, and final fitting
run on CPU without GPU device exposure.

```bash
bash scripts/norberto/frozen-logistic.sh plan

bash scripts/norberto/frozen-logistic.sh encode --device gpu \
  --output-dir results/norberto-frozen-logistic-new/embeddings

bash scripts/norberto/frozen-logistic.sh cross-validate \
  --features-dir results/norberto-frozen-logistic-new/embeddings \
  --output-dir results/norberto-frozen-logistic-new/cv

bash scripts/norberto/frozen-logistic.sh fit \
  --features-dir results/norberto-frozen-logistic-new/embeddings \
  --cv-dir results/norberto-frozen-logistic-new/cv \
  --output-dir results/norberto-frozen-logistic-new/model
```

Every output directory must be new. The shared cache defaults to
`.cache/norberto-logistic/embeddings.sqlite3`. `--base-dir`, `--cache`, and
`--batch-size` select other explicit locations/settings. Native Python uses
`PYTHONPATH=src python3 -m norberto_logistic`; editable installation also exposes
`norberto-logistic-ep2`. Use the pinned project dependencies and a suitable
PyTorch runtime, as in the existing NorBERTo environment.

The completed local candidate is in
`results/norberto-frozen-logistic-ep1-20261105/model`. Predict with an explicitly
chosen model, for example:

```bash
bash scripts/norberto/frozen-logistic.sh predict --device gpu \
  --model-dir results/norberto-frozen-logistic-ep1-20261105/model \
  --input data/test1.xlsx \
  --output-dir results/norberto-frozen-logistic-test1-new
```

Prediction reads only `resp_text` and writes original Excel row IDs, text hashes,
labels, and probabilities to `predictions.jsonl`. It does not read gold labels,
train a classifier, or perform official test scoring. Outside development
extraction/CV, initial prediction verification used three synthetic responses.
The subsequent reserved-holdout evaluation is described below. This candidate
has not been scored on either official test.

## Holdout evaluation

In the same pinned Python/ROCm environment, evaluate the saved development
classifier without refitting:

```bash
PYTHONPATH=src python3 scripts/norberto/evaluate-frozen-holdout.py \
  --training-run results/norberto-frozen-logistic-ep1-20261105 \
  --ep1-root ../EP1 --device gpu \
  --output-dir results/norberto-frozen-logistic-holdout-new \
  --recipe3-predictions results/norberto-recipe3-holdout-20261105/holdout/predictions.jsonl
```

The recipe-3 comparison is optional. The script requires a new output
directory and validates the pinned workbook/split, development feature row
identities, model/CV hashes, convergence, inner-only median C selection,
and row/group separation. It saves all 3,999 text-only predictions before
constructing labeled records, then scores and aligns saved historical LoRA,
logistic, and SVM controls. It never calls classifier fitting. Features,
predictions, metrics, encoder statistics, unchanged model hashes, and paired
similarity-group bootstrap comparisons are recorded separately from training.

The completed evaluation is in
`results/norberto-frozen-logistic-holdout-20261105/holdout/`: 1,694/3,999 correct,
42.36% accuracy, 42.08% macro-F1. Independent source parsing and direct NumPy
reconstruction verified every probability and metric. Original LoRA reached
45.31% on these same rows. No parameter was changed after seeing holdout
scores; the model has not been promoted.

## Artifacts and measured quality

Extraction exports a checked `features.npy`, text/row identities, and an encoder
manifest. CV exports three safe linear-head archives, fold metrics/inner trials,
16,093 OOF predictions, source hashes, and paired historical comparisons.
Final fitting exports `classifier.npz` and `manifest.json`. The NPZ contains
only coefficients, intercepts, and label order; loading uses `allow_pickle=False`.
Inference verifies both classifier and encoder identities before prediction.

The [results index](../../results/README.md#frozen-norberto-logistic-ep1-cross-validation)
records development OOF and the separately requested holdout evaluation, with
independent verification for both. The frozen encoder removes encoder
optimization from this pipeline; convexity does not guarantee better accuracy
or monotonic improvement with more data. This candidate remains separate from
the published EP1 model and ensemble selection.
