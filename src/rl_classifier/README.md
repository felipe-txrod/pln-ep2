# Frozen EP1 logistic regression classifier

`RLClassifier` exposes the trained logistic regression model; `rl` is its
ensemble member ID. There is no retraining, parameter search, vocabulary
reconstruction, or automatic calibration.

## Configuration and artifact

- `LogisticRegression`: `C=0.25`, `solver="lbfgs"`, `max_iter=1500`, seed 42.
- Word TF-IDF: n-grams 1–2, `min_df=2`, `max_df=0.98`, up to 60,000 features.
- Character TF-IDF: `char_wb`, n-grams 3–5, `min_df=3`, up to 50,000 features.
- Sublinear frequency, L2 normalization, FP32 matrices, and text preprocessing
  equivalent to the [SVM pipeline](../svm_classifier/README.md).
- Final training: all 20,092 original responses; classes `c1/c234/c5`.

The original artifact is `EP1/artifacts/final-logistic-20260928.joblib`.
The versioned independent copy is [`models/rl/model.joblib`](../../models/rl/README.md),
with configuration, metadata, and manifests. SHA-256 matches the EP1 audit:

```text
f7af61142f43e1c8a37f0cd5a8191dd6d83ba1222cd8ecf12281f9929ac827c9
```

The source [frozen manifest](frozen_manifest.json) pins parameters, versions,
and hashes. The previous `artifacts/rl-ep1/` path remains a fallback when no
published package exists. A missing or different artifact raises an error;
it never starts training.

## Usage

Use `requirements-lexical.txt` in a separate environment, as described in the
[SVM guide](../svm_classifier/README.md).

```bash
PYTHONPATH=src python -m rl_classifier predict \
  --input data/test2.xlsx --output-dir results/rl-test2
```

```python
from rl_classifier import RLClassifier

classifier = RLClassifier()
predictions = classifier.predict(["Example public response."])
label = predictions[0].label
probabilities = predictions[0].probabilities
```

`predict` returns labels, probabilities, and named class scores. The CLI writes
`predictions.jsonl` and `run.json` without modifying its input. `info` does not
load the model. Alternate locations use `--artifact`, `RL_ARTIFACT_PATH`, or
`RL_BUNDLE_DIR`, under the same integrity rules as SVM.

Loading and inference are lazy and CPU-only. Previous inference on all 900
test1 responses achieved **49.33% accuracy (444/900)**, with row IDs, texts,
parameters, and hashes verified. See the
[full evaluation](../../docs/results/lexical-test1.md).
