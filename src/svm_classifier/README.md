# Frozen EP1 SVM classifier

CPU inference with the trained `LinearSVC`, without training commands or
automatic fitting. Loading occurs only through `load()` or `predict()`.
Importing the package does not import scikit-learn, NumPy, PyTorch, or GPU libraries.

## Preserved configuration

- `LinearSVC`: `C=0.05`, `dual="auto"`, `max_iter=10000`, `tol=1e-4`, seed 42.
- Word TF-IDF: n-grams 1–2, `min_df=2`, `max_df=0.98`, up to 60,000 features.
- Character TF-IDF: `char_wb`, n-grams 3–5, `min_df=3`, up to 50,000 features.
- Sublinear term frequency, L2 normalization, and FP32 matrices.
- Preprocessing: HTML unescape, whitespace normalization, lowercase; accents,
  punctuation, and negation are preserved.
- Classes: `c1`, `c234`, `c5`; two CPU threads by default.

The original artifact is `EP1/artifacts/svm-fixed-holdout-20260923.joblib`,
trained on 16,093 development responses. The versioned independent copy is
[`models/svm/model.joblib`](../../models/svm/README.md), accompanied by
`config.json`, `metadata.json`, and manifests. The source
[frozen manifest](frozen_manifest.json) pins hashes and parameters.
The previous `artifacts/svm-ep1/` path remains a fallback when no published
package exists.

## Environment and usage

`requirements-lexical.txt` pins the EP1 dependencies. Use a separate Python 3.12
environment: the loader requires the saved NumPy, SciPy, scikit-learn, and joblib
versions. The general EP2 lock uses a different joblib version and does not
replace these requirements. GPU libraries and pandas are unnecessary.

From the EP2 root with these dependencies installed:

```bash
PYTHONPATH=src python -m svm_classifier predict \
  --input data/test2.xlsx --output-dir results/svm-development-test2
```

```python
from svm_classifier import SVMClassifier

classifier = SVMClassifier()
predictions = classifier.predict(["Example public response."])
label = predictions[0].label
scores = predictions[0].decision_scores
```

The CLI writes `predictions.jsonl` and `run.json`, preserving row IDs and text
hashes without changing the workbook or reading gold labels. `info` inspects the
manifest and file presence without loading the model. `--artifact` and
`SVM_ARTIFACT_PATH` select another copy with the same hash. `SVM_BUNDLE_DIR`
selects the directory containing metadata, configuration, and the default
`model.joblib`.

SVM provides labels and decision margins, not probabilities. Do not apply
softmax to margins to simulate calibrated probabilities. Ensemble participation
uses labels, consensus or strict majority, and the configured arbiter.

## Serialization and validation

The loader verifies the artifact hash before deserialization and remaps only
the historical `clarity.data.normalize_text` reference to its equivalent local
string preprocessing function. The copy requires neither EP1 imports nor its
directory. The adaptation uses the
[joblib 1.5.2 array loader](https://github.com/joblib/joblib/blob/1.5.2/joblib/numpy_pickle.py).
Vocabularies, IDF, and coefficients are loaded from the trained artifact.

The package previously completed CPU inference on all 900 test1 responses,
with **49.44% accuracy (445/900)**. Row IDs, texts, parameters, and hashes were
verified; see the [full evaluation](../../docs/results/lexical-test1.md).
