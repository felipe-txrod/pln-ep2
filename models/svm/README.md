# Frozen SVM

`model.joblib` contains the trained word/character TF-IDF + LinearSVC pipeline
with `C=0.05`, including vocabularies, IDF values, coefficients, and intercepts.
The file is approximately 3.59 MB and can be versioned in ordinary Git.

Original artifact SHA-256:

```text
9a30abb664d5023d70f444641c4095cf800acba0c98fd8658087e7adad78ee12
```

`metadata.json`, `config.json`, and `frozen_manifest.json` preserve EP1
provenance and parameters. `manifest.json` identifies the distributed copies.
Inference runs on CPU without retraining or artificial probabilities.

See the [usage guide](../../src/svm_classifier/README.md) and
[test1 evaluation](../../docs/avaliacao-lexicais-test1.md).
