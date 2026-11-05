# Frozen logistic regression

`model.joblib` contains the trained word/character TF-IDF + logistic regression
pipeline with `C=0.25`, including vocabularies, IDF values, coefficients, and
intercepts. The file is approximately 3.61 MB and can be versioned in ordinary
Git. The final training corpus contained 20,092 responses.

Original artifact SHA-256:

```text
f7af61142f43e1c8a37f0cd5a8191dd6d83ba1222cd8ecf12281f9929ac827c9
```

Configuration, metadata, and manifests accompany the weights. CPU inference
returns labels and probabilities without fitting the classifier again.

See the [usage guide](../../src/rl_classifier/README.md) and
[test1 evaluation](../../docs/results/lexical-test1.md).
