# Development-only logistic regression

`RLClassifier` is the ensemble member `rl`. Its default bundle is now
`models/rl-development/`, trained on the fixed 16,093 EP1 development rows
while reserving all 3,999 holdout rows. The local bundle has completed training and verification: 38 iterations,
25.08 seconds, exact training-row coverage, and zero holdout group overlap.
See the [trained model](../../models/rl-development/README.md). The earlier
lexical comparison saved only its winning SVM pipeline; the preserved
`models/rl/` artifact was fitted on all 20,092 rows and remains historical.

## Reproduce the selected model

From the EP2 root, using the existing offline CPU image and sibling EP1:

```bash
bash scripts/rl/train.sh --dry-run
bash scripts/rl/train.sh
```

These commands are for a new output directory; the completed local bundle must
not be overwritten. The first command validates dataset/split hashes, row counts, and disjoint
similarity components without fitting or writing artifacts. The second fits
only development, using EP1's original word/character TF-IDF builder and fixed
`C=0.25`, L-BFGS, L2, 1,500 iterations, seed 42, and two CPU threads. There is
no parameter search, holdout scoring, official-test access, or full-corpus fit.
Nonconvergence fails. Training writes safe provenance, exact training row
identities, checksummed joblib/metadata/configuration, and a manifest after
reloading and validating the exported pipeline.

`RL_EP1_ROOT` and `RL_IMAGE` override the sibling checkout and local image.
An existing output directory is rejected. `--output-dir` selects another new
bundle; an alternate bundle must also be selected explicitly for inference
and in the ensemble manifest. Training is not started by inference.

## Inference

Use the pinned dependencies in `requirements-lexical.txt`:

```bash
PYTHONPATH=src python3 -m rl_classifier predict \
  --input data/test2.xlsx --output-dir results/rl-development-test2
```

For test1, use `data/test1.xlsx` and `results/rl-development-test1`. Prediction
reads texts only and returns labels, probabilities, and named class scores.
The ensemble checks the run's artifact hash against
`models/rl-development/frozen_manifest.json`, so earlier full-corpus
predictions cannot be reused as development predictions.

A missing development bundle produces an actionable error and never falls
back to the historical full-corpus model. `RL_BUNDLE_DIR` selects an explicit
alternate bundle; `--artifact`/`RL_ARTIFACT_PATH` select another copy under
that bundle's integrity contract. Loading and prediction remain CPU-only.

The historical full-corpus weights in `models/rl/` are preserved for
traceability. Their earlier 49.33% test1 accuracy applies only to that model.
The development-only model has now been evaluated separately: 1,803/3,999
correct on holdout (45.09% accuracy, 44.78% macro-F1) and 447/900 on labeled
test1 (49.67% accuracy, 49.32% macro-F1). All source rows and metrics were
verified independently, with no refitting or parameter changes. See the
[evaluation record](../../docs/results/data/rl-development-evaluation.json).
