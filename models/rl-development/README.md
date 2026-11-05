# Development-only logistic regression

This is the selected RL bundle for EP2. Training completed on the fixed
16,093 development rows while excluding all 3,999 holdout rows. It uses the
original EP1 word/character TF-IDF pipeline and fixed C=0.25, L2/L-BFGS,
seed 42, and two CPU threads, without parameter search.

The fit converged in 38 iterations and completed in 25.08 seconds. The
exported artifact was reloaded and validated. A subsequent independent check
verified all 16,093 training row IDs, text hashes, labels, source/split hashes,
and zero overlap between 12,956 training and 3,281 holdout similarity groups.
Default RL loading and inference on three synthetic texts passed.

Artifact SHA-256:

```text
05d69e147a10b7c49ebe3893f247e0f23e8d61005aaf4fd7b7eb45845d49a20f
```

`manifest.json` records inference-file checksums; `frozen_manifest.json`
defines the loader and ensemble identity contract. Training metadata and row
identities remain available for provenance. Separate evaluation measured
1,803/3,999 correct on holdout (45.09% accuracy, 44.78% macro-F1) and 447/900
on test1 (49.67% accuracy, 49.32% macro-F1). All 4,899 source rows and metrics
were independently verified, with unchanged model hashes. Test2 remains
unevaluated. Earlier full-corpus RL scores do not apply to this model.

The ensemble selects this artifact when `rl` is included as a member or local
arbitrator. Its existing default membership remains Qwen + NorBERTo. Prediction
runs must still be generated for the desired input workbook; training itself
does not generate test predictions or submission files.

See the [RL workflow](../../src/rl_classifier/README.md) and
[verification record](../../docs/results/data/rl-development-verification.json).
