# Frozen inference models

This directory distributes the final inference artifacts without retraining.
Trained weights, TF-IDF vocabularies, tokenizers, and configuration files are
eligible for Git versioning. Each package includes a `manifest.json` mapping
its inference files to SHA-256 checksums.

| Directory | Versioned files | External dependency |
|---|---|---|
| [svm/](svm/README.md) | Trained TF-IDF + LinearSVC pipeline, `C=0.05`, in `model.joblib` | None |
| [rl/](rl/README.md) | Trained TF-IDF + logistic regression pipeline, `C=0.25`, in `model.joblib` | None |
| [norberto/](norberto/README.md) | LoRA adapter, classification head, tokenizer, configuration | Public NorBERTo base, pinned revision and checksums |
| [gpt/](gpt/README.md) | Configuration and prompt snapshots | Remote service; weights unavailable locally |

Qwen retains its own pipeline and artifact directories. The packaging scripts
exclude its checkpoints, configuration, and cache.

## Artifact formats

The `.joblib` files already contain learned coefficients, vocabularies, and IDF
values. The adapter `.safetensors` file contains LoRA parameters and the trained
classification head. No training, GGUF conversion, or adapter merging is needed.

The copies preserve the original artifact bytes. Loaders check hashes, classes,
and configuration before loading weights. SVM, logistic regression, and NorBERTo
prefer `models/`, while supporting their previous paths and environment overrides.

## Restore the public base

The NorBERTo base weights remain outside Git. `norberto/base_reference.json`
pins the public repository, commit, and SHA-256 of every required base file.
After cloning the repository, restore missing files without loading the model:

```bash
python3 scripts/models/prepare-base.py
```

The command uses only the Python standard library and verifies downloaded files
against the frozen inference contract. Verify an existing copy without networking:

```bash
python3 scripts/models/prepare-base.py --verify-only
```

`models/norberto/base_model/model.safetensors` is available locally after
preparation and remains ignored by Git. Restoration requires access to the pinned
public revision; the adapter alone cannot replace the base weights.

## Export and versioning

The exporter copies existing artifacts and checks their hashes. It does not
import model libraries, deserialize weights, train, or perform inference:

```bash
python3 scripts/models/freeze.py --model svm
python3 scripts/models/freeze.py --model rl
python3 scripts/models/freeze.py --model norberto
python3 scripts/models/freeze.py --model gpt
```

Exports use the preserved local EP1 artifacts. A fresh clone receives the
versioned inference files and does not need to export them again. Identical
files are accepted; different existing files are never overwritten. Use a new
`--output` directory for a different package. For NorBERTo, `--with-local-base`
also copies the local base weights while keeping them ignored by Git.

The trained artifacts fit ordinary Git; Git LFS is unnecessary for these files.
GitHub blocks individual files above 100 MiB; see the
[GitHub file size limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
Large public bases remain external.

Use the runtime versions recorded in the contracts and dependency files.
Keep caches, intermediate checkpoints, optimizer states, credentials, and raw
execution logs outside the versioned model packages.
