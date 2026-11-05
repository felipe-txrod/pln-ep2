# Frozen inference models

This directory distributes the final inference artifacts without retraining.
Trained weights, TF-IDF vocabularies, tokenizers, and configuration files are
eligible for Git versioning. Each package includes a `manifest.json` mapping
its inference files to SHA-256 checksums.

| Directory | Versioned files | External dependency |
|---|---|---|
| [svm/](svm/README.md) | Trained TF-IDF + LinearSVC pipeline, `C=0.05`, in `model.joblib` | None |
| [rl-development/](rl-development/README.md) | Verified development-only logistic pipeline, `C=0.25`, trained on 16,093 rows | None for inference |
| [rl/](rl/README.md) | Historical full-corpus logistic pipeline; not the default | None |
| [norberto/](norberto/README.md) | LoRA adapter, classification head, tokenizer, configuration | Public NorBERTo base, pinned revision and checksums |
| [qwen/](qwen/README.md) | Selected EP1-split QLoRA adapter, trained head, tokenizer and checksummed provenance | Public Qwen3-1.7B base at the pinned revision |
| [gpt/](gpt/README.md) | Configuration and prompt snapshots | Remote service; weights unavailable locally |
| [ensemble-soft-qwen/](ensemble-soft-qwen/README.md) | Default four-candidate calibration and soft weights; selected Qwen weight is zero | Selected SVM, development RL, original NorBERTo and EP1-split Qwen |
| [ensemble-stacking-qwen/](ensemble-stacking-qwen/README.md) | Default meta-LR on twelve calibrated probabilities | The four selected model packages and frozen calibrators |
| [ensemble-soft/](ensemble-soft/README.md) | Preserved original three-model calibration, soft weights and holdout roles | SVM, development RL and original NorBERTo |
| [ensemble-stacking/](ensemble-stacking/README.md) | Preserved original three-model multinomial meta-LR and grouped CV provenance | The original nine calibrated probabilities |

Qwen now defaults to `models/qwen/adapter/`; its historical run directories
remain preserved. Qwen checkpoints, optimizer states and public base weights
stay outside the versioned package. Qwen participates in soft voting and
stacking; the separate majority presets retain NorBERTo, SVM and RL.

## Artifact formats

The `.joblib` files already contain learned coefficients, vocabularies, and IDF
values. The adapter `.safetensors` file contains LoRA parameters and the trained
classification head. No training, GGUF conversion, or adapter merging is needed.

The copies preserve the original artifact bytes. Loaders check hashes, classes,
and configuration before loading weights. SVM, logistic regression, and NorBERTo
use their selected bundles under `models/`. RL defaults to `rl-development/`
and fails if that development-only bundle has not yet been trained. It does
not fall back to the historical full-corpus `rl/` package.

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
python3 scripts/models/freeze.py --model qwen \
  --training-run results/qwen-ep1-training-20261105
```

The `--model rl` exporter preserves the historical full-corpus model in
`models/rl/`; it does not select or generate the development-only RL model.
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
