# EP2 — e-SIC clarity classification

Supervised text classification with Qwen3-1.7B, NorBERTo, SVM, logistic
regression, and a configurable ensemble with GPT arbitration. The input is
`resp_text`; output labels are `c1`, `c234`, and `c5`.

## Documentation

Start with the **[documentation index](docs/README.md)**. Operational guides,
measured results, assignment references, and historical experiments are separate.

| Task | Guide |
|---|---|
| Train NorBERTo on 80% or 100% of EP1 training data | [NorBERTo training](docs/guides/norberto-training.md) |
| Predict test1, then compare with gold labels | [NorBERTo test1 workflow](docs/guides/norberto-test1.md) |
| Set up, train, and run Qwen QLoRA | [Qwen fine-tuning](docs/guides/qwen-finetuning.md) |
| Select ensemble members and the tie-breaking model | [Ensemble](docs/guides/ensemble.md) |
| Restore frozen models and their pinned public bases | [Model artifacts](models/README.md) |
| Compare measured classifier results | [Results index](docs/results/README.md) |
| Consult earlier GPT/Codex and training investigations | [Experiment archive](docs/archive/README.md) |

## Repository layout

```text
configs/       Classifier and experiment configurations
src/           Classifier packages and shared utilities
scripts/       Training, inference, evaluation, and artifact commands
models/        Frozen model packages and base references
data/         Input workbooks
results/       Local runs and predictions; ignored by Git
docs/          Guides, results, reference material, and archive
```

## Base installation

Use Python 3.11 or later and run commands from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e .
```

GPU training and inference use the ROCm environment documented in the model
guides. On the RX 6650 XT, the validated container is `pln-rocm:6.4.1`, with
`HSA_OVERRIDE_GFX_VERSION=10.3.0`. The base installation above does not select
or install a GPU-specific PyTorch distribution.

## Classifier packages

| Package | Role |
|---|---|
| `qwen_classifier` | Qwen3-1.7B classification with QLoRA adapters |
| [norberto_classifier](src/norberto_classifier/README.md) | Frozen original EP1 model; separate scripts handle new adapters |
| [svm_classifier](src/svm_classifier/README.md) | Frozen TF-IDF and SVM pipeline, CPU inference |
| [rl_classifier](src/rl_classifier/README.md) | Frozen TF-IDF and logistic regression pipeline, CPU inference |
| [ensemble_classifier](src/ensemble_classifier/README.md) | Configurable members and arbitration over saved predictions |
| `gpt_classifier` | Remote classification, example retrieval, and shared evaluation |

Frozen packages live in `models/`; newly trained adapters remain in their
individual run directories. The [model guide](models/README.md) explains
formats, integrity checks, restoration, and public base revisions.

## Data and evaluation

The original training workbook has **20,092 responses**. NorBERTo's frozen
partition uses **16,093 development rows** and **3,999 holdout rows**; training
on 100% includes that former holdout. Each experiment records its own partition.

**Test1 has 900 rows** and serves final comparison with EP1. Predict texts
before comparing with gold labels. **Test2 has 897 rows** and is the submission
workbook. Do not select prompts, hyperparameters, or ensemble rules using test1
labels. Holdout and test1 scores describe different datasets.

The [assignment](docs/reference/assignment.md) and
[report template](docs/reference/report-template.md) define the academic
requirements. Recorded metrics and their provenance are in the
[results section](docs/results/README.md).

## Verification

```bash
.venv/bin/python -m pytest -q
```

Optional model dependencies and real GPU integration checks are described in
the corresponding guides. Earlier GPT/Codex commands, retrieval studies, and
verification logs are retained in the [archive](docs/archive/README.md).
