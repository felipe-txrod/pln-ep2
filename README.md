# EP2 — e-SIC clarity classification

Supervised text classification with Qwen3-1.7B, NorBERTo, SVM, logistic
regression, and a configurable ensemble with GPT arbitration. The input is
`resp_text`; output labels are `c1`, `c234`, and `c5`.

The selected NorBERTo is the **unchanged EP1 delivery adapter** in
`models/norberto/`: LoRA rank 4/alpha 8, two epochs on 16,093 development rows.
It achieved 48.00% on test1 and 45.31% on the EP1 holdout. Experimental loss
corrections, refits, and the frozen-encoder logistic variant have been retired.

The active ensemble uses **NorBERTo + SVM + RL**, equal majority votes, and
RL to break three-way ties. Qwen is currently excluded from the default
membership. All three selected models reserve the same 3,999-row EP1 holdout.

## Documentation

Start with the **[documentation index](docs/README.md)**. Operational guides,
measured results, assignment references, and historical experiments are separate.

| Task | Guide |
|---|---|
| Reuse the unchanged EP1 NorBERTo adapter | [Original model](src/norberto_classifier/README.md) |
| Predict test1, then compare with gold labels | [NorBERTo test1 workflow](docs/guides/norberto-test1.md) |
| Set up, train, and run Qwen QLoRA | [Qwen fine-tuning](docs/guides/qwen-finetuning.md) |
| Train development-only lexical logistic regression with holdout reserved | [RL workflow](src/rl_classifier/README.md) |
| Select ensemble members and the tie-breaking model | [Ensemble](docs/guides/ensemble.md) |
| Browse runnable scripts by model and purpose | [Script index](scripts/README.md) |
| Restore frozen models and their pinned public bases | [Model artifacts](models/README.md) |
| Compare measured classifier results | [Results index](docs/results/README.md) |
| Reproduce NorBERTo + SVM + RL majority voting and its test1 comparison | [Majority ensemble](docs/results/ensemble-majority.md) |
| Consult earlier GPT/Codex and training investigations | [Experiment archive](docs/archive/README.md) |

## Repository layout

```text
configs/       Classifier and experiment configurations
src/           Classifier packages and shared utilities
scripts/       Commands grouped by model, artifacts, diagnostics, and experiments
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
| [norberto_classifier](src/norberto_classifier/README.md) | Unchanged original EP1 delivery adapter; inference only |
| [svm_classifier](src/svm_classifier/README.md) | Frozen TF-IDF and SVM pipeline, CPU inference |
| [rl_classifier](src/rl_classifier/README.md) | Frozen TF-IDF and logistic regression pipeline, CPU inference |
| [ensemble_classifier](src/ensemble_classifier/README.md) | Configurable members and arbitration over saved predictions |
| `gpt_classifier` | Remote classification, example retrieval, and shared evaluation |

Frozen packages live in `models/`; historical experimental adapters remain in
their original run directories. The [model guide](models/README.md) explains
formats, integrity checks, restoration, and public base revisions.

## Data and evaluation

The canonical training workbook, `data/train.xlsx`, has **20,092 responses**. NorBERTo's frozen
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
