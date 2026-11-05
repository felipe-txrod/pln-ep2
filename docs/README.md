# Documentation

[Project README](../README.md)

Use **Guides** for commands, **Results** for measured quality, **Reference**
for assignment material, and **Archive** for superseded experiments. All
command examples assume the EP2 repository root unless stated otherwise.

## Guides

| Subject | Document |
|---|---|
| Current NorBERTo training recipe, full corpus, and frozen development split | [Training](guides/norberto-training.md) |
| NorBERTo prediction and separate CPU scoring on test1 | [Test1 workflow](guides/norberto-test1.md) |
| Why recipe 3 preserves clipping and AdamW gradient units | [Optimizer compatibility](guides/norberto-optimizer-units.md) |
| Qwen installation, ROCm compatibility, QLoRA, training, and inference | [Qwen fine-tuning](guides/qwen-finetuning.md) |
| Classifier selection, prediction contracts, and tie-breaking | [Ensemble](guides/ensemble.md) |
| Frozen SVM, logistic regression, NorBERTo, and GPT configuration | [Model artifacts](../models/README.md) |

Package-specific inference interfaces remain next to their source:
[NorBERTo](../src/norberto_classifier/README.md),
[SVM](../src/svm_classifier/README.md),
[logistic regression](../src/rl_classifier/README.md), and
[ensemble](../src/ensemble_classifier/README.md).

## Results

The **[results index](results/README.md)** separates test1 comparisons from
holdout validation and links each report to its technical aggregate. Raw runs
and individual predictions remain under `results/` at the project root;
public aggregates are under [`docs/results/data/`](results/data/).

## Reference

| Document | Purpose |
|---|---|
| [Assignment](reference/assignment.md) | Official task, evaluation criteria, and deliverables |
| [Report template](reference/report-template.md) | Required report structure |

## Archive

The **[archive index](archive/README.md)** groups earlier GPT/Codex studies and
NorBERTo recipes 1–2, regression investigations, and diagnostics. These records
retain their original results and limitations; their commands are historical
references rather than the default workflow.
