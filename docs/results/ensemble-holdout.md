# Local ensemble: frozen holdout isolation and accuracy

[Results index](README.md) · [Audit aggregate](data/ensemble-holdout-audit.json)

The selected NorBERTo, SVM, and RL models use the same 16,093 development
rows. The 3,999 frozen holdout rows were excluded from fitting. Rechecking
the source split found **zero shared row IDs, zero shared similarity groups,
and zero normalized holdout texts present in development**. Qwen is excluded
from this evaluation and from the active default ensemble.

## Recomputed holdout results

All predictions cover the same 3,999 original workbook row IDs, without
filtering or reordering. Scores were recalculated from saved predictions;
no model was trained or executed again.

| Configuration | Correct / 3,999 | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Original EP1 NorBERTo | 1,812 | 45.31% | 45.31% |
| NorBERTo + SVM + RL, RL tiebreaker | 1,803 | 45.09% | 44.78% |
| Development-only RL | 1,803 | 45.09% | 44.78% |
| NorBERTo + SVM + RL, SVM tiebreaker | 1,802 | 45.06% | 44.76% |
| NorBERTo + SVM + RL, NorBERTo tiebreaker | 1,799 | 44.99% | 44.68% |
| Development-only SVM | 1,792 | 44.81% | 44.47% |

The best three-member majority again uses RL for ties, with 28 three-way
ties. It matches RL's accuracy but has different predictions and slightly
different macro-F1. NorBERTo alone leads this holdout by nine correct answers.
The [test1 comparison](ensemble-majority.md) has a different individual leader:
RL scores 447/900, while the same majority configuration scores 443/900.

## Evidence that the holdout was excluded

The unchanged training corpus has 20,092 rows, SHA-256
`0e9219233664ac675fc47982bbc822acd3bb15d67fb7bfc26b67bec047e46818`.
The common frozen split, from
`EP1/results/research/dataset-hypotheses-20260923/splits.csv`, has SHA-256
`0f78e7a809c7b7584db05b33cf03a73ff586d5de02ba824c16562d5070d41494`.
Development contains 12,956 similarity groups; holdout contains 3,281.

| Model | Checked training evidence | Holdout rows in fitting |
|---|---|---:|
| NorBERTo | All saved training row/group IDs, their training-manifest checksum, original adapter checksum, and original holdout evaluation identity | 0 |
| SVM | Original model checksum, unchanged exported metadata contract, pinned split, and development-only training path | 0 |
| RL | All saved training row IDs, source text hashes and labels, ordered identity checksum, model checksum, and training-script identity | 0 |

NorBERTo's saved `training-rows.csv` exactly matches every development row
and group in order. Its original adapter SHA-256 is
`d693f6e2995d117ad04a5692817527764fcccee6bfb2dde99120baf9fdd2d385`.
The original holdout report names this same adapter and training manifest.

RL's `models/rl-development/training-rows.jsonl` exactly matches all 16,093
development IDs, labels, and source text hashes. The ordered identity matches
its recorded checksum. Its selected weights SHA-256 is
`05d69e147a10b7c49ebe3893f247e0f23e8d61005aaf4fd7b7eb45845d49a20f`.
The evaluated holdout run names these same weights.

SVM's selected weights SHA-256 is
`9a30abb664d5023d70f444641c4095cf800acba0c98fd8658087e7adad78ee12`.
It matches the preserved EP1 artifact. Its metadata records the same frozen
split and 16,093 development rows. The preserved training pipeline passes
only `x[development]`, `y[development]`, and `groups[development]` to the
search/refit routine; holdout is passed only to prediction and scoring.
SVM did not save a separate training-row list, so its membership evidence
comes from this pinned split, artifact provenance, and recorded fit protocol.
The metadata's `dataset.rows=20092` describes the entire source workbook;
`fixed_validation.development_rows=16093` describes the fitting partition.

## Verification and limitations

The audit independently parsed the workbook using standard-library ZIP/XML,
including the existing numeric-text coercion on Excel row 5028. It checked
training lists, source identities, labels, groups, selected artifacts and
historical prediction provenance. The production majority function received
only text records and saved model votes. Every candidate vote was checked
again with an independent strict-majority count, and metrics were computed
directly from confusion matrices. Fifteen configurations produced 59,985
verified candidate rows. Historical single-model metrics were reproduced.

Raw evidence is under `results/ensemble-holdout-audit-20261105/`, containing
`audit.py`, `workbook_reader.py`, `isolation.json`, `summary.json`, and each
candidate's predictions and metrics. The public aggregate preserves hashes
of the original prediction files and training evidence.

Exclusion from supervised fitting does not make this an untouched final test:
the holdout was already inspected during earlier model research. These are
fixed-validation results. No models, adapters, caches, or historical results
were modified. No new test1/test2 inference or submission was generated.

The user explicitly excluded Qwen. Both regular ensemble configurations now
select `norberto svm rl`, strategy `majority`, and tiebreaker `rl`. The separate
majority presets retain the same settings. Qwen remains available only through
an explicit future member override; its saved artifacts are preserved.
