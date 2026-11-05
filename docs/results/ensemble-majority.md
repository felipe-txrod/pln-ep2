# NorBERTo, SVM, and RL majority comparison

[Results index](README.md) · [Verified aggregate](data/ensemble-majority-test1.json)

The strongest three-member majority configuration on the 900 labeled test1
rows was **NorBERTo + SVM + RL, with RL breaking three-way ties**: 443 correct,
49.22% accuracy, and 48.86% macro-F1. RL alone scored 447 correct (49.67%),
so the best genuine ensemble lost four correct predictions, or 0.44 percentage
points, relative to the strongest individual model.

| Configuration | Correct / 900 | Accuracy | Macro-F1 | Tiebreak rows |
|---|---:|---:|---:|---:|
| RL alone | 447 | 49.67% | 49.32% | 0 |
| SVM alone | 445 | 49.44% | 49.06% | 0 |
| All three, RL tiebreaker | 443 | 49.22% | 48.86% | 5 |
| All three, NorBERTo tiebreaker | 442 | 49.11% | 48.75% | 5 |
| All three, SVM tiebreaker | 440 | 48.89% | 48.52% | 5 |
| Original EP1 NorBERTo alone | 432 | 48.00% | 47.86% | 0 |

## Voting rule and candidate coverage

Every selected member contributes one hard label. A strict majority wins;
otherwise, the configured local arbitrator supplies its saved prediction.
No confidence thresholds, probability averaging, weights, calibration, new
inference, retraining, or remote requests are involved.

The comparison covered 15 configurations: three singletons, three pairs with
each of the three local arbitrators, and the full trio with each arbitrator.
These produce six distinct label sequences. A two-member ensemble that uses
one of its voters to break ties exactly reproduces that voter. The apparent
447-correct pair winners therefore duplicate RL; they do not improve it.
A pair with the third model as arbitrator reproduces the corresponding
three-member majority with that third model as tiebreaker.

All three models agreed on 579 rows. Another 316 rows had a 2-to-1 majority,
and five had three different labels. Pairwise agreement was:

| Pair | Agreement / 900 | Fraction |
|---|---:|---:|
| SVM and RL | 861 | 95.67% |
| NorBERTo and RL | 608 | 67.56% |
| NorBERTo and SVM | 584 | 64.89% |

This high lexical-model agreement limits the diversity of their votes. With
RL resolving ties, the trio changes only five of RL's predictions; these
changes lose four correct answers and gain none. This describes the observed
test1 result, not a general guarantee about future datasets.

## Artifacts, verification, and scope

A subsequent [holdout audit](ensemble-holdout.md) verified training isolation
and scored these same selected artifacts on the frozen 3,999-row partition.

All models trained on the frozen 16,093-row EP1 development partition, keeping
3,999 holdout rows outside fitting. NorBERTo is the unchanged EP1 adapter;
SVM uses fixed C=0.05, and RL uses fixed C=0.25. Selected manifests and saved
run identities were verified before importing predictions. Historical
full-corpus models were excluded.

All candidate predictions and configurations were persisted before reading
the gold workbook. Every one of the 900 source texts, row IDs, and positions
matched `data/test1-rotulado-para-teste-estatistico.xlsx`. An independent
standard-library ZIP/XML reader verified all 13,500 candidate predictions,
strict-majority routing, ties, confusion matrices, accuracy, macro-F1, and
model/run/file hashes. The CPU comparison took 1.27 seconds.

Raw evidence is under `results/ensemble-majority-comparison-20261105/`, with
per-candidate configurations, predictions, and metrics; `summary.json` stores
the ranking and provenance. `verify.py` and `independent-verification.json`
record the independent check.

This is an **exploratory comparison on previously consulted test1 labels**.
Ranking these configurations does not provide an independent final evaluation
or establish statistical superiority. At comparison time, delivery defaults
remained Qwen + NorBERTo. The user subsequently excluded Qwen; current defaults
are NorBERTo + SVM + RL with RL arbitration. The official test2-before-test1
protocol remains unchanged.
No test2 predictions or submission were generated.

## Reproduction

With the existing saved member runs and project dependencies available:

```bash
PYTHONPATH=src python3 -m ensemble_classifier compare-majority \
  --config configs/ensemble-majority-test1.json \
  --input data/test1.xlsx \
  --gold data/test1-rotulado-para-teste-estatistico.xlsx \
  --output-dir results/ensemble-majority-comparison-new
```

Use a new output directory. The separate production configuration,
`configs/ensemble-majority.json`, selects the same models, strict majority,
and RL tiebreaker for test2 once their test2 member predictions exist.
