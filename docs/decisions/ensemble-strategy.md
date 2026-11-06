# Decision: retain weighted soft voting as the primary approach

Date: 2026-11-06. Status: accepted by the project team.

[Documentation index](../README.md) · [Results index](../results/README.md)

## Evaluation of alternative ensemble strategies

Alongside soft voting, the project investigated more complex classifier
combinations, including stacking and selective classification assisted by GPT.
The GPT experiments compared different contexts, including retrieval-augmented
generation (RAG) with labeled historical training examples supplied during
inference. Retrieval alternatives included lexical, semantic and hybrid search,
as well as representations from the NorBERTo checkpoint previously fine-tuned
for this task.

Some exploratory experiments appeared promising, but their gains did not persist
in subsequent evaluations. GPT corrected some stacking errors while introducing
errors on instances that stacking had classified correctly. Changes to retrieval,
routing and acceptance policies did not demonstrate consistent improvements.

Weighted soft voting with **75% SVM and 25% NorBERTo** achieved **49.58% macro-F1
on Test1**, exceeding the stacking configurations evaluated on the same dataset.
The compared frozen results retain all 900 Test1 instances:

| Method | Correct / 900 | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Weighted soft voting | 447 | 49.67% | 49.58% |
| Stacking with SVM, LR and NorBERTo | 442 | 49.11% | 49.03% |
| Stacking with SVM, LR, NorBERTo and Qwen | 440 | 48.89% | 48.80% |

The soft-voting weights were originally selected using validation data. Both
the three-model artifact and the four-candidate artifact including Qwen assign
zero weight to LR; the latter also assigns zero weight to Qwen and reproduces
the same 900 predictions. The decision retains the existing calibrated
probability mixture, rather than a new combination of uncalibrated scores.

## Accepted decision

Retain **weighted soft voting as the project's primary classification approach**,
using the frozen SVM 0.75 / NorBERTo 0.25 weights. Its measured performance,
implementation simplicity and lack of additional language-model calls support
this choice.

Preserve stacking, selective GPT and RAG implementations, artifacts, caches and
results as part of the investigation. The experiments demonstrate that greater
ensemble complexity does not necessarily improve predictive performance.
They remain available for reproducibility and analysis; this decision does not
authorize additional training, GPT calls or experiment runs.

This record documents the approach choice without changing classifier code,
frozen packages or command defaults, and without generating a new Test2 delivery.

## Interpretation and limitations

The conclusion reflects the recorded data partitions and the exploratory nature
of part of the investigation. The Test1 comparison is retrospective: Test1 had
already been examined in earlier project work, and choosing the primary method
after comparing these scores does not constitute a new independent evaluation.
The numerical advantage is not a claim of statistically significant superiority.
Validation scores from different populations are not directly interchangeable
with the 900-row Test1 scores above.

Qwen checkpoint selection had already used the original holdout, and the
historical routing/acceptance replay retained fixed upstream OOF predictions
and limited logged GPT support. These limitations remain disclosed in the
source reports. Improvements in retrieval label matching are auxiliary results
and do not establish an improvement in GPT classification macro-F1.

## Evidence preserved

- [Original calibrated soft-voting protocol and Test1 results](../results/ensemble-soft.md).
- [Three- and four-model ensemble comparison, weights and limitations](../results/ensemble-qwen.md).
- [Selective stacking + GPT experiment](../results/stacking-gpt-cascade.md).
- [Exploratory RAG pilot](../results/rag-cascade-pilot.md) and
  [subsequent confirmation](../results/rag-cascade-confirmation.md).
- [Offline NorBERTo retrieval, MMR and intervention investigation](../results/rag-offline-research.md).
- Frozen soft-voting artifacts:
  [original three-model package](../../models/ensemble-soft/README.md) and
  [four-candidate package](../../models/ensemble-soft-qwen/README.md).
