# RAG context pilot: hybrid retrieval with three examples selected

[Aggregate metrics and accounting](data/rag-cascade-pilot.json) · [Protocol and commands](../guides/rag-stacking.md) · [Original preparation record](rag-cascade-preparation.md)

Follow-up: the [completed confirmation](rag-cascade-confirmation.md) selected
zero GPT routing; the pilot gain below did not persist. This report preserves
the earlier exploratory milestone and its original selection.

The manually launched pilot completed all **14 arms on the same 48 difficult OOF targets**, with 672 valid classifications and no final fallback. The best observed arm was **hybrid retrieval with three examples plus local predictions**: 19/48 correct, accuracy 39.58%, macro-F1 36.90%, versus stacking at 16/48, 33.33%, and 27.59%. It corrected eight stacking errors and harmed five correct stacking decisions.

The predeclared paired-arm mean selected **hybrid retrieval with three examples** for a possible later confirmation. This selects the retrieval recipe only. Context strategy and routing budget have not been confirmed, and no RAG model was frozen or promoted. The existing zero-GPT cascade default is unchanged.

## All pilot arms

All rows below use the same 48 targets. These are difficult selected cases, not overall holdout or test performance. Costs come from actual attempt journals; reused per-item allocations are not summed into the total.

| Context | Correct | Accuracy | Macro-F1 | Corrected | Harmed | Estimated USD |
|---|---:|---:|---:|---:|---:|---:|
| Stacking | 16/48 | 33.33% | 27.59% | — | — | 0 |
| text | 14/48 | 29.17% | 26.30% | 9 | 11 | 0.102298 |
| local | 18/48 | 37.50% | 33.07% | 6 | 4 | 0.123320 |
| tfidf-3-rag | 17/48 | 35.42% | 32.25% | 12 | 11 | 0.145720 |
| tfidf-3-local | 14/48 | 29.17% | 25.41% | 7 | 9 | 0.150175 |
| tfidf-5-rag | 16/48 | 33.33% | 30.17% | 9 | 9 | 0.164544 |
| tfidf-5-local | 15/48 | 31.25% | 28.65% | 6 | 7 | 0.168521 |
| semantic-3-rag | 11/48 | 22.92% | 19.72% | 6 | 11 | 0.127561 |
| semantic-3-local | 14/48 | 29.17% | 26.27% | 3 | 5 | 0.145309 |
| semantic-5-rag | 15/48 | 31.25% | 27.82% | 10 | 11 | 0.150446 |
| semantic-5-local | 14/48 | 29.17% | 25.41% | 5 | 7 | 0.179230 |
| hybrid-3-rag | 16/48 | 33.33% | 29.91% | 10 | 10 | 0.116455 |
| hybrid-3-local | 19/48 | 39.58% | 36.90% | 8 | 5 | 0.117164 |
| hybrid-5-rag | 15/48 | 31.25% | 29.78% | 11 | 12 | 0.148044 |
| hybrid-5-local | 14/48 | 29.17% | 25.60% | 5 | 7 | 0.184201 |

Names ending in `rag` contain retrieved examples without local predictions; `local` RAG arms contain both. `text` is text-only; the plain `local` arm has local predictions without retrieved examples. Each RAG method compares three and five demonstrations.

## What helped in this sample

- Local predictions improved text-only GPT from 14 to 18 correct, macro-F1 26.30% to 33.07%.
- Adding three hybrid demonstrations to local predictions produced one further correct case, macro-F1 33.07% to 36.90%.
- The best RAG-only arm was TF-IDF with three demonstrations: 17 correct, macro-F1 32.25%. Hybrid RAG-only had 16 correct and macro-F1 29.91%.
- Five demonstrations did not beat the selected three-example hybrid recipe. More context was not consistently better; semantic-only retrieval was especially weak.
- The strongest combined arm still recovered only 3/17 true c1 cases. All three classes improved in F1 relative to stacking, but class c1 remains a major limitation.

These observations do not identify a causal mechanism for individual mistakes. Similar wording or topic may not convey a subjective clarity rating; that remains a hypothesis, not a measured explanation.

## Per-class results and confusion matrices

| System | F1 c1 | F1 c234 | F1 c5 |
|---|---:|---:|---:|
| Stacking | 16.67% | 47.06% | 19.05% |
| Text | 10.00% | 38.89% | 30.00% |
| Local | 17.39% | 51.06% | 30.77% |
| Hybrid 3 + local | 25.00% | 50.00% | 35.71% |

Rows are true classes; columns are predicted classes, ordered `c1`, `c234`, `c5`.

Stacking:

```text
 2 10  5
 5 12  4
 0  8  2
```

Hybrid 3 + local:

```text
 3  7  7
 4 11  6
 0  5  5
```

## Resources and CLI-update recovery

The complete pilot used **84 successful batched calls**, plus **three archived failed attempts**, for 87 registered attempts within the original cap of 128. The final resume made ten new successful calls and reused 592 cached classifications. A Codex update changed the default executable from 0.166.0 to 0.167.0 during the first invocation. Resuming with the original versioned 0.166.0 executable restored exact planned metadata; no prompts, source fingerprints, base models or labels were changed.

- Recorded usage: **1,065,917 input tokens**, including **147,840 cached input tokens**, and **17,205 output tokens**.
- Total standard API-equivalent estimate: **USD 2.022988**, using configured prices of USD 2 input / 0.10 cached input / 10 output per million. This is not a ChatGPT subscription invoice.
- Mean successful request latency: **7.93 seconds**. Summed successful request latency: **666.49 seconds**; concurrency means this is not wall time.
- Archived failed attempts added 0.55 seconds of measured latency and have unknown token usage. Zero recorded failed-attempt tokens do not prove zero remote consumption; the estimate excludes unknown usage.
- Run reports from the final invocation contain cache-replay wall times for earlier arms. A cumulative pilot wall time across manual interruption/resumption is not available.
- No GPT calls were made to score or verify this pilot.

The authoritative totals sum the 84 successful active attempt histories once, plus the three archived failure histories. Per-arm amortized cached-receipt estimates can reuse earlier batches and must not be added as if each were a new billed call. Recovery preserves the failed journals, prior partial outputs, checksums and resource notes.

## Isolation, verification and limits

The original 16,093-row training bank excludes all 3,999 holdout rows. Filtering/deduplication leaves 11,562 indexed examples. All selected demonstrations came from that bank, with three/five complete examples returned for every RAG target. The frozen four-member stacking and nested OOF routing predictions were reused without base retraining. Independent preflight checked all 672 prompts for literal target text, valid train-only labels, local evidence, duplicates and diversity constraints.

All 14 prediction sets were complete and checksum-verified before scoring opened the validation gold. An independent standard-library calculation reproduced every confusion matrix, accuracy, macro-F1, and corrected/harmed count. The 1,884-row confirmation cohort remains unused and has no similarity groups shared with the pilot. Test1 and test2 were not accessed by this experiment.

The observed best arm gains three correct cases over stacking. The descriptive exact two-sided McNemar p-value is **0.5811**, before accounting for searching 14 arms. This small exploratory sample does **not** establish a significant generalization improvement. The nested meta-model OOF protocol excludes meta in-sample fitting; inherited exposure from historical Qwen checkpoint selection on the holdout remains.

Further classification requires explicit authorization. No confirmation, new frozen RAG package, test2 delivery or retrospective test1 evaluation was started.

Local evidence: `results/rag-context-pilot-20261106/pilot/{plan,run,selection,independent-summary}.json`, per-arm requests/context traces/predictions/attempts, `runtime-recovery-20261106/`, and the persistent response/embedding caches. Raw evidence is local and ignored by Git; the linked aggregate preserves the public metrics without texts or individual labels.
