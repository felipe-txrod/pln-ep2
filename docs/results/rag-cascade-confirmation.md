# RAG context confirmation: stacking without GPT selected

[Aggregate and independent verification](data/rag-cascade-confirmation.json) · [Pilot report](rag-cascade-pilot.md) · [Protocol](../guides/rag-stacking.md)

The manually launched confirmation completed all four context variants, with **94 successful calls, no failed attempts and no fallback**. Its 1,884 OOF validation rows were retained. The best result was **original stacking with zero GPT routing**: 910 correct, accuracy **48.30%**, macro-F1 **48.08%**. Every tested nonzero routing configuration lost both accuracy and macro-F1 against the same baseline.

The best nonzero macro-F1 was local predictions without RAG at 5% routing: 906 correct, accuracy 48.09%, macro-F1 47.84%. Hybrid RAG with three examples plus local predictions scored 47.83% at 5% and 47.82% at 10%. The exploratory pilot gain did not persist on confirmation. No new remote execution, production-default change, frozen package or final-test evaluation was made during scoring.

## Complete paired comparison

Metrics include every confirmation row, not only the routed subset. Five percent routes 94 targets; ten percent routes 188. Both budgets reuse the same saved responses; they are not separate API runs. Corrected and harmed counts compare against stacking on the same rows.

| Context | GPT routing | Correct | Accuracy | Macro-F1 | Corrected | Harmed |
|---|---:|---:|---:|---:|---:|---:|
| Stacking | 0% | 910/1884 | 48.30% | 48.08% | 0 | 0 |
| text | 5% | 907/1884 | 48.14% | 47.80% | 23 | 26 |
| text | 10% | 891/1884 | 47.29% | 46.83% | 36 | 55 |
| local | 5% | 906/1884 | 48.09% | 47.84% | 12 | 16 |
| local | 10% | 901/1884 | 47.82% | 47.52% | 23 | 32 |
| hybrid-3-rag | 5% | 901/1884 | 47.82% | 47.55% | 17 | 26 |
| hybrid-3-rag | 10% | 899/1884 | 47.72% | 47.37% | 36 | 47 |
| hybrid-3-local | 5% | 906/1884 | 48.09% | 47.83% | 17 | 21 |
| hybrid-3-local | 10% | 907/1884 | 48.14% | 47.82% | 29 | 32 |

The stored selection has `budget: 0.0` and the inactive recipe name `text`, chosen by the predefined tie breaker. This means stacking only; it does not mean text-only GPT won. The existing zero-GPT cascade already has this behavior and was preserved.

## Per-class F1 and observed error pattern

| Configuration | F1 c1 | F1 c234 | F1 c5 |
|---|---:|---:|---:|
| Stacking | 44.36% | 45.43% | 54.45% |
| Local, 5% | 44.44% | 44.59% | 54.49% |
| Hybrid RAG + local, 5% | 44.22% | 44.41% | 54.87% |
| Hybrid RAG + local, 10% | 44.24% | 44.14% | 55.08% |

For hybrid RAG plus local predictions at 10%, the number of correctly classified c5 rows increased from 361 to 377 and c1 from 238 to 240. Correct c234 rows fell from 311 to 290. Those changes produced three fewer correct decisions overall: 29 errors corrected versus 32 correct decisions harmed. Predicted c5 frequency increased from 699 to 742, while c234 fell from 699 to 644. This is an observed shift toward high-clarity labels; its cause is not established by this experiment.

Confusion matrices use true rows and predicted columns in order `c1`, `c234`, `c5`.

Stacking:

```text
238 211 138
159 311 200
 89 177 361
```

Hybrid RAG + local, 10%:

```text
240 196 151
166 290 214
 92 158 377
```

## Actual resources

| Context | Successful calls | Estimated USD | Arm wall seconds |
|---|---:|---:|---:|
| text | 23 | 0.341154 | 94.87 |
| local | 24 | 0.363062 | 100.76 |
| hybrid-3-rag | 23 | 0.459068 | 94.38 |
| hybrid-3-local | 24 | 0.496268 | 101.33 |

- Confirmation total: **94 successful calls**, zero failures, within its 128-attempt cap. Identical requests were deduplicated without dropping evaluation rows.
- Known tokens: **957,003 input**, including **236,544 cached input**, and **19,498 output**.
- Configured standard API-equivalent estimate: **USD 1.6595524**. This is not a ChatGPT subscription invoice.
- Summed arm wall time: **391.34 seconds** (6 minutes 31 seconds), excluding local preparation. Mean request latency: **8.09 seconds**; summed request latency: **760.71 seconds**, with two concurrent workers.
- Combined pilot and confirmation: **178 successful calls plus three archived failed pilot attempts**, 181 registered attempts. Known-usage estimate: **USD 3.6825404**. Archived pilot failures retain unknown usage excluded from that estimate.
- Scoring and independent verification made **zero new GPT calls**.

The model/runtime, original texts, training-only RAG examples and prompts remained frozen. Codex 0.166.0 was selected using its immutable versioned executable, avoiding the pilot update interruption. Cache, attempt histories and previous experiments were retained.

## Isolation, verification and next-state boundaries

All 48 pilot similarity groups are excluded from confirmation; 40 additional meta-validation rows in those groups are excluded as well, leaving 1,884 of the original 1,972 OOF rows. Retrieval uses only the original 16,093-row base-training bank, excluding all 3,999 holdout rows. The four contexts are text-only, local predictions, hybrid RAG with three examples, and both evidence sources. Routing depends only on saved OOF predictions and uncertainty.

Every context had a persisted, checksum-verified 1,884-row prediction file before scoring opened the validation labels. An independent standard-library verifier reconstructed routing at 0/5/10%, all twelve metric sets and confusion matrices, corrected/harmed counts, final 10% prediction files, and resource totals directly from the attempt journals. All checks passed.

Selecting zero routing follows the predeclared macro-F1 criterion. Small observed losses are not proof that GPT is universally inferior. The pilot sample and confirmation cohort have different composition and routing coverage, so their absolute metric levels should not be compared as a learning curve. Previous router development on OOF validation and historical Qwen checkpoint exposure remain inherited limitations.

**Test1 and test2 were not opened in this stage.** Test2 has no gold here and only receives a prediction workbook for delivery; final accuracy/F1 require labeled test1 after configuration freezing. No final prediction or freezing command was executed as part of this verification. This completed confirmation does not need to be rerun.

Local evidence: `results/rag-context-pilot-20261106/confirmation/{plan,run,selection,independent-summary}.json` and each context directory with requests, retrieval traces, decisions, predictions and attempt histories. Public aggregates contain no original texts or individual gold labels.
