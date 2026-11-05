# Frozen stacking + selective GPT: measured experiment

[Results index](README.md) · [Commands and protocol](../guides/stacking-gpt-cascade.md) · [Verified aggregate](data/stacking-gpt-cascade.json)

**Selective GPT did not improve this experiment.** Validation selected the local stacking
baseline, with zero GPT calls, before the final gold workbook was opened. All three
predeclared nonzero budgets also reduced macro-F1 and accuracy on test1. The implemented
cascade remains available with configurable budgets, persistent cache, retries and fallback;
its selected production behavior retains stacking. No classifier or routing rule was
changed in response to final scores.

## Validation-only model and budget selection

The frozen base models use the original 16,093 development rows. The reserved 3,999
rows retain their original roles: 2,027 fit probability calibrators; 1,972 support
meta-training and routing development through **nested 5x5 grouped OOF predictions**.
Each outer target and its similarity group are excluded from both meta fitting and
inner C selection. All five outer folds selected C=0.1 using their own training data.
The production stacking model and all base artifacts remain byte-identical.

Routing combines normalized entropy, maximum probability, top-two margin and native
pairwise disagreement among SVM, RL, NorBERTo and Qwen. The fixed twelve-candidate
search captured 122 OOF errors among the initial 197 routed rows. All twelve candidates
captured the same number; the deterministic tie rule selected confidence weight 0.9,
maximum-probability threshold 0.5 and margin threshold 0.05. Ranking and eligibility
for final targets depend exclusively on predictions, with no gold inputs.

At the initial 10% budget, prompts were compared on the same selected rows:

| Validation method | Correct / 1,972 | Macro-F1 | Corrected | Harmed |
|---|---:|---:|---:|---:|
| Nested OOF stacking | 936 | 47.2977% | 0 | 0 |
| GPT with original text only | 925 | 46.5751% | 31 | 42 |
| GPT with local predictions | 929 | 46.8407% | 22 | 29 |

Local predictions helped relative to the text-only prompt, but both underperformed
stacking. Only the local-context prompt was extended to 20%, reusing all 197
previous decisions. The frozen prompt includes the original text, class definitions,
native local labels, calibrated probabilities and stacking probabilities; these
are identified as fallible predictions, never as ground truth.

Validation budget comparison:

| GPT budget | Routed | Correct | Accuracy | Macro-F1 | Corrected | Harmed |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 0 | 936/1972 | 47.4645% | 47.2977% | 0 | 0 |
| 5% | 98 | 937/1972 | 47.5152% | 47.2762% | 16 | 15 |
| 10% | 197 | 929/1972 | 47.1095% | 46.8407% | 22 | 29 |
| 20% | 394 | 937/1972 | 47.5152% | 47.1016% | 56 | 55 |

The 5% and 20% budgets each gained one net correct answer, but reduced the primary
macro-F1 metric. Zero budget therefore won the validation selection. These are
selection diagnostics, not independent estimates after tuning.

## Final comparison: all 900 original rows

The prompt, routing formula, thresholds, budget choices and source fingerprints
were frozen in `models/stacking-gpt-cascade/` before final inference. The union
of the 5/10/20% selections required at most 180 targets. Every prediction file
was saved before the separate evaluator opened test1 gold. No final label
selected the forwarded examples or modified the frozen configuration.

Measured test1 results:

| GPT budget | Routed | Correct | Accuracy | Macro-F1 | Corrected | Harmed |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 0 | 440/900 | 48.8889% | 48.8034% | 0 | 0 |
| 5% | 45 | 435/900 | 48.3333% | 48.1824% | 4 | 9 |
| 10% | 90 | 434/900 | 48.2222% | 48.0477% | 9 | 15 |
| 20% | 180 | 436/900 | 48.4444% | 48.1761% | 30 | 34 |

The lower-budget results reuse answers from the same nested ranked selections.
Other changed labels remained incorrect: 8 at 5%, 11 at 10%, and 20 at 20%.
There were no failed calls or fallback instances in the measured runs.

| GPT budget | F1 c1 | F1 c234 | F1 c5 |
|---|---:|---:|---:|
| 0% | 45.7594% | 46.4023% | 54.2484% |
| 5% | 45.3280% | 46.2687% | 52.9506% |
| 10% | 45.3280% | 45.5235% | 53.2915% |
| 20% | 45.6000% | 45.0704% | 53.8578% |

Confusion matrices use true classes in rows and predicted classes in columns,
ordered `c1, c234, c5`:

```text
GPT 0%:
[116, 125, 59]
[55, 158, 87]
[36, 98, 166]

GPT 5%:
[114, 117, 69]
[53, 155, 92]
[36, 98, 166]

GPT 10%:
[114, 115, 71]
[53, 150, 97]
[36, 94, 170]

GPT 20%:
[114, 108, 78]
[51, 144, 105]
[35, 87, 178]

```

## Calls, tokens, cost and latency

The runtime was `codex-cli 0.166.0`, requesting `gpt-6.1-sol` with reasoning
`low`. Original text was preserved, strict JSON was validated, and the isolated
CLI had tools and repository context disabled. The CLI exposes no temperature
control. The remote model is a requested alias; replay uses the recorded response
cache rather than assuming future requests are byte-deterministic.

| Stage | Target instances | Item cache hits | New calls | API-equivalent USD | Wall seconds |
|---|---:|---:|---:|---:|---:|
| validation_text_only_10pct | 197 | 0 | 24 | 0.374843 | 96.02 |
| validation_with_local_10pct | 197 | 0 | 25 | 0.450464 | 94.07 |
| validation_selected_prompt_extension_20pct | 394 | 197 | 25 | 0.455602 | 91.25 |
| final_frozen_budget_comparison | 180 | 0 | 23 | 0.391949 | 86.17 |

Total: **97 calls**, **883,532 input tokens**
(including 147,840 cached input tokens),
**18,669 output tokens**, and **USD 1.672858**
as a standard API-equivalent estimate. Authentication used the existing ChatGPT
Codex login; the estimate is not a measured subscription invoice. Prices were
checked on 2026-11-05 against the official
[GPT-6.1 Sol model page](https://developers.openai.com/api/docs/models/gpt-6.1-sol):
USD 2 input, 0.10 cached input, and 10 output per million tokens.

The complete final comparison took 86.17 wall seconds with two concurrent
requests; summed request latency was 171.38 seconds. Each request grouped up
to eight independent targets. Reports allocate each shared batch proportionally
to its unique targets for lower-budget cost/latency accounting:

| Final GPT budget | Routed | Batch receipts touched | Allocated USD | Allocated request seconds |
|---|---:|---:|---:|---:|
| 0% | 0 | 0 | 0.000000 | 0.00 |
| 5% | 45 | 6 | 0.103906 | 44.47 |
| 10% | 90 | 12 | 0.205357 | 86.41 |
| 20% | 180 | 23 | 0.391949 | 171.38 |

These are allocations from the shared experiment, not independent timing or
billing measurements from rerunning each budget. The selected zero-budget
production path was also actually executed: zero GPT attempts, with all 900
labels exactly matching frozen stacking. Replaying all 180 final GPT decisions
with the backend request limit set to zero reproduced every cached label, with
180 item-cache hits and no new attempts.

## Verification and limitations

All **201 available tests passed**, including the SDK modules. Tests cover
percentage caps, all-four-member native disagreement, outer/inner group isolation,
original-text preservation, cache reuse across budgets and batch boundaries,
bounded retries, circuit breaking, explicit fallback, immutable outputs and
final-gold access only after all predictions are persisted.

An independent standard-library verifier checked original workbook identities,
all nested OOF coefficient outputs, routing arithmetic, percentage caps, cached
GPT request/response provenance, validation-only prompt/budget selection, final
classifications, per-class F1, confusion matrices and every corrected/harmed case.
Maximum OOF probability reconstruction error was 1.67e-16.
Raw evidence and the independent script remain in
`results/cascade-verification-20261105/`; routing validation, final prediction
and scoring have separate preserved directories.

Qwen checkpoint selection previously used all 3,999 holdout rows. Official files
also contain normalized-text repetitions across test1 and development (85 rows),
calibration (11) and meta-training (9), as recorded by the earlier
[overlap audit](ensemble-soft.md). No new test1 label entered fitting,
but OOF meta prediction cannot remove this inherited exposure without changing
the original models/data. All 900 rows remain evaluated. This is an exploratory
comparison with historical test1 exposure, not a new untouched final sample.

No base model was retrained, no original experiment was overwritten, no remote
configuration was tuned after final scoring, and no test2 submission was generated.
