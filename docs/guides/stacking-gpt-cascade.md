# Frozen stacking with selective GPT decisions

The completed [measured experiment](../results/stacking-gpt-cascade.md) selected
zero GPT calls on validation; all three nonzero budgets also reduced test1
quality. Its frozen package is `models/stacking-gpt-cascade/`. Configurable
routing and explicit budget comparisons remain available as described below.

The cascade reuses SVM, development-only logistic regression, the original EP1
NorBERTo adapter, Qwen, the frozen probability calibrators, and the published
four-member stacking model. No base classifier is refitted. GPT replaces the
stacking decision only for the selected uncertain targets; API failures retain
the original stacking decision and are explicitly marked as fallback.

## Data and selection protocol

The original development partition has 16,093 rows. The 3,999 reserved rows
already have two group-disjoint roles: 2,027 fit probability calibrators and
1,972 train the stacking meta-model. Calibration rows are excluded from routing
development. They cannot serve as clean routing validation for the complete
calibrated pipeline because their labels fitted the calibrators.

Routing development uses **nested five-by-five grouped OOF** predictions on the
1,972 meta rows. Each outer target and its similarity group are excluded from
both meta-model fitting and inner regularization selection. The original
calibrators remain frozen and never used these 1,972 labels. Only the small
meta-model is fitted for these OOF estimates; production stacking and base
weights are unchanged. The saved fold audit includes training/validation IDs,
portable numerical models, and inner CV results.

Qwen checkpoint selection used the full original holdout, and earlier research
consulted holdout/test1. This inherited exposure cannot be undone by meta-model
OOF. The new routing stage excludes its outer targets from supervised meta
fitting, but does not claim untouched validation for the entire historical
model-selection process.

The 900 final texts are never used to select a routing rule, prompt, threshold
or budget. Prediction loads `Text` records without gold. All final prediction
files are persisted and verified before the separate evaluation stage opens
the gold workbook. This is an explicitly requested exploratory test1 experiment;
it does not establish the assignment's test2-before-test1 delivery order.

## Hybrid uncertainty and budgets

For stacking probabilities p, define normalized entropy H, top-two margin M,
and maximum probability P. The confidence uncertainty is
`U = (H + (1-M) + 1.5*(1-P))/3`. Disagreement D is the fraction of the six pairs
of four native member decisions that disagree. SVM uses its original margin
argmax for this vote, even when probability calibration changes its argmax.

The ranking score is `w*U + (1-w)*D`. A target is eligible if its maximum
probability or margin is below the configured threshold, or any members
disagree. A fixed twelve-candidate search on OOF predictions chooses w and
thresholds by captured stacking errors at the initial 10% budget. Stable ties
use text checksums and source row IDs, without labels. The saved search is a
selection diagnostic, not an independent estimate of routing generalization.

Each budget selects at most `floor(N * fraction)` eligible instances in ranked
order. Thus 5%, 10%, and 20% select at most 98, 197, and 394 routing-validation
instances, or 45, 90, and 180 final test1 instances. Fractions apply to the
eligible evaluation cohort's total N, including rows that remain local.
Selections are nested. Zero budget is also compared so GPT is never assumed
to improve the result.

The first remote stage compares text-only and local-prediction prompts on the
same initial 10% subset. Macro-F1 over all OOF rows chooses the prompt; accuracy
and then text-only break ties. Only that prompt is extended to 20%, reusing
previous answers. Budget selection then uses OOF macro-F1 over 0%, 5%, 10% and
20%; ties favor the lower budget. Both choices are frozen before final inference.

## GPT transport and accounting

The default is **gpt-6.1-sol, low**, through the existing `CodexBackend`. It
uses fresh isolated `codex exec` sessions, a strict JSON schema, no tools,
repository context, gold files or session continuation. Original `resp_text`
is sent without cleaning or truncation. Class definitions and instructions
are explicit. The optional context includes native base decisions, calibrated
probabilities and stacking output, identified as fallible predictions.

The CLI exposes no temperature parameter, so temperature is omitted. It
supports low reasoning and structured output; see the official
[model documentation](https://developers.openai.com/api/docs/models/gpt-6.1-sol)
and [Codex non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode).

The remote model identifier is a requested alias. Persistent response replay
supports exact reproducibility; new future requests may differ even with the
same alias and reasoning setting.

SQLite persists validated responses and individual decisions keyed by exact
model, reasoning, instructions, original text, optional predictions and runtime.
Item caching allows reuse when budgets or batch boundaries change. Up to eight
independent targets share a Codex request; two requests may run concurrently.
No labels are copied from one target to another. Exact repeated requests share
one answer. Batch identifiers are opaque, unrelated to source IDs or classes.

Every attempt is journaled before sending. Retries are explicitly bounded,
including after timeouts that might already have consumed tokens. A failed
batch opens a circuit breaker for subsequent unsent batches. Failed calls do
not populate the successful-decision cache. A missing GPT decision retains the
stacking label and a visible `fallback` status; it never invents a GPT answer.

Reports distinguish attempted calls, successful target decisions, cache hits,
fallback, token usage, summed request latency and wall-clock time. API-equivalent
cost uses standard prices recorded in the config: USD 2 input, 0.10 cached
input, and 10 output per million tokens. These are estimates from the official
model page, checked on 2026-11-05, **not an invoice for ChatGPT subscription
usage**. Failed attempts with unavailable usage are counted and flagged; their
unknown cost is not silently treated as known zero. Per-budget accounting uses
per-instance shares of shared batches; actual experiment totals are separate.

## Configuration and execution

Use the project's Python environment with its pinned CPU dependencies and an
authenticated Codex CLI >=0.165.0. `codex login` can provide existing ChatGPT
authentication. Alternatively, supply `CODEX_API_KEY` through the environment
for the Codex invocation. Do not write credentials into config files or Git.

`configs/cascade.json` defines the experiment. Optional `CASCADE_MODEL`,
`CASCADE_REASONING_EFFORT`, `CASCADE_BATCH_SIZE`, `CASCADE_WORKERS`,
`CASCADE_RETRIES`, `CASCADE_TIMEOUT`, and the three `CASCADE_*_PRICE_PER_MILLION`
variables override it **during preparation**. The resolved configuration is
then immutable. `CASCADE_CODEX_BIN` selects the executable and `CASCADE_CACHE`
selects the persistent cache. Runtime identity must match validation at final
inference. The defaults remain the explicitly requested model and effort.

From the EP2 root, use fresh preparation/publication/prediction destinations:

```bash
PYTHONPATH=src python3 -m cascade_classifier prepare \
  --config configs/cascade.json --output-dir results/cascade-validation-new

PYTHONPATH=src python3 -m cascade_classifier validate-initial \
  --validation-dir results/cascade-validation-new

PYTHONPATH=src python3 -m cascade_classifier validate-extended \
  --validation-dir results/cascade-validation-new

PYTHONPATH=src python3 -m cascade_classifier freeze \
  --validation-dir results/cascade-validation-new \
  --output-dir models/stacking-gpt-cascade-new
```

Preparation reuses pinned member score runs and needs no GPU or GPT call.
The two validation commands perform remote classification. Stages can resume
with their original cache/journal; changing inputs requires a new directory.
An interrupted unknown-usage attempt counts toward the retry limit.

For a final comparison, reuse a verified, completed stacking prediction run:

```bash
PYTHONPATH=src python3 -m cascade_classifier predict \
  --model models/stacking-gpt-cascade-new/model.json \
  --input data/test1.xlsx \
  --stacking-run results/ensemble-stacking-qwen-test1-20261105 \
  --compare-budgets --output-dir results/cascade-test1-new

PYTHONPATH=src python3 -m cascade_classifier evaluate \
  --run results/cascade-test1-new \
  --gold data/test1-rotulado-para-teste-estatistico.xlsx \
  --output-dir results/cascade-test1-evaluation-new
```

`--compare-budgets` explicitly queries the union up to the frozen 20% limit;
smaller-budget variants reuse those answers. Without it, `predict` uses only
the validation-selected budget and the local baseline, avoiding unnecessary
experimental calls. If validation selects zero, production prediction makes
no GPT requests. For another input, first generate a verified stacking run
using the [four-model workflow](ensemble-qwen.md), then pass it here.

Reports include macro-F1, accuracy, per-class F1, confusion matrices, corrected
and harmed instances, fallback and resource consumption. Evaluation never
changes a frozen rule, model, prompt or budget.
