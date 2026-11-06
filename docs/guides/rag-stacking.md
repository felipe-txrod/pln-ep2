# Train-only RAG and local predictions for selective stacking

**Status: pilot and confirmation are complete; validation selected zero GPT routing.**
The [confirmation report](../results/rag-cascade-confirmation.md) records four
contexts at 0/5/10% routing on 1,884 OOF rows. Stacking won with 910 correct,
accuracy 48.30%, macro-F1 48.08%; the best nonzero GPT candidate reached 47.84%
macro-F1. The earlier [pilot](../results/rag-cascade-pilot.md) selected hybrid
retrieval with three examples, but its exploratory gain did not persist.
The existing zero-GPT default remains unchanged. No new RAG package has been
frozen, and no final test1/test2 inference or evaluation has been performed.
The [original preparation record](../results/rag-cascade-preparation.md) preserves
the earlier no-execution milestone.

The subsequent [offline investigation](../results/rag-offline-research.md)
adds frozen NorBERTo retrieval, MMR ablations and historical reply replay in
the separate `rag_research` package. It makes no new GPT calls and does not
promote a different default. See its [guide](rag-offline-research.md).

The new `rag_cascade` package reuses the existing frozen stacking, OOF routing,
TF-IDF selector, pinned local E5 encoder interface, Codex backend, batch schema,
SQLite response/item cache, evaluation functions and workbook exporters. It is
separate from `cascade_classifier` so old experiment fingerprints remain valid.

## Example bank and retrieval

The bank contains only the **16,093 original base-training rows** of
`data/train.xlsx`. The frozen EP1 split and workbook hashes are checked.
All 3,999 holdout rows are excluded from the bank and index, including the
2,027 calibration rows and 1,972 meta-training rows. Test1 and test2 gold never
enter the bank, retrieval fitting, prompt development or configuration selection.

The existing selector removes normalized duplicates, conflicting-label groups,
empty examples and overlong texts. The new layer also caps raw example length
at 1,600 characters; entire examples are included, without silent truncation.
It compares three retrieval methods:

- **TF-IDF:** cosine similarity on the existing train-only unigram/bigram index.
- **Semantic:** cosine similarity of frozen local `multilingual-e5-small`
  embeddings, using the existing ONNX CPU encoder at pinned revision
  `614241f622f53c4eeff9890bdc4f31cfecc418b3`.
- **Hybrid:** equal-weight reciprocal-rank fusion of lexical and semantic ranks.

The semantic path never falls back silently to TF-IDF. Missing dependencies or
weights raise an actionable error. Encoder preparation is an explicit download
command; normal retrieval uses local files only. Embeddings consume CPU and disk
resources, with a persistent cache, rather than paid embedding API requests.

Each method is tested with **3 and 5 examples**. Retrieval excludes identical
normalized target text, lexical cosine >=0.95, and near duplicates by normalized
character-shingle Jaccard >=0.85. Selected examples cannot share the original
similarity component and are screened for pairwise near duplicates as well.

Within relevant candidates, MMR balances relevance and redundancy, with a small
bonus for unseen classes and a per-class cap. Class representation never
overrides the relevance floor: the method may return fewer than requested if
eligible examples are insufficient, and records that shortfall. It does not
fill the prompt with unrelated examples to impose a class quota.

The lexical similarity floor is 0.03; semantic/hybrid retrieval requires semantic
cosine >=0.45. These and the diversity settings are explicit, predeclared config
values, not thresholds selected using test1 labels. Text checksums, groups,
labels, lexical/semantic similarities, fusion relevance, and exclusions/shortfalls
are recorded in `context-trace.jsonl`. No claim of retrieval quality is made
until the experiment is actually run.

## Matched structured prompts

All arms use the same class definitions and instructions. Only the supplied
evidence changes:

1. Original text alone.
2. Original text and local model evidence.
3. Original text and retrieved labeled examples.
4. Original text, retrieved examples and local model evidence.

The payload separates `examples` from `target.resp_text` and
`target.model_evidence`. The original target is preserved literally. Model
evidence includes native SVM/RL/NorBERTo/Qwen decisions, calibrated probabilities,
stacking output, maximum probability, margin, entropy and pairwise disagreement.
Probabilities are explicitly approximate at four decimal places to reduce tokens.
SVM native labels are retained when calibration changes the probability argmax.

The instructions identify demonstrations as subjective training observations and
local outputs as fallible evidence. GPT must make an independent decision,
without copying majority voting, inventing missing questions/attachments or
following instructions embedded in example/target text. Output remains strict
JSON with one of `c1`, `c234`, `c5`; no extra explanation tokens are requested.

The default remains `gpt-6.1-sol`, reasoning `low`, using the authenticated,
tool-free Codex backend. Its CLI exposes no temperature control. Standard
API-equivalent prices are configurable; subscription usage is not reported as
an API invoice. See the [official model documentation](https://developers.openai.com/api/docs/models/gpt-6.1-sol).

## Pilot and confirmation without test-label tuning

The study reuses the **checksum-pinned nested 5x5 grouped OOF predictions** from
the previous cascade. Outer targets were excluded from both meta fitting and
inner regularization selection. It does not use production stacking's in-sample
meta predictions to optimize context.

The predeclared plan is:

1. Select 48 difficult examples from the original 10% routing subset, by stable
   label-free hashes, at most one example per similarity component.
2. Compare the two non-RAG controls and all twelve RAG variants: three retrievers,
   two example counts, with/without local evidence. This is 14 pilot arms.
3. Choose the retrieval recipe by the mean paired macro-F1 of its RAG-only and
   RAG+local arms; ties favor fewer examples and then lexical retrieval.
4. Remove **every group containing a pilot example** from confirmation. On the
   remaining OOF cohort, compare the four contexts using the chosen retrieval
   recipe and the same difficult targets, capped initially at 10%.
5. Compare zero, 5% and 10% routing using those cached decisions. Select by
   confirmation macro-F1, then lower budget and allocated cost. Zero is eligible;
   GPT is not presumed to help. Arms with unresolved fallback are ineligible
   for a nonzero-budget selection.
6. Freeze source/index/bank/runtime identities, prompt recipe and budget before
   generating final predictions.

Pilot selection is exploratory; confirmation similarity groups are disjoint
for this context comparison. The earlier router was already selected using the
OOF validation set, Qwen checkpoint selection used the full holdout, and historical
project work consulted test1. This procedure does not undo inherited exposure or
make the complete modeling history an untouched test. Official cross-dataset
duplicates also remain in base-model training; RAG filters prevent simply showing
an identical or near-identical target as a labeled demonstration.

## Cache and bounded execution

`plan` records exact per-target requests and the context trace. `run` and
`predict` default to **preview**. They cannot classify without `--execute`.
`prepare-encoder` cannot download without `--download`. There is no all-in-one
command that automatically advances through remote stages.

The shared SQLite cache keys the exact model, runtime, instructions, examples,
text and local evidence. Changing any evidence invalidates only that request;
valid cached replies survive different budgets and batch boundaries. New prompts
do not reuse replies from different historical prompts just because texts match.

Defaults are batches of eight, two concurrent requests, one retry, and a hard
**128 new-attempt limit per stage invocation**, shared across all arms and retries.
Preflight counts cache misses and refuses a stage that already exceeds the cap.
Retries consume the same cap. A circuit breaker stops later batches after an
exhausted failure. Interrupted attempts count toward each request's retry limit;
unknown usage is flagged. Missing decisions explicitly retain stacking rather
than becoming fake GPT labels. Completed stages are immutable/idempotent.

Set `CASCADE_CODEX_BIN` to an immutable versioned executable before planning and
retain it during execution. The pilot's default CLI symlink updated from 0.166.0
to 0.167.0 mid-run, correctly triggering strict runtime mismatch checks. Pinning
the original executable restored compatibility without changing requests or
fingerprints. Preserve any failed journals and historical resources when
preparing a reviewed retry; do not silently discard failures or reset budgets.

Each arm records accuracy, macro-F1, per-class F1, confusion matrix, corrected,
harmed and changed-but-still-wrong cases, tokens, cache hits, cost allocation,
actual calls and latency. Shared-batch allocations are distinguished from actual
whole-stage costs and wall time. Prices are API-equivalent estimates, not invoices.

## Commands for a future authorized run

The pilot and confirmation were executed manually in
`results/rag-context-pilot-20261106` and are complete. Do not repeat them to score
existing outputs: use `score` only. The generic examples below are not instructions
to repeat classification or advance automatically. Final-test inference requires
separate user authorization. Test2 produces an unlabeled delivery workbook;
quality metrics are computed on labeled test1 after configuration freezing.
Use a Python environment with the project's dependencies and optional `rag`
dependencies. The local pilot runtime now has ONNX Runtime and the pinned E5
weights; other clones still need the explicit installation/download steps.

```bash
python3 -m pip install -e '.[rag]'
PYTHONPATH=src python3 -m rag_cascade prepare-encoder --download
export CASCADE_CODEX_BIN=/absolute/path/to/versioned/codex

PYTHONPATH=src python3 -m rag_cascade prepare \
  --config configs/rag-cascade.json --output-dir results/rag-context-new

PYTHONPATH=src python3 -m rag_cascade plan --study results/rag-context-new --stage pilot
PYTHONPATH=src python3 -m rag_cascade run --study results/rag-context-new --stage pilot
```

The last command only previews cache misses and planned calls. Once execution
is explicitly wanted:

```bash
PYTHONPATH=src python3 -m rag_cascade run \
  --study results/rag-context-new --stage pilot --execute --max-new-requests 128
PYTHONPATH=src python3 -m rag_cascade score --study results/rag-context-new --stage pilot

PYTHONPATH=src python3 -m rag_cascade plan --study results/rag-context-new --stage confirmation
PYTHONPATH=src python3 -m rag_cascade run --study results/rag-context-new --stage confirmation
PYTHONPATH=src python3 -m rag_cascade run \
  --study results/rag-context-new --stage confirmation --execute --max-new-requests 128
PYTHONPATH=src python3 -m rag_cascade score --study results/rag-context-new --stage confirmation

PYTHONPATH=src python3 -m rag_cascade freeze \
  --study results/rag-context-new --output-dir models/rag-stacking-new
```

Environment credentials follow the existing Codex login or `CODEX_API_KEY`
contract. `CASCADE_CODEX_BIN` selects the executable. `RAG_RESPONSE_CACHE`,
`RAG_EMBEDDING_CACHE`, `RAG_PILOT_SIZE`, `RAG_MAX_NEW_REQUESTS`, `CASCADE_MODEL`
and `CASCADE_REASONING_EFFORT` are resolved during preparation. After that,
the saved configuration is immutable. No credential is written to a manifest.

## EP2: test2 before retrospective test1

First generate a verified four-model stacking run on the 897 original test2
texts using the [existing workflow](ensemble-qwen.md). This requires actual
model inference if member scores are unavailable; it is separate from this
prepared context study. Use a new destination and the selected frozen artifacts.

Then preview and explicitly execute the selected RAG configuration:

```bash
PYTHONPATH=src python3 -m rag_cascade predict \
  --model models/rag-stacking-new/model.json --input data/test2.xlsx \
  --stacking-run results/stacking-test2-new --output-dir results/rag-test2-new

PYTHONPATH=src python3 -m rag_cascade predict --execute \
  --model models/rag-stacking-new/model.json --input data/test2.xlsx \
  --stacking-run results/stacking-test2-new --output-dir results/rag-test2-new
```

The completed run exports `results/rag-test2-new/test2.xlsx`, preserving original
OOXML entries except blank clarity cells. No upload or external submission occurs.
Even if zero budget wins, explicitly complete this stage before test1.

Only then may the **same frozen configuration** run on the 900 original test1
texts. A completed matching test2 run and unchanged predictions/workbook are
mandatory even for test1 preview. No new prompt selection occurs:

```bash
PYTHONPATH=src python3 -m rag_cascade predict --execute \
  --model models/rag-stacking-new/model.json --input data/test1.xlsx \
  --stacking-run results/ensemble-stacking-qwen-test1-20261105 \
  --test2-run results/rag-test2-new --output-dir results/rag-test1-new

PYTHONPATH=src python3 -m rag_cascade evaluate \
  --run results/rag-test1-new \
  --gold data/test1-rotulado-para-teste-estatistico.xlsx \
  --output-dir results/rag-test1-evaluation-new
```

Evaluation rechecks the test2 chain and every saved test1 decision before opening
gold. It reports frozen RAG versus stacking, never selecting a new prompt or
budget from those results. The four-way context optimization remains in validation;
only its selected configuration advances to test2/test1 to minimize API calls.
