# Offline retrieval and selective-intervention research

[Documentation index](../README.md) · [Configuration](../../configs/rag-research.json)

The [completed study](../results/rag-offline-research.md) found no candidate
meeting the paid-trial screen. Keep the historical stacking/zero-GPT default.

This workflow adds no GPT execution command. It inspects frozen artifacts,
extracts local representations, compares retrieval methods, and replays existing
GPT answers. It does not modify SVM, LR, NorBERTo, Qwen, stacking, their training
partitions, historical experiment packages or response-cache identities.

The independent `rag_research` package preserves the legacy source fingerprints
checked by `rag_cascade` and `cascade_classifier`. A source snapshot and SHA-256
identities accompany measured experiments. Research code is not promoted to the
ensemble defaults automatically.

## Data boundaries

- Bank: the original 16,093 base-training rows only. All 3,999 holdout rows stay
  outside retrieval fitting. The inherited selector deduplicates normalized
  texts, removes conflicting-label duplicates, and limits raw example length
  to 1,600 characters; 11,562 examples remain.
- Queries: the 1,972 meta-training rows with checksum-pinned nested OOF stacking
  outputs. The 2,027 calibration rows are not reused as independent queries.
- Retrieval development: original outer folds 0/1/2. Retrieval audit: folds 3/4.
  Similarity groups do not cross this boundary. Query gold is consumed only by
  the scorer after examples are selected, never by the retriever.
- Intervention research: only the 48 pilot plus 188 confirmation queries have
  GPT outcomes for all four matched contexts. Unqueried outcomes are unknown;
  routing replay is explicitly restricted to logged support.
- No test1/test2 input or gold is used. Historical Qwen checkpoint exposure and
  previous OOF router development remain inherited limitations.

## Frozen representations and caches

NorBERTo is a 22-layer ModernBERT, hidden size 768, with the original fine-tuned
rank-4 LoRA adapter on Wqkv. The verified adapter SHA-256 begins `d693f6e2`.
The original classifier uses mean pooling; the actual tokenizer inserts [CLS]
ID 3 as its first token. Both masked mean and first-token pooling are evaluated
at layers 11 and 22, with FP32/SDPA and the unchanged 512-token head/tail input.

All four views come from one frozen forward pass per batch. The final view uses
`last_hidden_state` **after** final normalization, while the installed model's
`hidden_states[-1]` is before it. Intermediate hidden states are indexed by their
completed layer count. Padding is excluded from mean pooling; attended special
tokens are included, matching the classifier's pooling contract.

Every vector is finite, L2-normalized, and checksummed in a separate SQLite
cache. Its namespace includes adapter/base identities, extraction code,
tokenization contract, pooling/layers, numerical runtime, device and batch size.
Cold/warm values preserve the same stored bits. Batched commits permit reuse
after interruption. Exported NPZ arrays use `allow_pickle=False`, original row
IDs, text hashes and separate bank/query membership.

The older `.cache/norberto-logistic/` contains unadapted base-model features;
these cannot be relabeled as fine-tuned vectors. It is preserved. The existing
E5 cache is reused, with missing texts encoded locally without downloading.

## Retrieval methods and ablations

`ConfigurableRetriever` supports TF-IDF, E5, NorBERTo, TF-IDF+E5,
TF-IDF+NorBERTo, E5+NorBERTo and all three. Dense vectors use cosine similarity.
Fusions use equal reciprocal ranks (`61 / (60 + rank)`), preserving ties and
the historical hybrid's scale. Raw TF-IDF/E5/NorBERTo scores are not averaged.

The configuration selects the NorBERTo layers/poolings and numerical runtime.
`SelectionPolicy` independently configures relevance weight, missing-class
bonus, per-class cap and candidate count. The fixed experiment compares:

| Policy | Relevance weight | Class bonus | Maximum per class | Candidates |
|---|---:|---:|---:|---:|
| legacy | 0.8 | 0.04 | 2 | 120 |
| no_bonus | 0.8 | 0 | 2 | 120 |
| no_cap | 0.8 | 0.04 | none | 120 |
| no_bonus_or_cap | 0.8 | 0 | none | 120 |
| relevance | 1 | 0 | none | 120 |
| no_redundancy | 1 | 0.04 | 2 | 120 |
| pool40 | 0.8 | 0.04 | 2 | 40 |
| pool240 | 0.8 | 0.04 | 2 | 240 |
| weight50 | 0.5 | 0.04 | 2 | 120 |

The current two-source method is reproduced as the `tfidf_e5`/`legacy` arm.
The complete grid has 171 combinations of methods, four NorBERTo views and
policies, plus label-blind random/length baselines. All keep duplicate/group
guards. Identical queries, lexical cosine >=0.95, character-shingle near
duplicates and repeated training components are excluded from demonstrations.

Outputs include Hit@1, class coverage at three examples, label purity,
document-level label recall, retrieved class frequencies, diversity, cosine
means, candidate ranks and per-class diagnostics. Class Recall@3 means at least
one example of the true class; document-level recall divides by that class's
bank size. These are auxiliary retrieval measures, not GPT macro-F1.

The development-only proposal maximizes balanced Hit@1 and then balanced
purity. The separate audit estimates whether that proposal holds up. The
report also exposes class coverage and diversity tradeoffs; homogeneous
examples and diversified examples are not assumed superior.

## Distinct routing and acceptance decisions

Pre-call features use native base labels, calibrated local/stacking
probabilities, margin, entropy, disagreement and optional original retrieval
evidence. Post-call features may additionally use the actual GPT class and its
agreement with base votes, stacking and retrieved examples. Neither receives
the query's true label as an inference feature.

Research utility models estimate `P(corrected) - P(harmed)` with regularized
logistic regression. These small new policies do not refit the project's base
logistic classifier. Dictionary encoding, scaling, regularization, thresholds
and family selection are fitted inside grouped training folds only. Five
original outer folds assess the entire selector, with three grouped inner
folds selecting parameters. A separate pilot-only selection is replayed on the
confirmation cohort. Rules and family-specific scores are descriptive alongside
the primary joint nested selector.

Upstream stacking OOF predictions remain fixed. A policy-training row's
historical meta-model may have used labels from a policy-evaluation fold;
the added policy itself never fits evaluation groups. Consequently this
conditional replay is not an unbiased fresh end-to-end nested estimate.
Regenerating upstream fits would require a separately authorized study; no
positive family score is used here to claim or promote a better cascade.

All routing comparisons restrict requests to the 236 historical queries with
known GPT outcomes. Keeping unqueried targets as stacking is part of this
restricted replay policy, not an imputed GPT response. New routing outside
that support cannot be validated without new outcomes. Cached answers are
never paired with new NorBERTo examples as if GPT had seen those contexts.

`rule_can_change_prediction` can safely avoid a call if no possible valid GPT
class could pass a chosen post-call rule and change stacking. This mathematical
equivalence reduces required target queries while preserving that rule's
outputs. It does not establish that the rule improves quality. Per-target
savings must be rebatched to estimate actual API-call savings.

## Execute locally, without GPT

Use fresh output directories to preserve prior outputs. The launcher uses the
existing `pln-rocm:6.4.1` image for extraction with networking disabled. GPU
extraction uses the existing RX 6650 XT compatibility settings; change `device`
to `cpu` in a separate configuration to omit GPU devices.

```bash
bash scripts/rag/offline.sh encode \
  --output-dir results/rag-offline-new/features

bash scripts/rag/offline.sh retrieval \
  --embeddings results/rag-offline-new/features \
  --output-dir results/rag-offline-new/retrieval

bash scripts/rag/offline.sh interventions \
  --output-dir results/rag-offline-new/interventions

bash scripts/rag/offline.sh report \
  --retrieval results/rag-offline-new/retrieval \
  --interventions results/rag-offline-new/interventions \
  --output-dir results/rag-offline-new/report
```

`RAG_RESEARCH_CONFIG`, `RAG_RESEARCH_PYTHON`, `RAG_RESEARCH_RUNTIME` and
`NORBERTO_IMAGE` configure the launcher. The local default CPU runtime is
`.cache/cascade-runtime:.cache/rag-runtime`; another compatible Python
environment can use `python -m rag_research` directly. `rag-research-ep2` is
available after installing the project. The CLI exposes no GPT execution flag.

The retriever retains `prepare_targets` and `retrieve(Text, method=..., count=3)`
interfaces. Its examples contain the original text, true training label,
text hash, row/group IDs and similarities; they are compatible with the existing
`rag_cascade.prompt.build_request`. A future context planner can inject the
new retriever in a new study without editing frozen historical packages.

## Screens and limitations

A retrieval proposal warrants a limited future matched GPT trial only when its
development-selected configuration improves audit Hit@1 with a positive lower
paired group-bootstrap bound and does not reduce mean purity versus the old
hybrid. Relevance, class coverage and diversity still require review; an
offline screen is not evidence of improved GPT classification. A learned
intervention selector needs positive held-out macro-F1 evidence before adding
its complexity. Individual best retrospective families do not satisfy that
criterion.

No paid run, final-test command or default promotion follows automatically.
The measured report records which strategies pass or fail these screens.

## Measured evidence and verification

The completed artifacts are in `results/rag-offline-20261106/`:

- `features/`: four checksummed frozen views and encoder/runtime provenance.
- `retrieval/`: 171 variant traces, two controls, bank, metrics and exact
  implementation snapshot; all 236 historical hybrid sequences reproduce.
- `interventions-final/`: primary joint nested policy, family diagnostics,
  pilot-trained confirmation replay and all 288 group-disjoint fit records.
- `report/`: consolidated Markdown and JSON; earlier report/replay
  directories remain preserved.
- `independent-verification.json`: independently recomputed metrics for
  341,156 retrieval rows and nine intervention policies, source hashes,
  unit-vector checks, and fit-group separation.

The final random/length controls have maximum query lexical cosine below 0.95.
An explicit baseline guard was added after measurement without changing any
selected example in this study; the exact measured source remains snapshotted.
Runtime caches and intermediate evidence are local and ignored by Git. Public
aggregates and the Markdown report are versioned. The complete mocked/offline
regression suite passed 229 tests. Existing fixed final-set regression fixtures
were used only to detect compatibility regressions, never to choose research
configurations; the research commands do not load test1/test2.
