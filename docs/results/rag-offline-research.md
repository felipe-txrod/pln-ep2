# Offline RAG and selective-GPT investigation

[Machine-readable evidence](data/rag-offline-research.json)

This study made **zero new GPT calls**, did not use test1/test2 for optimization or configuration selection, and never retrained SVM, LR, NorBERTo, Qwen or stacking. Historical experiment packages and response caches were preserved. New small utility classifiers are research-only policies, not replacements for the trained base LR.

## Verified historical problem

On the 188 confirmation queries, hybrid-3-local corrected 29 stacking errors and harmed 32 correct decisions. Of the harms, 26 had true class c234. Its first retrieved example matched the query label 60/188 times (31.91%). RAG changed 55 GPT decisions versus local predictions alone: 21 exclusive correct decisions versus 15 lost correct decisions.

Mean first-neighbor E5 cosine was 0.9505 despite these mismatches. High semantic similarity therefore did not establish rating agreement in this domain. Topic resemblance and perceived clarity are different signals; the latter explanation remains a hypothesis rather than a proven cause.

| Descriptive validation case | Query row | Gold | Stacking | GPT | First training row / label | E5 cosine |
|---|---:|---|---|---|---|---:|
| harmed_c234 | 2218 | c234 | c234 | c5 | 15171 / c1 | 0.9507 |
| corrected | 1808 | c5 | c234 | c5 | 8942 / c1 | 0.9353 |
| highest_cosine_label_mismatch | 3767 | c5 | c234 | c1 | 3259 / c1 | 0.9964 |

These are retrospective error diagnostics, not development examples for a new prompt. Query/example excerpts, all four native classifier predictions, local-only GPT decisions and the three retrieved labels are included in the aggregate; full original traces remain local. A retrieved example's subjective clarity rating cannot be transferred to a target solely from topic or wording overlap. The original request and attachments are unavailable, so this review cannot establish why the human assessor assigned a particular score.

The highest-cosine mismatch shares a long administrative attachment-forwarding template, yet has different gold labels; its lexical cosine remains below the duplicate exclusion threshold. The harmed c234 example has three base votes for c5 while stacking is correct, demonstrating why majority support alone cannot guarantee a safe override. In the corrected case, RAG changes local-only GPT from c234 to the correct c5 despite the first neighbor being c1. Retrieved label agreement is therefore insufficient to explain individual successes or failures causally.

| Confirmation true class (analysis only) | Corrected | Harmed | Net correct change |
|---|---:|---:|---:|
| c1 | 7 | 5 | +2 |
| c234 | 5 | 26 | -21 |
| c5 | 17 | 1 | +16 |

The concentration of harm in c234 motivates diagnostic protection rules, but does not validate them. The inference-time protection rule uses the **predicted stacking class**, never the target's true class.

## Frozen NorBERTo extraction

The checkpoint is ModernBERT with 22 layers, hidden size 768 and LoRA rank 4 on Wqkv. The adapter is the unchanged EP1 artifact (`d693f6e2…`). The classifier was trained with masked mean pooling. We extracted mean and first-token representations at layers 11 and 22, in the same frozen FP32/SDPA forward pass, with the original cased 512-token head/tail preprocessing and L2 normalization.

The tokenizer actually inserts [CLS] ID 3 and [SEP] ID 4. CLS features use the actual first token. The installed Transformers model exposes its last `hidden_states` entry before `final_norm`; the last-layer view uses `last_hidden_state` after that normalization, matching the classifier input. Intermediate layer 11 uses its own pre-final-normalization state.

Feature exports retain 11,562 eligible training examples and 1,972 validation rows. Missing texts are encoded once for all requested views, and cached vectors are reused. Existing E5 vectors are also reused. The older NorBERTo-logistic cache contains **unadapted base-model** vectors and cannot represent this adapter; it was preserved rather than mislabeled or overwritten.

## Retrieval comparison

There are 171 predeclared method/view/policy variants, plus random and length controls. Development uses original OOF folds 0/1/2 (1143 rows), and audit uses folds 3/4 (829 rows), with similarity groups disjoint. The proposal selected **only on development** is `tfidf_e5_norberto--layer11_mean--relevance`. The seven-method table fixes the NorBERTo view selected from NorBERTo-only development performance: `layer22_mean`. It holds the legacy MMR policy fixed.

Hit@1 is first-neighbor class agreement. Class Recall@3 means at least one of the three examples has the true class; purity@3 is the fraction matching it. These use gold only after retrieval. Document-level label recall is also exported with the class-bank-size denominator. None is a measure of GPT macro-F1.

| Method | Hit@1 | Class Recall@3 | Purity@3 | Unique classes |
|---|---:|---:|---:|---:|
| tfidf | 37.83% | 92.09% | 35.46% | 2.688 |
| e5 | 38.29% | 100.00% | 33.33% | 3.000 |
| norberto | 38.64% | 100.00% | 33.33% | 3.000 |
| tfidf_e5 | 38.79% | 83.67% | 35.73% | 2.423 |
| tfidf_norberto | 38.13% | 85.34% | 36.36% | 2.419 |
| e5_norberto | 38.79% | 85.85% | 36.61% | 2.425 |
| tfidf_e5_norberto | 38.59% | 86.00% | 36.66% | 2.422 |
| random | 32.76% | 71.15% | 33.32% | 2.119 |
| length | 34.63% | 71.15% | 34.47% | 2.086 |

These rows describe all 1,972 validation queries. Configuration choice uses development only; audit values for the chosen proposal are shown separately below.

| Selected proposal split | Balanced Hit@1 | Balanced purity@3 | Class Recall@3 |
|---|---:|---:|---:|
| development | 40.37% | 37.71% | 72.97% |
| audit | 37.97% | 37.31% | 71.53% |

Against the current TF-IDF+E5/legacy policy, the proposal's audit Hit@1 difference is -1.21%; descriptive group-bootstrap 95% interval [-4.41%, +1.99%]. Purity@3 changes by +1.93%. This interval describes retrieval label matching, not GPT improvement.

## Pooling and layer ablation

| NorBERTo view / policy | Development balanced Hit@1 | Audit balanced Hit@1 | Audit purity@3 |
|---|---:|---:|---:|
| layer11_mean/legacy | 37.01% | 35.85% | 33.33% |
| layer11_mean/relevance | 37.01% | 35.85% | 35.67% |
| layer11_cls/legacy | 34.31% | 38.86% | 33.33% |
| layer11_cls/relevance | 34.31% | 38.86% | 37.88% |
| layer22_mean/legacy | 38.95% | 38.28% | 33.33% |
| layer22_mean/relevance | 38.95% | 38.28% | 37.07% |
| layer22_cls/legacy | 36.87% | 41.40% | 33.33% |
| layer22_cls/relevance | 36.87% | 41.40% | 39.24% |

| Encoder view | Random-pair cosine mean | Standard deviation | Mean-vector norm |
|---|---:|---:|---:|
| layer11_mean | 0.8068 | 0.5289 | 0.8812 |
| layer11_cls | 0.9248 | 0.0447 | 0.9619 |
| layer22_mean | 0.9534 | 0.0278 | 0.9761 |
| layer22_cls | 0.7960 | 0.1051 | 0.8936 |
| e5 | 0.8849 | 0.0212 | 0.9411 |

The four frozen views are not assumed to improve retrieval. High random-pair cosine and mean-vector norm indicate anisotropy: even unrelated training pairs can have high cosine. The wide layer11-mean spread further cautions against transferring an absolute cosine threshold between views. E5 uses its original query prefix and head truncation; NorBERTo uses its original head/tail policy, so cross-encoder differences include tokenization and truncation differences. Selecting layer22-CLS because its audit number looks higher would violate the development-only selection rule.

## MMR and diversity ablation

Legacy MMR has relevance weight 0.8, unseen-class bonus 0.04, maximum two examples per class and 120 candidates. All variants retain exact/near-duplicate exclusion, unique similarity groups and train-only examples. Fusion uses equal reciprocal ranks, not raw mixed score scales.

The class bonus is identical for every candidate at the first selection, with no redundancy yet. It cannot explain the weak first neighbor by itself. The bonus/cap mainly affect the second and third examples. Candidate rank and cosine shifts show when diversity replaces a closer neighbor.

| Current TF-IDF+E5 policy | Hit@1 | Recall@3 | Purity@3 | Unique classes | Mean selected rank |
|---|---:|---:|---:|---:|---:|
| legacy | 38.79% | 83.67% | 35.73% | 2.423 | 2.62 |
| no_bonus | 38.79% | 75.15% | 36.19% | 2.158 | 2.44 |
| no_cap | 38.79% | 82.00% | 36.33% | 2.359 | 2.51 |
| no_bonus_or_cap | 38.79% | 70.23% | 37.58% | 1.974 | 2.27 |
| relevance | 38.79% | 70.33% | 37.64% | 1.975 | 2.24 |
| no_redundancy | 38.79% | 81.80% | 36.02% | 2.367 | 2.54 |
| pool40 | 38.79% | 83.67% | 35.73% | 2.423 | 2.62 |
| pool240 | 38.79% | 83.67% | 35.73% | 2.423 | 2.62 |
| weight50 | 38.79% | 89.00% | 35.58% | 2.578 | 2.87 |

| Dense retriever / policy | Recall@3 | Purity@3 | Unique classes | Mean selected rank |
|---|---:|---:|---:|---:|
| e5--none--legacy | 100.00% | 33.33% | 3.000 | 5.70 |
| e5--none--no_bonus | 77.84% | 37.10% | 2.174 | 3.62 |
| e5--none--no_cap | 100.00% | 33.33% | 3.000 | 5.70 |
| e5--none--no_bonus_or_cap | 72.52% | 38.07% | 2.001 | 3.43 |
| e5--none--relevance | 72.26% | 37.68% | 1.984 | 2.21 |
| e5--none--no_redundancy | 99.95% | 33.37% | 2.998 | 3.81 |
| norberto--layer22_mean--legacy | 100.00% | 33.33% | 3.000 | 5.44 |
| norberto--layer22_mean--no_bonus | 78.25% | 36.78% | 2.176 | 3.56 |
| norberto--layer22_mean--no_cap | 100.00% | 33.33% | 3.000 | 5.44 |
| norberto--layer22_mean--no_bonus_or_cap | 73.83% | 37.54% | 2.030 | 3.45 |
| norberto--layer22_mean--relevance | 72.16% | 37.34% | 1.995 | 2.20 |
| norberto--layer22_mean--no_redundancy | 100.00% | 33.33% | 2.999 | 3.88 |

For the E5 and selected NorBERTo-only legacy arms, every query retrieves all three classes. This makes Recall@3 exactly 100% and purity exactly one third; it is a consequence of forced diversity, not successful identification of the target class. Removing the cap alone does not fix it; removing the class bonus does. Dense cosine scores are tightly concentrated, so a fixed 0.04 bonus can dominate small relevance differences. Reciprocal-rank fusion changes the score spread, explaining why the same policy behaves differently across dense and fused arms.

For the current hybrid, removing both bonus and cap raises purity while lowering class coverage; expanding the pool from 40 to 240 produces no aggregate benefit here. Removing redundancy alone does not solve the mismatch. Neither homogeneous nor class-diverse demonstrations are presumed better for GPT. These are retrieval tradeoffs, not confirmed GPT gains.

## Per-class retrieval diagnostics

| Method | True class | Hit@1 | Class Recall@3 | Purity@3 |
|---|---|---:|---:|---:|
| tfidf | c1 | 35.69% | 89.23% | 34.14% |
| tfidf | c234 | 33.80% | 91.37% | 33.76% |
| tfidf | c5 | 44.32% | 95.65% | 38.62% |
| e5 | c1 | 36.33% | 100.00% | 33.33% |
| e5 | c234 | 34.94% | 100.00% | 33.33% |
| e5 | c5 | 43.86% | 100.00% | 33.33% |
| norberto | c1 | 37.46% | 100.00% | 33.33% |
| norberto | c234 | 33.95% | 100.00% | 33.33% |
| norberto | c5 | 44.95% | 100.00% | 33.33% |
| tfidf_e5 | c1 | 36.50% | 77.49% | 32.26% |
| tfidf_e5 | c234 | 35.50% | 84.87% | 34.75% |
| tfidf_e5 | c5 | 44.63% | 88.34% | 40.18% |
| tfidf_norberto | c1 | 36.33% | 82.64% | 34.24% |
| tfidf_norberto | c234 | 34.23% | 85.15% | 35.12% |
| tfidf_norberto | c5 | 44.17% | 88.18% | 39.76% |
| e5_norberto | c1 | 41.64% | 82.96% | 35.74% |
| e5_norberto | c234 | 29.99% | 86.70% | 35.12% |
| e5_norberto | c5 | 45.72% | 87.71% | 39.09% |
| tfidf_e5_norberto | c1 | 38.59% | 82.48% | 34.62% |
| tfidf_e5_norberto | c234 | 33.10% | 86.99% | 35.64% |
| tfidf_e5_norberto | c5 | 44.63% | 88.34% | 39.76% |
| random | c1 | 29.58% | 64.95% | 29.05% |
| random | c234 | 29.70% | 69.45% | 32.30% |
| random | c5 | 39.19% | 79.00% | 38.57% |
| length | c1 | 21.06% | 72.51% | 30.55% |
| length | c234 | 47.10% | 77.79% | 41.02% |
| length | c5 | 34.06% | 62.52% | 31.05% |

| Method | Retrieved c1 | Retrieved c234 | Retrieved c5 | Mean lexical cosine | Mean E5 cosine | Mean NorBERTo cosine |
|---|---:|---:|---:|---:|---:|---:|
| tfidf | 31.66% | 32.74% | 35.60% | 0.4443 | n/a | n/a |
| e5 | 33.33% | 33.33% | 33.33% | 0.3399 | 0.9456 | n/a |
| norberto | 33.33% | 33.33% | 33.33% | 0.3066 | n/a | 0.9861 |
| tfidf_e5 | 29.77% | 34.09% | 36.14% | 0.4213 | 0.9450 | n/a |
| tfidf_norberto | 30.51% | 35.18% | 34.31% | 0.3948 | n/a | 0.9849 |
| e5_norberto | 30.93% | 34.48% | 34.58% | 0.3527 | 0.9428 | 0.9853 |
| tfidf_e5_norberto | 30.59% | 35.14% | 34.26% | 0.4019 | 0.9433 | 0.9831 |
| random | 29.43% | 32.74% | 37.83% | 0.0314 | n/a | n/a |
| length | 30.71% | 39.69% | 29.60% | 0.0381 | n/a | n/a |

Per-class similarity results are preserved in the machine-readable aggregate and full local retrieval report. Similarities are reported in their own encoder space; they are not comparable confidence probabilities.

## Routing versus acceptance

Pre-call models see calibrated stacking/base probabilities, native class predictions, margin, entropy, disagreement and optionally the original retrieved-example statistics. Post-call models can additionally see the returned GPT class and its agreement with stacking, base votes and examples. Query gold never enters either feature function. New NorBERTo retrieval contexts are **not** paired with old GPT answers as if the API had seen them.

The utility model estimates P(corrected)-P(harmed). It is a small regularized logistic model fitted inside each training fold. Five original group-disjoint outer folds assess the added policy conditionally on fixed historical upstream predictions; three inner group folds choose family, regularization, threshold and routing budget. The joint selector includes the family search, avoiding selection of a family on the same OOF scores used to claim its gain.

This is not a fresh end-to-end nested evaluation: a policy-training row's historical OOF stacking predictor may have been fitted using labels from a policy-evaluation fold. Own-group meta predictions remain OOF, and all newly fitted policy transforms/models exclude evaluation groups, but the upstream OOF matrix is frozen rather than regenerated for each policy split. Historical Qwen exposure and prior router development also remain. These restrictions prevent claiming an unbiased new cascade generalization estimate from this replay, even for a positive family score.

Only 236 historical queries have observed replies. Routing replay is restricted to this support; the other 1736 targets keep stacking because the replay policy never queries them. Effects of a new router outside this support cannot be identified without new GPT outcomes. Acceptance-only references use all logged calls (11.97% of this retrospective population), not a deployable 10% budget guarantee. Routed policies have per-fold 5/10% caps and include zero routing.

| Grouped OOF policy | Correct /1972 | Accuracy | Macro-F1 | Corrected | Harmed | Requested targets in replay |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 936 | 47.46% | 47.30% | 0 | 0 | 0 |
| unconditional_logged | 936 | 47.46% | 47.20% | 37 | 37 | 236 |
| rules | 930 | 47.16% | 46.97% | 10 | 16 | 236 |
| accept_local | 933 | 47.31% | 47.12% | 15 | 18 | 236 |
| route_local | 938 | 47.57% | 47.32% | 19 | 17 | 116 |
| accept_retrieval | 932 | 47.26% | 47.06% | 12 | 16 | 192 |
| route_retrieval | 937 | 47.52% | 47.30% | 15 | 14 | 89 |
| cascade | 936 | 47.46% | 47.26% | 9 | 9 | 72 |
| joint_nested_selection | 935 | 47.41% | 47.19% | 16 | 17 | 163 |

The joint selector's descriptive 95% group-bootstrap interval for macro-F1 difference is [-0.71%, +0.45%]. It does not establish a benefit. Individual family improvements are exploratory; retrospective hypothesis generation on these historical results is an additional source of optimism.

## Fixed-rule controls and safe pre-call skips

| Fixed rule | Correct | Macro-F1 | Corrected | Harmed | Equivalent requested targets | Targets safely skipped |
|---|---:|---:|---:|---:|---:|---:|
| protect_c234 | 941 | 47.50% | 11 | 6 | 71 | 165 |
| two_base_votes | 935 | 47.17% | 25 | 26 | 188 | 48 |
| three_base_votes | 937 | 47.33% | 5 | 4 | 28 | 208 |
| two_examples | 937 | 47.30% | 17 | 16 | 89 | 147 |
| votes_and_example | 936 | 47.22% | 24 | 24 | 162 | 74 |
| protect_and_votes | 940 | 47.48% | 5 | 1 | 45 | 191 |

Fixed-rule results are descriptive and were not selected on a fresh test set. A pre-call skip is safe relative to its corresponding post-call rule when **none of the three valid GPT labels can both pass the rule and change stacking**. Exhaustive synthetic tests and the logged replay verify prediction equivalence. Counts are target instances, not API batches; a future plan must rebatch them to measure call savings. This does not prove that the rule itself beats stacking.

| Same-budget uncertainty reference | Correct | Macro-F1 | Requested targets |
|---|---:|---:|---:|
| uncertainty_matched_to_route_local | 930 | 46.96% | 116 |
| uncertainty_matched_to_route_retrieval | 932 | 47.09% | 89 |

A separate chronological check chose the policy only on 48 pilot queries and evaluated the remaining 1884 confirmation rows. It selected `two_examples`: 907 correct, macro-F1 47.88%, versus stacking 910 and 48.08%.

## Screening for any later paid evaluation

Development-selected retrieval proposal: `tfidf_e5_norberto--layer11_mean--relevance`. Offline screening result: **does not pass the predeclared paid-trial screen**. Screening requires a positive lower audit Hit@1 difference interval and no mean purity decline versus the current hybrid. Class coverage/diversity/relevance remain tradeoffs requiring review. No paid call is authorized or executed by this report.

Learned intervention positive-lower-bound screen: **not passed**. Do not promote a rule because it looks best retrospectively, or add more paid rounds to rescue an unconfirmed result. No production default is changed by this report. If a retrieval candidate passes the auxiliary screen, a small future matched trial must freeze prompts/routing first, retain a text+local control, and compare pure relevance with constrained MMR before claiming GPT quality improvement.

## Decisions from the measured evidence

- Retain the historical default unless a separately authorized validation establishes a better cascade. Fixed-upstream replay cannot promote a new default.
- Keep NorBERTo retrieval as an optional offline research component. The development-selected proposal fails the paid-trial screen and does not earn promotion.
- Candidate-pool expansion and learned acceptance need demonstrated benefit before adding their complexity to the active pipeline.
- Keep the bonus/cap/redundancy switches and safe pre-call equivalence checks for reproducible diagnostics. Diversity's measurable effect does not establish a better GPT prompt.
- No new retrieval configuration currently justifies a paid trial under the recorded evidence criteria. A later authorized study would need fresh grouped validation, frozen routing/prompts, a local-only control and explicit cost limits. Do not pick a new winner by reusing this audit.

## Reproducibility and boundaries

Legacy retrieval trace reproduction: 236/236 exact example-ID sequences. The raw output contains 171 complete trace files, baseline traces, a source snapshot and hashes; model weights, cache namespaces, IDs and folds are pinned. All embedding arrays are finite/unit-normalized and caches are checksummed. Historical checkpoints, experiments and GPT responses were not overwritten.

Research token usage, new GPT cost and remote latency are all zero. Local computation time is exported separately; replayed requested-target counts do not represent new calls or a measured reduction in billed API batches.

No test1/test2 optimization or final prediction was performed. Nested meta-model OOF prevents own-group meta fitting, subject to the fixed-upstream limitation above. No original base classifier or stacking was refitted. Future GPT evaluation must use newly valid cache keys for changed contexts; old replies cannot demonstrate improvements from new demonstrations.

## Independent verification and reproduction

The complete mocked/offline regression suite passed **229 tests**. An independent
scorer recomputed the three main retrieval metrics for all 173 methods/controls
(341,156 query records), verified training labels and unique groups, checked all
four finite unit-vector arrays, confirmed 288 policy fits exclude evaluation
groups, and reproduced all nine intervention metric sets. Existing fixed final-set
regression fixtures detect compatibility regressions only; they did not select
any research configuration. Historical classifier/cascade sources and frozen
artifacts have no changes. The explicit random/length lexical guard added after
measurement excludes none of the examples selected in this study; the exact
measured source remains preserved with its protocol hashes.

Local evidence is under `results/rag-offline-20261106/`: `features/`, `retrieval/`,
`interventions-final/`, `report/`, and `independent-verification.json`. Previous
report/replay directories are retained. See the
[offline research guide](../guides/rag-offline-research.md) for commands, runtime
requirements and data boundaries. Raw texts, caches, model features and 579 MB
of intermediate evidence remain local; the compact aggregate is versioned.
