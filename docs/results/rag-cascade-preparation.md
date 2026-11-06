# RAG context study: prepared, execution deferred

[Commands and protocol](../guides/rag-stacking.md) · [Configuration](../../configs/rag-cascade.json)

The user explicitly requested preparing the implementation without running the
experiment because their token allowance was running low. This change therefore
contains code, configuration, documentation and offline synthetic tests only.

| Activity | Status |
|---|---|
| Train-only TF-IDF, semantic E5 and hybrid retrieval | Implemented; real-data execution deferred |
| Three/five-example comparison and four context ablations | Prepared; no new measured quality results |
| New GPT classifier calls | **0** |
| New real embeddings or model downloads | **0** |
| New base-model fitting or real-data inference | **0** |
| New production test2 workbooks or test1 evaluations | **0** |
| Existing experiments and frozen models | Preserved |

The prepared study excludes all holdout rows from the RAG bank. It reuses
checksum-pinned nested OOF meta-model predictions, selects retrieval in a small
pilot and compares the four contexts in separate similarity groups. The bank,
prompts and selection never use test gold. Future test1 inference and evaluation
are blocked until the same frozen configuration has a completed, unchanged test2
run and workbook.

Runtime controls retain persistent response/item caching, explicit attempt caps,
bounded retries, circuit breaking and stacking fallback. Classification requires
`--execute`; encoder downloads require `--download`. Defaults only preview calls.

The complete offline suite passed **215 tests in 58.60 seconds**. A subsequent
empty-route regression and final checks passed **15 RAG tests**, with fake
embeddings and a fake backend. Containers had networking disabled and no GPU
devices attached. These tests verify implementation behavior, not RAG accuracy.

ONNX Runtime and the E5 weights still need preparation when an actual study is
authorized. No new RAG artifact has been selected or published, and the previous
zero-GPT cascade choice remains unchanged. Any claim of improvement must wait
for the explicitly deferred experiment.
