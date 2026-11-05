# Stacking with selective GPT decisions

This package freezes the OOF-selected routing rule, GPT prompt mode, model,
reasoning effort, candidate budgets, provenance and source/runtime fingerprints.
It references the unchanged `models/ensemble-stacking-qwen/` model. GPT weights
are remote; validated responses remain in the ignored persistent SQLite cache.

**Validation selected zero GPT budget.** Nested grouped OOF macro-F1 was
47.2977% for stacking, versus 47.2762%, 46.8407% and 47.1016% for 5%, 10% and
20% routing. The local-prediction prompt was better than text-only in the initial
10% ablation, but still below stacking. This package does not assume GPT helps.

The frozen nonzero configurations were then evaluated on all 900 test1 rows:

| GPT budget | Correct | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| 0% | 440 | 48.8889% | 48.8034% |
| 5% | 435 | 48.3333% | 48.1824% |
| 10% | 434 | 48.2222% | 48.0477% |
| 20% | 436 | 48.4444% | 48.1761% |

Production `cascade_classifier predict` uses the selected zero budget, retaining
the stacking output without GPT calls. Explicit `--compare-budgets` reproduces
the frozen 0/5/10/20% experimental comparison with a maximum 20% union of targets.
All selection precedes final gold access; no final result changed this package.

The remote classifier requests `gpt-6.1-sol`, reasoning `low`, through isolated
Codex sessions and strict structured output. The CLI exposes no temperature
control. Item-level cache, bounded retries, a circuit breaker and explicit
stacking fallback are implemented in the separate `cascade_classifier` package.

`manifest.json` pins `model.json`, `validation.json`, `prompt-ablation.json`
and `preparation.json`. The latter records original source/partition identities
and nested 5x5 OOF validation provenance: 16,093 base-training rows, 2,027
calibration rows, and 1,972 meta/routing rows. Qwen's checkpoint selection
already consulted the full holdout, so inherited exposure remains disclosed.

See the [commands](../../docs/guides/stacking-gpt-cascade.md),
[measured experiment](../../docs/results/stacking-gpt-cascade.md), and
[verified aggregate](../../docs/results/data/stacking-gpt-cascade.json).
