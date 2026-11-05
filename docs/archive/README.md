# Experiment archive

[Documentation index](../README.md)

Historical records are retained for traceability. Use the
[current guides](../README.md#guides) for supported workflows and the
[results index](../results/README.md) for current comparisons.

## GPT and Codex studies

| Record | Content |
|---|---|
| [Usage and earlier workflows](gpt/usage.md) | Original installation, remote backends, development, exports, and statistical comparison |
| [Decisions and requirements](gpt/decisions.md) | Initial GPT protocol and data analysis |
| [First result](gpt/first-result.md) | Error analysis, costs, and initial findings |
| [Hybrid retrieval](gpt/hybrid-retrieval.md) | Retrieval method, pilot, and 300-row comparison |
| [Verification history](gpt/verification.md) | Earlier implementation checks and experiment milestones |

## NorBERTo training investigations

| Record | Content |
|---|---|
| [Legacy full-corpus test1 evaluation](norberto/legacy-full-test1.md) | Original accumulation behavior, 20,092 training rows |
| [Recipe-2 full-corpus test1 evaluation](norberto/recipe2-full-test1.md) | Corrected mean loss before optimizer-unit compatibility |
| [Recipe-2 development holdout evaluation](norberto/recipe2-development-holdout.md) | Instability reproduced on the frozen development partition |
| [Regression investigation](norberto/regression-investigation.md) | Data, tokenizer, training arguments, and loss-normalization checks |
| [Optimizer and backend audit](norberto/instability-audit.md) | CPU/GPU gradients, finite differences, and first-attention localization |

The [recipe-3 correction](../guides/norberto-optimizer-units.md) and its
[holdout evaluation](../results/norberto-recipe3-holdout.md) describe the
subsequent optimizer-compatible implementation. Public aggregates for both
current and archived experiments remain in [`results/data/`](../results/data/).
