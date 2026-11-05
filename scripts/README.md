# Script entry points

[Project documentation](../docs/README.md)

Run commands from the EP2 root. Scripts are grouped by model or purpose.
Dataset, model, cache, and result locations remain relative to the project root.

## NorBERTo

| Entry point | Purpose |
|---|---|
| `norberto/train.sh` | Train using the pinned full-corpus or development recipe |
| `norberto/test1.sh predict` | Predict with a selected completed adapter on GPU |
| `norberto/test1.sh compare` | Score saved predictions against test1 labels on CPU |
| `norberto/evaluate-test1.py` | Python implementation of the test1 stages |
| `norberto/evaluate-holdout.py` | Validate a completed development-only adapter |

```bash
bash scripts/norberto/train.sh --partition full \
  --output-dir /workspace/results/norberto-full-recipe3-new

bash scripts/norberto/test1.sh predict \
  --training-run results/norberto-development-recipe3 \
  --output-dir results/norberto-test1-new

bash scripts/norberto/test1.sh compare \
  --prediction-dir results/norberto-test1-new \
  --output-dir results/norberto-test1-metrics-new
```

See the [training guide](../docs/guides/norberto-training.md) and
[test1 guide](../docs/guides/norberto-test1.md) for environment and data details.

## SVM

Evaluate a newly trained EP1 artifact without replacing the frozen model:

```bash
bash scripts/svm/test1.sh \
  --artifact results/svm-full/model.joblib \
  --input data/test1.xlsx \
  --gold data/test1-rotulado-para-teste-estatistico.xlsx \
  --output-dir results/svm-full-test1
```

This offline CPU-only command validates the saved recipe and adjacent metadata,
persists text-only predictions, then reads gold labels to produce `metrics.json`.
The output includes accuracy, macro-F1, precision, recall, and the confusion
matrix. Use a new output directory. `SVM_IMAGE` overrides the existing image.

## Qwen

| Entry point | Purpose |
|---|---|
| `qwen/container.sh` | Run the Qwen CLI in the configured ROCm image |
| `qwen/evaluate-test1.py` | Evaluate an exported trained adapter on test1 |

[Qwen workflow and compatibility](../docs/guides/qwen-finetuning.md).

## Model artifacts

| Entry point | Purpose |
|---|---|
| `models/prepare-base.py` | Restore or verify the pinned public NorBERTo base |
| `models/freeze.py` | Export existing inference artifacts without training |

```bash
python3 scripts/models/prepare-base.py --verify-only
python3 scripts/models/freeze.py --help
```

[Artifact formats and restoration](../models/README.md).

## Diagnostics and experiments

`diagnostics/audit-dataset-overlap.py` compares the seven provided EP1/EP2
workbooks on CPU, without model loading. It reports file/content identity,
exact and normalized text overlap, and conflicting labels. See the
[dataset audit](../docs/results/dataset-overlap.md) for results and reproduction.

`diagnostics/norberto/` contains optimizer, backend, sensitivity, and training
regression audits. `probe-training-prefix.py` runs bounded real training when
explicitly invoked; the other probes may perform forward/backward passes.
These are diagnostic tools rather than the normal training workflow.

`experiments/gpt/pilot-efficiency.py` contains the earlier paired GPT/Codex
pilot. It prepares a dry run by default; `--execute` enables remote requests.
See the [historical GPT study](../docs/archive/gpt/first-result.md).

## Compatibility commands

Three commands remain at the scripts root:

- `train-norberto-full.py` and `train-norberto-full.sh` retain the original
  implementation and exact bytes. Existing jobs and training manifests continue
  to reference them; `norberto/train.sh` delegates to this launcher.
- `prepare-model-base.py` forwards to `models/prepare-base.py` so restoration
  instructions emitted by frozen model packages remain valid.

The reorganized entry points do not change model recipes, output paths,
existing adapters, or training state. Choose a fresh destination for every run.
