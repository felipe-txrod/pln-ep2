# NorBERTo training

[Documentation index](../README.md)

Use recipe 3 for new adapters. It preserves the original EP1 optimizer units
for clipping and AdamW while normalizing example weights across accumulated
microbatches. The [optimizer guide](norberto-optimizer-units.md) documents the
implementation and verification; older recipes are in the
[experiment archive](../archive/README.md#norberto-training-investigations).

## Environment

Run commands from the EP2 root. The launcher requires the sibling EP1 sources
and Hugging Face cache, and the validated local image `pln-rocm:6.4.1`.
`NORBERTO_EP1_ROOT` and `NORBERTO_IMAGE` select another local checkout or image.
The container is offline, mounts EP1/cache read-only, and uses
`HSA_OVERRIDE_GFX_VERSION=10.3.0` on the RX 6650 XT.

## Train on 100% of the training data

```bash
NORBERTO_IMAGE=pln-rocm:6.4.1 \
bash scripts/train-norberto-full.sh \
  --partition full \
  --output-dir /workspace/results/norberto-full-recipe3
```

This initializes a fresh adapter from the pinned public base and uses all
**20,092 EP1 training responses**, including the former holdout. It trains for
two fixed epochs with no validation or checkpoint selection. The former
holdout is training data for this adapter and cannot serve as unseen validation.
Historical full-corpus runs took approximately 39–40 minutes on the recorded
RX 6650 XT environment; duration is not a completion guarantee.

## Train on the frozen development partition

```bash
NORBERTO_IMAGE=pln-rocm:6.4.1 \
bash scripts/train-norberto-full.sh \
  --partition development \
  --output-dir /workspace/results/norberto-development-recipe3-new
```

This uses the original **16,093 development rows** and reserves **3,999 holdout
rows**, with no random resplitting. The pinned split checksum, row IDs, counts,
class coverage, and zero row/group overlap are checked before tokenization.
The default split is
`EP1/results/research/dataset-hypotheses-20260923/splits.csv`; `--holdout-splits`
can select another copy with the same checksum.

The completed recipe-3 development run took 31.32 minutes. Its quality is
reported separately in the [results section](../results/README.md), where
holdout validation and test1 comparison have distinct tables.

## Validate the plan before training

Append `--dry-run` to either command. A dry-run checks data and parameters
without importing model libraries, creating the output directory, or training:

```bash
bash scripts/train-norberto-full.sh \
  --partition full \
  --output-dir /workspace/results/norberto-full-recipe3-preview \
  --dry-run
```

Every execution requires a new output directory. An existing directory is
rejected; change `--output-dir` rather than overwriting earlier artifacts.

## Fixed parameters and provenance

| Parameter | Value |
|---|---|
| Base | `Itau-Unibanco/NorBERTo-base` |
| Public revision | `db73446f89c96044863ea05a39f680524b84bccb` |
| Epochs | 2 |
| LoRA | Rank 4, alpha 8, dropout 0.05, `Wqkv` in all 22 layers |
| Context | 512 tokens, head/tail truncation, case preserved |
| Precision / attention | FP32 / SDPA |
| Batch / accumulation | 4 / 4 |
| Learning rate | 0.001 |
| Weight decay / warmup ratio | 0.01 / 0.1 |
| Seed / CPU threads | 42 / 2 |
| Recipe | Version 3, explicit optimizer gradient scale 4 |

The wrapper reuses EP1's training function with the effective configuration in
`models/norberto/config.json` and the recipe-3 Trainer extension. It records
source hashes, dataset/split hashes, optimizer units, selected training rows,
completion state, and exported adapter checksums. Reported loss is an example
mean. Logged gradient norms use EP1 optimizer units; divide by 4 when comparing
with recipe-2 sample-unit norms.

## Outputs and evaluation

Each selected output directory contains:

| Artifact | Content |
|---|---|
| `final_model/` | New LoRA adapter, classification head, and tokenizer |
| `run.json`, `training.json`, `config.json` | Status, recipe, parameters, metrics, and hashes |
| `training-rows.jsonl` | Training row IDs, labels, and text hashes |
| `checkpoints/` | Intermediate training checkpoints |

A new adapter is distinct from the frozen EP1 model and is not automatically
installed in its inference package or selected by the ensemble.

Use the [two-stage test1 workflow](norberto-test1.md) to predict and then score
saved predictions. For a completed development-only run, the separate
[`evaluate-norberto-holdout.py`](../../scripts/evaluate-norberto-holdout.py)
script can validate the frozen holdout. Holdout results and the recorded
training audits are in the [recipe-3 report](../results/norberto-recipe3-holdout.md).
