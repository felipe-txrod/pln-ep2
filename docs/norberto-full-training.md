# NorBERTo refitting on the full EP1 training corpus

## Controlled development-only comparison

To train with the same corrected loss and numerical parameters on the original
EP1 development partition, excluding its frozen holdout:

```bash
NORBERTO_IMAGE=pln-rocm:6.4.1 \
bash scripts/train-norberto-full.sh \
  --partition development \
  --output-dir /workspace/results/norberto-development-normalized
```

This uses exactly **16,093 development rows** and reserves **3,999 holdout rows**.
It preserves the original split, rather than generating a random 80/20 partition.
The split CSV SHA-256, row IDs, label coverage, group separation, and counts are
verified before tokenization. The split defaults to
`EP1/results/research/dataset-hypotheses-20260923/splits.csv`; another copy of the
same pinned CSV can be selected with `--holdout-splits`.

The output is a new adapter, initialized from the pinned base and trained for
two fixed epochs. No holdout text is tokenized or optimized, and the holdout is
not used for checkpoint selection or automatic evaluation. A later holdout
evaluation is a separate operation. The historical holdout has been consulted
previously, so this does not make it historically untouched.

Append `--dry-run` to validate the plan without importing model libraries,
creating outputs, or starting training. The development dry-run verified zero
row/group overlap and class counts 5,090/5,436/5,567 in `c1/c234/c5` order.
Nine partition tests passed. The default `--partition full` still selects all
20,092 rows and was checked separately by dry-run.

The completed development-only run took **31.94 minutes** and achieved
**35.66% accuracy (1,426/3,999)** and **35.59% macro-F1** on the frozen holdout,
compared with **45.31% accuracy** for the original EP1 adapter on the same rows.
See the [holdout validation](avaliacao-norberto-development-holdout.md) for
partition checks, class metrics, and the observed gradient instability.

## Recipe version 2: corrected loss normalization

The current command normalizes cross entropy by the total example count across
gradient accumulation, fixing a confirmed legacy ModernBERT/Trainer bug.
Optimizer hyperparameters remain unchanged, but this is a new loss computation,
recorded as recipe version 2. The historical 341/900 refit used the legacy loss.
Nine CPU tests and a dry-run passed. The completed version-2 run achieved
**408/900 (45.33%)** in its isolated
[test1 comparison](avaliacao-norberto-normalized-test1.md). See the
[investigation](investigacao-norberto-full.md) for initial evidence and causal limits.

The historical output directory already exists. Inspect a new version-2 plan
without starting training:

```bash
bash scripts/train-norberto-full.sh \
  --output-dir /workspace/results/norberto-full-normalized-new --dry-run
```

## Training command and original parameters

From the EP2 root, start a fresh adapter using all 20,092 original EP1 training
responses, including the 3,999 rows previously reserved as the internal holdout:

```bash
bash scripts/train-norberto-full.sh
```

To inspect the data and parameters without loading a model or starting training:

```bash
bash scripts/train-norberto-full.sh --dry-run
```

The wrapper requires the sibling EP1 repository and its existing Hugging Face
cache. `NORBERTO_EP1_ROOT` can select another local EP1 checkout. The default
Docker image is `pln-rocm:6.4.1`; `NORBERTO_IMAGE` overrides it. The container
has no network access and mounts EP1 and its model cache read-only. ROCm uses
`HSA_OVERRIDE_GFX_VERSION=10.3.0` on the validated RX 6650 XT environment.

The implementation reuses the original EP1 `clarity.finetune.train_one` function
and the effective configuration in `models/norberto/config.json`: two epochs,
LoRA rank 4/alpha 8/dropout 0.05, `Wqkv`, context 512, head/tail truncation,
FP32, SDPA, learning rate 0.001, batch 4, accumulation 4, weight decay 0.01,
warmup ratio 0.1, and seed 42. It initializes from the pinned public base,
without loading the original trained adapter.

No partition filtering, checkpoint selection, validation, or test workbook is
used. The original configuration retains historical split descriptors; the
wrapper never invokes the development-only `final` orchestration or a splitter.
The dataset hash, original effective configuration, and exact 20,092-row coverage
are checked before training. An existing output directory is rejected.

Outputs are stored under `results/norberto-full-training/`:

- `final_model/`: the new adapter, trained classification head, and tokenizer.
- `run.json`, `training.json`, and `config.json`: status, parameters, and metrics.
- `training-rows.jsonl`: row IDs, labels, and text hashes for the full corpus.
- `checkpoints/`: intermediate training checkpoints.

Use `--output-dir /workspace/results/another-new-directory` for another run.
The original EP1 adapter and submission remain unchanged. The new adapter is a
distinct model and is not automatically installed into the frozen EP1 loader
or selected by the ensemble. After refitting, this historical holdout cannot
serve as an unseen evaluation set for the new model.

Preparation was validated on all 20,092 rows with matching dataset and parameter
hashes. This validation did not import model libraries or execute training.

The completed run and its separate test1 evaluation are documented in
[the full-refit evaluation](avaliacao-norberto-full-test1.md), including artifact
verification, class metrics, and comparison with the original EP1 submission.
