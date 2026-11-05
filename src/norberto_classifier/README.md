# Frozen EP1 NorBERTo inference package

This package exposes the final model already trained in EP1. It has no
training, hyperparameter selection, or evaluation command.

## Preserved model

| Property | Value |
|---|---|
| Base | `Itau-Unibanco/NorBERTo-base` |
| Revision | `db73446f89c96044863ea05a39f680524b84bccb` |
| Original artifact | `EP1/artifacts/norberto-lora-final-ctx512-20260928` |
| Adapter | LoRA rank 4, alpha 8, dropout 0.05, `Wqkv` modules |
| Input | Only `resp_text`, case-sensitive |
| Context | 512 tokens, head/tail truncation |
| Precision and attention | FP32, SDPA, `reference_compile=False` |
| Class order | `c1`, `c234`, `c5` |
| Inference | Batch 8, two threads, CPU by default |

Official adapter SHA-256:

```text
d693f6e2995d117ad04a5692817527764fcccee6bfb2dde99120baf9fdd2d385
```

The approved selection budget allowed three epochs; final training used two,
selected from the median best epoch across folds. `approved_config.json`
contains the budget and `config.json` the final training parameters. The
original `run.json` records this distinction. Training used 16,093 development
responses and excluded the 3,999 holdout responses.

## Files and integrity

The default package is [`models/norberto/`](../../models/norberto/README.md):

```text
norberto/
  manifest.json
  frozen_manifest.json
  base_reference.json
  approved_config.json
  config.json
  run.json
  base_model/
    config.json
    model.safetensors
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
  final_model/
    adapter_config.json
    adapter_model.safetensors
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
```

The adapter, classification head, tokenizer, configuration, and technical
metadata are versioned. `base_model/model.safetensors` is ignored by Git;
`base_reference.json` pins its public revision and SHA-256. After cloning:

```bash
python3 scripts/models/prepare-base.py
```

This restores and verifies missing base files without loading a model or
training. `--verify-only` checks existing files without accessing the network.
The complete local bundle occupies approximately 607 MB. The public base is
licensed under CC BY-NC-SA 4.0; attribution and source links are in the
[model package](../../models/norberto/README.md).

The source [frozen manifest](frozen_manifest.json) identifies each inference
file by SHA-256. Before loading, the package rejects altered files, a different
base revision, or another adapter. Generated PEFT README and `training_args.bin`
are omitted because inference does not need them; historical `run.json` retains
their original records.

`NORBERTO_ARTIFACT_DIR` or `--artifact-dir` selects an alternate complete bundle.
Without a published package, the previous `artifacts/norberto-ep1/` location
remains a fallback. After preparing the base, inference requires neither EP1
nor Hugging Face access. Model packages are excluded from Docker build contexts.

## Usage

The environment requires an appropriate CPU/ROCm PyTorch build, Transformers
4.57.1, PEFT 0.17.1, Accelerate 1.10.1, and common EP2 dependencies. The `norberto`
extra in `pyproject.toml` declares the three adaptation libraries but does not
select the PyTorch distribution.

From the EP2 root with dependencies installed:

```bash
PYTHONPATH=src python -m norberto_classifier predict \
  --input data/test2.xlsx \
  --output-dir results/norberto-test2 \
  --device cpu
```

The CLI reads text only, ignores any gold labels, and requires a new output
directory. It writes `predictions.jsonl` with original row IDs, text hashes,
classes, and probabilities, plus `run.json` with provenance and configuration.
It does not modify inputs or generate submission XLSX files. GPU use requires
explicit `--device gpu`.

```python
from norberto_classifier import NorbertoClassifier

classifier = NorbertoClassifier(device="cpu")
predictions = classifier.predict(["Example public response."])
label = predictions[0].label
probabilities = predictions[0].probabilities
```

Importing the package or constructing the classifier loads no PyTorch, weights,
or tokenizer. Loading occurs through `load()` or `predict()` with texts.
Parameters remain frozen; inference uses `torch.inference_mode()`.

## Limitations

Normalization, truncation, and adapter loading were ported from the EP1
inference modules identified in the manifest. Numerical equivalence with that
original path has not yet been validated. File hashes establish artifact
identity; historical validation refers to the original EP1 model.

New inference does not replace the official EP1 predictions in statistical
comparisons. Follow the EP2 protocol: freeze the method, submit test2, then
compare test1 against EP1.
