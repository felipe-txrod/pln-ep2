# Selected Qwen: exact EP1 development partition

[Results index](README.md) · [Model package](../../models/qwen/README.md)
· [Verification aggregate](data/qwen-ep1-selected-model.json)

The completed run `results/qwen-ep1-training-20261105/` is now the default
Qwen model. Its selected adapter was copied without changing its weights to
`models/qwen/adapter/`, together with the complete tokenizer, training
provenance, validation metrics, frozen split, and SHA-256 manifest.

## Training and validation

The configuration is Qwen3-1.7B revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, QLoRA NF4, rank 8/alpha 16,
context 256, batch 4 and accumulation 4, for three epochs. All 3,018 optimizer
updates completed, with no skipped FP16 overflow steps. Training took
11,273.37 seconds, approximately 3 h 8 min. The selected checkpoint is
`checkpoint-3018`, and its weights exactly match the preserved adapter.

| Partition | Rows | Use |
|---|---:|---|
| Frozen EP1 development | 16,093 | Weight fitting |
| Frozen EP1 holdout | 3,999 | Evaluation and best-checkpoint selection |

All 20,092 source texts, labels, and IDs were checked. Training IDs match
development-only RL, and validation IDs match the frozen split. Rows,
similarity groups, and normalized text do not cross the training boundary.

Validation achieved **1,791/3,999 correct (44.79%)**, macro-F1 **44.44%**.
The confusion matrix, with true classes in rows and predicted classes in
columns ordered `c1, c234, c5`, is:

```text
[[571, 359, 327],
 [432, 463, 522],
 [260, 308, 757]]
```

The holdout has been consulted historically and selects the Qwen checkpoint;
this score is validation rather than an untouched final test. Test1 and test2
were not evaluated while preserving this model.

## Per-row export ordering audit

The installed Trainer also applies `group_by_length` to evaluation. Aggregate
metrics correctly pair logits with the Trainer's labels, but the original
export zipped these reordered outputs with the original sequential records.
Consequently, the historical `validation-predictions.json` has incorrect
row-to-prediction associations. It must not be used for row-level analysis.

The exact output permutation was recovered on CPU using the locally saved
`checkpoint-3018/rng_state.pth` and original length-grouped sampler. The
reconstructed order changes 3,998 of 3,999 positions and reproduces every
entry of the recorded confusion matrix, including all 1,791 correct answers.
Aligned predictions and the recovered permutation were saved separately
under `results/qwen-ep1-publication-20261105/`; the historical source file and
weights were preserved.

Future Qwen training keeps length grouping for training batches but uses
sequential evaluation. An explicit label-order check blocks misassociated
exports, and a real CPU Trainer regression test verifies output row order
with variable sequence lengths. No retraining or weight change was needed
to preserve the selected adapter.

## Default inference and artifact verification

Every package file was checksum-verified. All 225 saved tensors are finite,
including the trained 3-by-2,048 classification head. The selected default
was reloaded on the actual ROCm GPU and classified three synthetic texts with
finite logits and normalized probabilities; its files remained unchanged.

Prediction now defaults to the saved package:

```bash
QWEN_IMAGE=pln-rocm:6.4.1 \
QWEN_RUNTIME_PATH=/workspace/.cache/qwen-runtime \
bash scripts/qwen/container.sh predict \
  --input data/test2.xlsx --output-dir results/qwen-ep1-test2
```

This example requires a new output directory and performs inference when
explicitly executed. It was not run during model preservation. Use
`--adapter` only for an explicit alternative. The separate test1 evaluation
script also defaults to the verified package; `--training-run` remains an
explicit historical-run override with its original source-identity checks.

Qwen stays excluded from the default ensemble. Its optional member paths use
`models/qwen/adapter` and new `results/qwen-ep1-test1`/`-test2` directories,
so historical predictions are not reused for the newly selected weights.

To reproduce the byte-preserving export from the original local run:

```bash
python3 scripts/models/freeze.py --model qwen \
  --training-run results/qwen-ep1-training-20261105
```
