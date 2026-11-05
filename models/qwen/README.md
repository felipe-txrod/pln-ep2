# Selected Qwen model

This package is the default Qwen3-1.7B classifier. It preserves the exact
QLoRA adapter, trained classification head, and tokenizer exported by
`results/qwen-ep1-training-20261105/`. Training completed three epochs and
3,018 updates on the frozen EP1 development partition: 16,093 training rows
and 3,999 validation rows, with no shared rows or similarity groups.

The selected adapter is under `adapter/`. Its raw weights SHA-256 is
`c419c68572ab8e5cd6bda6730714ec27cb28d2d7374d5954a812469346427e53`;
the inference identity, including configuration and tokenizer files, is
`a045c5726a85c466914c55f9cee9fb0d412af4d66387e14b590e53b4782beed5`.
`manifest.json` pins every packaged inference and provenance file.

The public base remains external, pinned to Qwen/Qwen3-1.7B revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, restored through
`base_reference.json` into the existing `.cache/qwen-model/` cache.
The saved adapter is not a merged model. Checkpoints and optimizer state stay
in the original ignored training directory.

`qwen_classifier predict` uses this verified package when `--adapter` is
omitted. Explicit historical overrides remain available. Missing or changed
default files fail validation instead of selecting an older model. Optional
Qwen entries in soft voting and stacking point here. The separate majority
presets retain NorBERTo, SVM and RL. Subsequent native test1 inference scored
414/900 (46.00%); see the [ensemble integration](../../docs/results/ensemble-qwen.md).

Validation scored **1,791/3,999 (44.79%)**, macro-F1 **44.44%**. Validation
selected the best checkpoint and received no gradient updates. This is a
historically consulted holdout, not an independent final test. The source
training report is preserved in `training.json`, without per-step timings.

The original per-row validation export associated grouped prediction output
with sequential workbook IDs. Its order was recovered exactly from the final
checkpoint RNG and sampler, reproducing the recorded confusion matrix.
Correctly aligned predictions are preserved separately under
`results/qwen-ep1-publication-20261105/validation-predictions-aligned.jsonl`.
The historical source file remains unchanged. Future evaluation uses a
sequential sampler and checks label order before writing records; this change
does not modify this adapter's weights.

See the [verification and usage guide](../../docs/results/qwen-ep1-model.md).
