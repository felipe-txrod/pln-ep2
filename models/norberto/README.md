# Frozen NorBERTo-512 adapter

`final_model/adapter_model.safetensors` contains the final EP1 LoRA adapter and
trained classification head. Tokenizers and configurations are also versioned.
The public base weights remain separate and outside Git.

- Base: `Itau-Unibanco/NorBERTo-base`.
- Revision: `db73446f89c96044863ea05a39f680524b84bccb`.
- LoRA: rank 4, alpha 8, dropout 0.05, `Wqkv` target modules.
- Context: 512 tokens, head/tail truncation, case-sensitive inputs.
- Final training: two epochs on 16,093 development responses.
- Inference: FP32, SDPA, `reference_compile=False`.

Adapter SHA-256:

```text
d693f6e2995d117ad04a5692817527764fcccee6bfb2dde99120baf9fdd2d385
```

`base_reference.json` records the required base revision and file hashes.
After cloning, run `python3 scripts/prepare-model-base.py` to restore and verify
missing base files without loading or retraining the model. Git ignores the
large base weights file.

Itaú Unibanco and ICTi published the base under **CC BY-NC-SA 4.0**, as recorded
in the [pinned model card](https://huggingface.co/Itau-Unibanco/NorBERTo-base/blob/db73446f89c96044863ea05a39f680524b84bccb/README.md).
This adapter adds the EP1 supervised classification task to the original model.
Preserve attribution and the [base license terms](https://creativecommons.org/licenses/by-nc-sa/4.0/).

See the [inference guide](../../src/norberto_classifier/README.md).
