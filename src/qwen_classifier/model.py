"""Build and reload a Qwen classification head with LoRA adapters."""
from __future__ import annotations

from pathlib import Path

from .config import Config, LABEL2ID


def tokenizer_for(config: Config, cache_dir: Path, *, local_only: bool = True, saved: Path | None = None):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(saved) if saved else config.model_id, revision=None if saved else config.revision,
        cache_dir=str(cache_dir), local_files_only=local_only, trust_remote_code=False,
        padding_side="right", truncation_side="right")
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token = "<|endoftext|>"
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("Classification requires distinct padding and EOS token IDs.")
    return tokenizer


def build_model(config: Config, tokenizer, cache_dir: Path, *, adapter: Path | None = None):
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

    if config.mode == "auto":
        raise ValueError("Resolve the quantization mode through compatibility tests before model loading.")
    kwargs = {"revision": config.revision, "cache_dir": str(cache_dir), "local_files_only": True,
              "trust_remote_code": False, "num_labels": len(LABEL2ID),
              "label2id": LABEL2ID, "id2label": {i: label for label, i in LABEL2ID.items()},
              "problem_type": "single_label_classification", "dtype": torch.float16,
              "attn_implementation": config.attention, "low_cpu_mem_usage": True,
              "device_map": {"": 0}}
    if config.mode == "qlora":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16, llm_int8_skip_modules=["score"])
    base = AutoModelForSequenceClassification.from_pretrained(config.model_id, **kwargs)
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    if config.mode == "qlora":
        import bitsandbytes as bnb
        layers = [module for module in base.modules() if isinstance(module, bnb.nn.Linear4bit)]
        if not layers or any(m.weight.quant_state is None or m.weight.quant_state.quant_type != "nf4" for m in layers):
            raise RuntimeError("The model was not quantized to NF4.")
        if adapter is None:
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
            # PEFT casts all nonquantized weights to FP32, including the 311M-parameter embedding.
            # Keep frozen embeddings in FP16 to save approximately 594 MiB on this GPU.
            base.get_input_embeddings().to(dtype=torch.float16)
    if adapter is not None:
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    else:
        model = get_peft_model(base, LoraConfig(
            task_type=TaskType.SEQ_CLS, r=config.rank, lora_alpha=config.alpha,
            lora_dropout=config.dropout, bias="none", target_modules=list(config.target_modules),
            modules_to_save=["score"]))
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        # AMP GradScaler requires FP32 trainable weights; frozen base weights remain FP16/NF4.
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    if any(p.device.type != "cuda" for p in model.parameters()):
        raise RuntimeError("CPU/disk offload is not enabled for this single-GPU training pipeline.")
    return model
