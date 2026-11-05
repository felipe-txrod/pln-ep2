"""Audit the complete corrected Trainer loop against manual tiny-model updates."""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace


def run(output: Path, device: str = "cpu", recipe: str = "sample") -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (ModernBertConfig, ModernBertForSequenceClassification,
                              Trainer, TrainerCallback, TrainingArguments)
    from norberto_classifier.training_loss import sample_normalized_trainer

    if output.exists():
        raise FileExistsError("Use a new diagnostic output path.")
    if device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("The GPU diagnostic requires an available accelerator.")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = ModernBertConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, max_position_embeddings=64, local_attention=16,
        pad_token_id=5, num_labels=3, reference_compile=False,
        classifier_dropout=0, attention_dropout=0, mlp_dropout=0, embedding_dropout=0)
    config._attn_implementation = "sdpa"
    model = get_peft_model(ModernBertForSequenceClassification(config), LoraConfig(
        task_type="SEQ_CLS", r=4, lora_alpha=8, lora_dropout=0, target_modules=["Wqkv"]))
    reference = copy.deepcopy(model)
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    tokens = torch.randint(6, 64, (29, 12))
    labels = torch.arange(29) % 3
    dataset = [{"input_ids": tokens[i], "attention_mask": torch.ones(12, dtype=torch.long),
                "labels": labels[i]} for i in range(29)]
    module = SimpleNamespace(Trainer=Trainer)
    output.parent.mkdir(parents=True, exist_ok=True)
    from norberto_classifier.training_recipe import compatible_normalized_trainer
    context = compatible_normalized_trainer if recipe == "compatible" else sample_normalized_trainer
    gradient_scale = 4 if recipe == "compatible" else 1
    with context(module):
        trainer = module.Trainer(model=model, train_dataset=dataset, args=TrainingArguments(
            output_dir=str(output.parent / f"tiny-loop-{device}"), use_cpu=device == "cpu",
            per_device_train_batch_size=4, gradient_accumulation_steps=4,
            num_train_epochs=2, learning_rate=.001, warmup_ratio=.1, weight_decay=.01,
            optim="adamw_torch", max_grad_norm=1, save_strategy="no", report_to=[],
            disable_tqdm=True, logging_steps=1, logging_nan_inf_filter=False,
            dataloader_pin_memory=False, label_names=["labels"], seed=42, data_seed=42))
    trainer.create_optimizer_and_scheduler(num_training_steps=4)
    named = dict(model.named_parameters())
    reference_named = dict(reference.named_parameters())
    names_by_id = {id(parameter): name for name, parameter in named.items()}
    groups = [{**{key: value for key, value in group.items() if key != "params"},
               "params": [reference_named[names_by_id[id(parameter)]] for parameter in group["params"]]}
              for group in trainer.optimizer.param_groups]
    defaults = {key: value for key, value in trainer.optimizer.defaults.items()
                if key in inspect.signature(torch.optim.AdamW).parameters}
    optimizer = torch.optim.AdamW(groups, **defaults)
    batches, counts, steps, microbatch_losses = [], [], [], []
    original_step = trainer.training_step
    original_clip = trainer.accelerator.clip_grad_norm_

    def gradients(current):
        return torch.cat([p.grad.detach().flatten().cpu() for p in current.parameters()
                          if p.requires_grad and p.grad is not None])

    def training_step(current, inputs, num_items_in_batch=None):
        if not batches:
            assert all(p.grad is None or not p.grad.count_nonzero() for p in current.parameters())
        batches.append({key: value.detach().cpu().clone() for key, value in inputs.items()})
        counts.append(int(num_items_in_batch))
        observed = original_step(current, inputs, num_items_in_batch=num_items_in_batch)
        microbatch_losses.append(observed.item())
        return observed

    def clip(parameters, max_norm, norm_type=2):
        reference.zero_grad(set_to_none=True)
        combined = {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}
        gold = combined.pop("labels")
        reference.train()
        loss = torch.nn.functional.cross_entropy(reference(**combined).logits, gold)
        (loss * gradient_scale).backward()
        expected = gradients(reference)
        observed = gradients(model)
        relative = ((observed - expected).norm() / expected.norm()).item()
        torch.testing.assert_close(observed, expected, atol=2e-6, rtol=2e-4)
        assert counts == [len(gold)] * len(batches), (counts, len(gold))
        assert abs(sum(microbatch_losses) - loss.item()) < 1e-6
        pre_norm = original_clip(parameters, max_norm, norm_type)
        torch.nn.utils.clip_grad_norm_(reference.parameters(), max_norm, norm_type)
        post_norm = gradients(model).norm().item()
        assert post_norm <= 1.00001
        lr = trainer.optimizer.param_groups[0]["lr"]
        assert abs(lr - (0, .001, .001 * 2 / 3, .001 / 3)[len(steps)]) < 1e-12
        for group, actual_group in zip(optimizer.param_groups, trainer.optimizer.param_groups):
            group["lr"] = actual_group["lr"]
        optimizer.step()
        steps.append({"update": len(steps) + 1, "microbatch_sizes": [len(b["labels"]) for b in batches],
                      "reported_item_counts": counts.copy(), "reference_loss": loss.item(),
                      "gradient_relative_error": relative, "preclipping_norm": float(pre_norm),
                      "postclipping_norm": post_norm, "learning_rate": lr})
        batches.clear()
        counts.clear()
        microbatch_losses.clear()
        return pre_norm

    class VerifyUpdate(TrainerCallback):
        def on_optimizer_step(self, args, state, control, **kwargs):
            differences = []
            for name, parameter in named.items():
                expected = reference_named[name]
                torch.testing.assert_close(parameter.detach().cpu(), expected, atol=2e-6, rtol=2e-4)
                differences.append((parameter.detach().cpu() - expected.detach()).abs().max().item())
            steps[-1]["parameter_max_absolute_error"] = max(differences)

    trainer.add_callback(VerifyUpdate())
    trainer.training_step = training_step
    trainer.accelerator.clip_grad_norm_ = clip
    result = trainer.train()
    assert trainer.state.global_step == 4 and not batches
    assert all(torch.equal(named[name].detach().cpu(), value) for name, value in frozen.items())
    assert abs(result.training_loss - sum(step["reference_loss"] for step in steps) / 4) < 1e-6
    report = {"status": "passed", "model": "random tiny ModernBERT with PEFT", "device": device,
              "manual_reference_device": "cpu", "recipe": recipe, "optimizer_gradient_scale": gradient_scale,
              "production_model_training_performed": False, "production_weights_modified": False,
              "tiny_frozen_parameters_unchanged": True,
              "test_data_accessed": False, "tiny_model_optimizer_updates": 4,
              "accelerator_accumulation_steps": trainer.accelerator.gradient_accumulation_steps,
              "model_accepts_loss_kwargs": trainer.model_accepts_loss_kwargs,
              "compute_loss_func": trainer.compute_loss_func.__name__, "steps": steps,
              "trainer_loss": result.training_loss,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--recipe", choices=("sample", "compatible"), default="sample")
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.device, args.recipe), indent=2))


if __name__ == "__main__":
    main()
