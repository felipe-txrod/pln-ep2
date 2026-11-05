"""Audit two saved NorBERTo runs and test loss scaling with synthetic CPU backpropagation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gradient_history(run: dict) -> list[dict]:
    rows = run["training"]["history"]
    result = []
    for start, end in ((0, .5), (.5, 1), (1, 1.5), (1.5, 2.01)):
        selected = [row for row in rows if start < row.get("epoch", 0) <= end and "loss" in row]
        result.append({"epoch_start": start, "epoch_end": min(end, 2), "logged_steps": len(selected),
                       "mean_loss": statistics.mean(row["loss"] for row in selected),
                       "mean_grad_norm": statistics.mean(row["grad_norm"] for row in selected),
                       "max_grad_norm": max(row["grad_norm"] for row in selected)})
    return result


def synthetic_gradient_probe(output: Path) -> list[dict]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import ModernBertConfig, ModernBertForSequenceClassification, Trainer, TrainingArguments, set_seed

    set_seed(42)
    config = ModernBertConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=64,
                             local_attention=16, global_attn_every_n_layers=1, pad_token_id=5,
                             cls_token_id=3, sep_token_id=4, num_labels=3, classifier_pooling="mean",
                             classifier_dropout=0, attention_dropout=0, mlp_dropout=0,
                             embedding_dropout=0, reference_compile=False,
                             problem_type="single_label_classification")
    config._attn_implementation = "sdpa"
    model = get_peft_model(ModernBertForSequenceClassification(config), LoraConfig(
        task_type=TaskType.SEQ_CLS, r=4, lora_alpha=8, lora_dropout=0, target_modules=["Wqkv"]))
    tokens = torch.randint(6, 64, (16, 12))
    targets = torch.arange(16) % 3
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}

    def gradients():
        return torch.cat([(parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).flatten()
                          for parameter in model.parameters() if parameter.requires_grad]).detach().clone()

    def sample_loss(outputs, labels, num_items_in_batch=None):
        count = labels.numel() if num_items_in_batch is None else num_items_in_batch
        return torch.nn.functional.cross_entropy(outputs.logits.float(), labels, reduction="sum") / count

    reports = []
    for sizes in ((4, 4, 4, 4), (4, 4, 4, 1)):
        count = sum(sizes)
        model.load_state_dict(initial)
        model.zero_grad(set_to_none=True)
        model.train()
        logits = model(input_ids=tokens[:count], attention_mask=torch.ones_like(tokens[:count])).logits
        reference_loss = torch.nn.functional.cross_entropy(logits.float(), targets[:count])
        reference_loss.backward()
        reference_gradient = gradients()
        result = {"microbatch_sizes": list(sizes), "reference_loss": reference_loss.item()}
        for corrected in (False, True):
            model.load_state_dict(initial)
            model.zero_grad(set_to_none=True)
            trainer = Trainer(model=model, args=TrainingArguments(
                output_dir=str(output / "synthetic"), use_cpu=True, per_device_train_batch_size=4,
                gradient_accumulation_steps=4, report_to=[], disable_tqdm=True, dataloader_pin_memory=False),
                compute_loss_func=sample_loss if corrected else None)
            if corrected:
                trainer.model_accepts_loss_kwargs = False
            trainer.current_gradient_accumulation_steps = len(sizes)
            accumulated_loss = 0.0
            offset = 0
            for size in sizes:
                # Backpropagate only. No Trainer.train or optimizer.step is executed.
                batch = {"input_ids": tokens[offset:offset + size],
                         "attention_mask": torch.ones_like(tokens[offset:offset + size]),
                         "labels": targets[offset:offset + size]}
                accumulated_loss += trainer.training_step(model, batch, num_items_in_batch=torch.tensor(count)).item()
                offset += size
            observed = gradients()
            relative_error = (observed - reference_gradient).norm() / reference_gradient.norm()
            normalized_error = (observed / observed.norm() - reference_gradient / reference_gradient.norm()).norm()
            item = {"model_accepts_loss_kwargs": trainer.model_accepts_loss_kwargs,
                    "accelerator_accumulation_steps": trainer.accelerator.gradient_accumulation_steps,
                    "accumulated_loss": accumulated_loss,
                    "loss_ratio": accumulated_loss / reference_loss.item(),
                    "gradient_norm_ratio": (observed.norm() / reference_gradient.norm()).item(),
                    "gradient_relative_error": relative_error.item(),
                    "normalized_gradient_direction_error": normalized_error.item()}
            result["sample_normalized" if corrected else "legacy"] = item
            if corrected:
                torch.testing.assert_close(observed, reference_gradient, atol=1e-6, rtol=1e-4)
                assert math.isclose(accumulated_loss, reference_loss.item(), rel_tol=1e-6)
        if sizes == (4, 4, 4, 4):
            assert math.isclose(result["legacy"]["gradient_norm_ratio"], 4, rel_tol=1e-5)
        reports.append(result)
    assert all(torch.equal(value, initial[name]) for name, value in model.state_dict().items())
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(args.ep1_root / "src"))
    from clarity.data import load_training_data
    from gpt_classifier.data import read_xlsx
    import torch
    from safetensors.torch import load_file

    if torch.cuda.is_available():
        raise RuntimeError("Run diagnostics without GPU device exposure.")
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    old_root = args.ep1_root / "artifacts/norberto-lora-final-ctx512-20260928"
    new_root = root / "results/norberto-full-training"
    old, new = read(old_root / "run.json"), read(new_root / "run.json")
    frame = load_training_data(args.ep1_root / "data/train.xlsx")
    records, _ = read_xlsx(args.ep1_root / "data/train.xlsx", labeled=True)
    mismatches = [row.excel_row for row, (text, label) in zip(records, frame.itertuples(index=False, name=None))
                  if (row.text, row.label) != (text, label)]
    assert len(frame) == len(records) == 20092 and not mismatches
    with (old_root / "training-rows.csv").open(newline="", encoding="utf-8") as stream:
        development = [int(row["excel_row"]) for row in csv.DictReader(stream)]
    old_cache = read(args.ep1_root / ".cache/finetune-tokens" / (old["tokenization"]["cache_fingerprint"] + ".json"))
    new_cache = read(root / ".cache/finetune-tokens" / (new["tokenization"]["cache_fingerprint"] + ".json"))
    feature_mismatches = [row_id for row_id, feature in zip(development, old_cache["features"])
                          if feature != new_cache["features"][row_id - 2]]
    assert len(development) == len(old_cache["features"]) == 16093 and not feature_mismatches
    assert len(new_cache["features"]) == len(records)
    argument_sets = []
    for bundle, report, expected in ((old_root, old, old["adapter_sha256"]["training_args.bin"]),
                                      (new_root, new, new["adapter_files_sha256"]["training_args.bin"])):
        path = bundle / "final_model/training_args.bin"
        if checksum(path) != expected:
            raise ValueError("Refusing to deserialize unverified original training arguments.")
        argument_sets.append(torch.load(path, map_location="cpu", weights_only=False).to_dict())
    argument_differences = {key: {"old": argument_sets[0].get(key), "new": argument_sets[1].get(key)}
                            for key in argument_sets[0].keys() | argument_sets[1].keys()
                            if argument_sets[0].get(key) != argument_sets[1].get(key)}
    norms = {}
    for name, bundle in (("original", old_root), ("full", new_root)):
        state = load_file(str(bundle / "final_model/adapter_model.safetensors"))
        modules = []
        for key, a in state.items():
            if not key.endswith("lora_A.weight"):
                continue
            b = state[key.replace("lora_A.weight", "lora_B.weight")]
            squared_norm = torch.trace((b.T @ b) @ (a @ a.T)).clamp_min(0)
            modules.append({"module": key.removesuffix(".lora_A.weight"), "a_norm": a.norm().item(),
                            "b_norm": b.norm().item(), "scaled_update_norm": (2 * squared_norm.sqrt()).item()})
        norms[name] = {"modules": modules,
                       "classifier_weight_norm": next(value.norm().item() for key, value in state.items() if key.endswith("classifier.weight")),
                       "classifier_bias_norm": next(value.norm().item() for key, value in state.items() if key.endswith("classifier.bias"))}
    probes = synthetic_gradient_probe(args.output.parent)
    report = {"status": "passed", "device": "cpu", "full_model_training_performed": False,
              "optimizer_updates_performed": 0, "synthetic_backpropagation_only": True,
              "data_reader_mismatches": len(mismatches), "shared_tokenized_rows_checked": len(development),
              "tokenized_feature_mismatches": len(feature_mismatches), "recipe_matches": old["config"] == new["config"],
              "training_argument_differences": argument_differences,
              "history": {"original": gradient_history(old), "full": gradient_history(new)},
              "weight_norms": norms, "synthetic_gradient_probe": probes}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("weight_norms", "history")}, indent=2, default=str))
    print("Mean scaled LoRA update norms:", {key: statistics.mean(module["scaled_update_norm"] for module in value["modules"])
                                            for key, value in norms.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
