"""Compare saved adapters on training-only data and CPU/GPU gradients without optimization."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.ep1_root / "src"))
    from clarity import finetune as original
    from gpt_classifier.data import LABELS, read_xlsx
    from transformers import DataCollatorWithPadding
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict
    import torch

    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[3]
    old_root = args.ep1_root / "artifacts/norberto-lora-final-ctx512-20260928"
    new_root = root / "results/norberto-full-training"
    old = json.loads((old_root / "run.json").read_text())
    new = json.loads((new_root / "run.json").read_text())
    assert old["config"] == new["config"]
    config = new["config"]
    records, info = read_xlsx(args.ep1_root / "data/train.xlsx", labeled=True)
    assert info["sha256"] == new["dataset_sha256"]
    with (old_root / "training-rows.csv").open(encoding="utf-8", newline="") as stream:
        development = {int(row["excel_row"]) for row in csv.DictReader(stream)}
    cache = json.loads((root / ".cache/finetune-tokens" /
                        (new["tokenization"]["cache_fingerprint"] + ".json")).read_text())
    rng = random.Random(42)
    selected, short = [], []
    for label in LABELS:
        candidates = [record.excel_row - 2 for record in records
                      if record.excel_row in development and record.label == label]
        selected.extend(rng.sample(candidates, 100))
        short.extend([index for index in candidates if len(cache["features"][index]["input_ids"]) <= 64][:4])
    selected.sort()
    selection_digest = hashlib.sha256(json.dumps([records[index].excel_row for index in selected]).encode()).hexdigest()
    tokenizer = original.AutoTokenizer.from_pretrained(old_root / "final_model", local_files_only=True)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    short_batch = collator([cache["features"][index] for index in short])
    short_labels = torch.tensor([LABELS.index(records[index].label) for index in short])
    results = {}
    for name, bundle in (("original", old_root), ("full", new_root)):
        print(f"Checking {name}: CPU/GPU synthetic-free training-row gradients and 300 shared training predictions.", flush=True)
        model = original.make_model(config, config["seed"], adapter_path=bundle / "final_model", allow_download=False)
        for key, parameter in model.named_parameters():
            parameter.requires_grad_("lora_" in key or "modules_to_save.default" in key)
        model.eval()  # Disable dropout to isolate numerical backend differences.
        observations = {}
        for device in ("cpu", "cuda"):
            model.to(device)
            model.zero_grad(set_to_none=True)
            logits = model(**{key: value.to(device) for key, value in short_batch.items()}).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), short_labels.to(device))
            loss.backward()  # No optimizer exists; weights are never updated.
            gradient = torch.cat([parameter.grad.detach().flatten() for parameter in model.parameters()
                                  if parameter.requires_grad and parameter.grad is not None]).cpu()
            assert torch.isfinite(logits).all() and torch.isfinite(gradient).all()
            observations[device] = {"logits": logits.detach().cpu(), "gradient": gradient,
                                    "loss": loss.item(), "gradient_norm": gradient.norm().item()}
        reference, observed = observations["cpu"], observations["cuda"]
        model.zero_grad(set_to_none=True)
        correct = 0
        total_loss = 0.0
        predictions = dict.fromkeys(LABELS, 0)
        with torch.inference_mode():
            for offset in range(0, len(selected), 8):
                indices = selected[offset:offset + 8]
                batch = collator([cache["features"][index] for index in indices])
                labels = torch.tensor([LABELS.index(records[index].label) for index in indices], device="cuda")
                logits = model(**{key: value.to("cuda") for key, value in batch.items()}).logits.float()
                assert torch.isfinite(logits).all()
                total_loss += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
                guessed = logits.argmax(dim=1)
                correct += int((guessed == labels).sum().item())
                for value in guessed.cpu().tolist():
                    predictions[LABELS[value]] += 1
        actual_state = get_peft_model_state_dict(model)
        saved_state = load_file(str(bundle / "final_model/adapter_model.safetensors"))
        assert set(actual_state) == set(saved_state)
        assert all(torch.equal(actual_state[key].detach().cpu(), value) for key, value in saved_state.items())
        results[name] = {"short_training_rows": len(short),
                         "cpu_loss": reference["loss"], "gpu_loss": observed["loss"],
                         "cpu_gradient_norm": reference["gradient_norm"], "gpu_gradient_norm": observed["gradient_norm"],
                         "cpu_gpu_max_logit_difference": (observed["logits"] - reference["logits"]).abs().max().item(),
                         "cpu_gpu_relative_gradient_difference": ((observed["gradient"] - reference["gradient"]).norm() /
                                                                  reference["gradient"].norm()).item(),
                         "shared_training_rows": 300, "shared_training_correct": correct,
                         "shared_training_accuracy": correct / 300, "shared_training_cross_entropy": total_loss / 300,
                         "shared_training_prediction_counts": predictions, "weights_unchanged": True}
        del logits, loss, gradient, actual_state, model
        gc.collect()
        torch.cuda.empty_cache()
    result = {"status": "completed", "training_data_only": True, "test1_accessed": False,
              "optimizer_updates": 0, "sample_seed": 42, "rows_per_class": 100,
              "selection_sha256": selection_digest, "results": results}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
