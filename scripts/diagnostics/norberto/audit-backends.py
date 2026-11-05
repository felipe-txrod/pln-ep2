"""Compare fixed training-row gradients across saved adapters and attention backends."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new diagnostic output path.")
    sys.path.insert(0, str(args.ep1_root / "src"))
    from clarity import finetune as engine
    from gpt_classifier.data import LABELS, read_xlsx
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    import torch

    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[3]
    bundles = {"original": root / "models/norberto",
               "full_normalized": root / "results/norberto-full-normalized",
               "development_normalized": root / "results/norberto-development-normalized"}
    training = json.loads((bundles["development_normalized"] / "run.json").read_text())
    config = training["config"]
    records, info = read_xlsx(args.ep1_root / "data/train.xlsx", labeled=True)
    assert info["sha256"] == training["dataset_sha256"]
    rows = [json.loads(line) for line in (bundles["development_normalized"] / "training-rows.jsonl").read_text().splitlines()]
    indices = {row["excel_row"] for row in rows}
    cached = json.loads((root / ".cache/finetune-tokens" /
                         (training["tokenization"]["cache_fingerprint"] + ".json")).read_text())
    by_row = {row["excel_row"]: feature for row, feature in zip(rows, cached["features"])}
    selected = []
    for label in LABELS:
        selected.extend([r for r in records if r.excel_row in indices and r.label == label
                         and len(by_row[r.excel_row]["input_ids"]) <= 64][:4])
    assert len(selected) == 12
    tokenizer = engine.AutoTokenizer.from_pretrained(bundles["original"] / "final_model", local_files_only=True)
    batch = engine.DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)([by_row[r.excel_row] for r in selected])
    labels = torch.tensor([LABELS.index(r.label) for r in selected])
    results = {}
    for name, bundle in bundles.items():
        print(f"Auditing {name}: CPU SDPA, CPU eager, GPU SDPA gradients; no optimizer.", flush=True)
        weights = bundle / "final_model/adapter_model.safetensors"
        before = checksum(weights)
        model = engine.make_model(config, config["seed"], adapter_path=bundle / "final_model", allow_download=False)
        for key, parameter in model.named_parameters():
            parameter.requires_grad_("lora_" in key or "modules_to_save.default" in key)
        model.eval()
        observations = {}
        for device, attention in (("cpu", "sdpa"), ("cpu", "eager"), ("cuda", "sdpa")):
            model.to(device)
            model.base_model.model.config._attn_implementation = attention
            model.zero_grad(set_to_none=True)
            prediction = model(**{key: value.to(device) for key, value in batch.items()}, output_hidden_states=True)
            logits = prediction.logits
            loss = torch.nn.functional.cross_entropy(logits.float(), labels.to(device))
            loss.backward()
            parts = {key: parameter.grad.detach().cpu().flatten() for key, parameter in model.named_parameters()
                     if parameter.requires_grad and parameter.grad is not None}
            gradient = torch.cat(list(parts.values()))
            assert torch.isfinite(gradient).all() and torch.isfinite(logits).all()
            observations[f"{device}_{attention}"] = {
                "logits": logits.detach().cpu(), "gradient": gradient,
                "loss": loss.item(), "gradient_norm": gradient.norm().item(),
                "largest_parameter_gradient_norms": dict(sorted(
                    ((key, value.norm().item()) for key, value in parts.items()),
                    key=lambda pair: pair[1], reverse=True)[:10]),
                "hidden_state_rms": [hidden.detach().float().square().mean().sqrt().item()
                                     for hidden in prediction.hidden_states]}
        reference = observations["cpu_sdpa"]
        comparisons = {}
        for key, value in observations.items():
            comparisons[key] = {field: item for field, item in value.items() if field not in ("logits", "gradient")}
            comparisons[key]["max_logit_difference_from_cpu_sdpa"] = (value["logits"] - reference["logits"]).abs().max().item()
            comparisons[key]["relative_gradient_difference_from_cpu_sdpa"] = (
                (value["gradient"] - reference["gradient"]).norm() / reference["gradient"].norm()).item()
        saved = load_file(str(weights))
        actual = get_peft_model_state_dict(model)
        assert set(saved) == set(actual)
        assert all(torch.equal(value, actual[key].detach().cpu()) for key, value in saved.items())
        assert checksum(weights) == before
        results[name] = {"adapter_weights_sha256": before, "backends": comparisons, "weights_unchanged": True}
        del model, actual, saved, observations, prediction, logits, loss, parts, gradient, reference
        gc.collect()
        torch.cuda.empty_cache()

    # Two independent fresh constructions must reproduce the same trainable initialization.
    initialization_hashes = []
    for _ in range(2):
        model = engine.make_model(config, config["seed"], allow_download=False)
        hasher = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                hasher.update(name.encode())
                hasher.update(parameter.detach().cpu().numpy().tobytes())
        initialization_hashes.append(hasher.hexdigest())
        del model
        gc.collect()
    assert initialization_hashes[0] == initialization_hashes[1]
    report = {"status": "completed", "production_model_training_performed": False,
              "optimizer_updates": 0, "training_data_only": True, "test_data_accessed": False,
              "training_rows": [r.excel_row for r in selected],
              "sample_shape": list(batch["input_ids"].shape), "results": results,
              "fresh_trainable_initialization_sha256": initialization_hashes,
              "script_sha256": checksum(Path(__file__))}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "gradient_norms": {
        name: {backend: value["gradient_norm"] for backend, value in result["backends"].items()}
        for name, result in results.items()}, "fresh_initialization_repeatable": True}, indent=2))


if __name__ == "__main__":
    main()
