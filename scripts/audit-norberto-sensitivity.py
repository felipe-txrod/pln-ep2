"""Check the unstable adapter's CPU derivatives and first-attention localization."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep1-root", type=Path, required=True)
    parser.add_argument("--backend-report", type=Path, required=True)
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

    if torch.cuda.is_available():
        raise RuntimeError("Run this diagnostic without GPU device exposure.")
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    bundle = root / "results/norberto-development-normalized"
    run = json.loads((bundle / "run.json").read_text())
    audit = json.loads(args.backend_report.read_text())
    records, info = read_xlsx(args.ep1_root / "data/train.xlsx", labeled=True)
    assert info["sha256"] == run["dataset_sha256"]
    saved_rows = [json.loads(line) for line in (bundle / "training-rows.jsonl").read_text().splitlines()]
    cache = json.loads((root / ".cache/finetune-tokens" /
                        (run["tokenization"]["cache_fingerprint"] + ".json")).read_text())
    features = {row["excel_row"]: feature for row, feature in zip(saved_rows, cache["features"])}
    selected = [records[row - 2] for row in audit["training_rows"]]
    assert all(record.excel_row in features for record in selected)
    tokenizer = engine.AutoTokenizer.from_pretrained(bundle / "final_model", local_files_only=True)
    batch = engine.DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)([features[r.excel_row] for r in selected])
    labels = torch.tensor([LABELS.index(r.label) for r in selected])
    model = engine.make_model(run["config"], run["config"]["seed"],
                              adapter_path=bundle / "final_model", allow_download=False)
    for key, parameter in model.named_parameters():
        parameter.requires_grad_("lora_" in key or "modules_to_save.default" in key)
    model.eval()
    first_attention = model.base_model.model.model.layers[0].attn.Wqkv

    def loss():
        return torch.nn.functional.cross_entropy(model(**batch).logits.float(), labels)

    baseline = loss()
    baseline.backward()
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad and p.grad is not None}
    energy = {name: p.grad.double().square().sum().item() for name, p in parameters.items()}
    first_energy = sum(value for name, value in energy.items() if ".layers.0.attn.Wqkv.lora_" in name)
    total_energy = sum(energy.values())
    chosen_name = max(parameters, key=lambda name: parameters[name].grad.abs().max().item())
    chosen = parameters[chosen_name]
    flat = chosen.view(-1)
    coordinate = int(chosen.grad.abs().view(-1).argmax())
    analytic = chosen.grad.view(-1)[coordinate].item()
    initial_value = flat[coordinate].detach().clone()
    finite_differences = []
    try:
        with torch.no_grad():
            for epsilon in (1e-3, 1e-4, 1e-5):
                flat[coordinate].copy_(initial_value + epsilon)
                positive_value = flat[coordinate].item()
                positive_loss = loss().item()
                flat[coordinate].copy_(initial_value - epsilon)
                negative_value = flat[coordinate].item()
                negative_loss = loss().item()
                derivative = (positive_loss - negative_loss) / (positive_value - negative_value)
                finite_differences.append({"epsilon": epsilon, "central_derivative": derivative,
                                           "analytic_derivative": analytic,
                                           "relative_error": abs(derivative - analytic) / abs(analytic)})
    finally:
        with torch.no_grad():
            flat[coordinate].copy_(initial_value)
    assert min(row["relative_error"] for row in finite_differences) < .02
    model.zero_grad(set_to_none=True)
    scaling = first_attention.scaling["default"]
    try:
        first_attention.scaling["default"] = 0
        ablated_loss = loss()
        ablated_loss.backward()
        ablated_norm = sum(p.grad.double().square().sum().item() for p in model.parameters()
                           if p.requires_grad and p.grad is not None) ** .5
    finally:
        first_attention.scaling["default"] = scaling
    actual = get_peft_model_state_dict(model)
    saved = load_file(str(bundle / "final_model/adapter_model.safetensors"))
    assert set(actual) == set(saved)
    assert all(torch.equal(actual[name].detach(), tensor) for name, tensor in saved.items())
    report = {"status": "passed", "device": "cpu", "training_data_only": True,
              "optimizer_updates": 0, "production_weights_unchanged": True,
              "baseline_loss": baseline.item(), "baseline_gradient_norm": total_energy ** .5,
              "first_attention_gradient_energy_fraction": first_energy / total_energy,
              "finite_difference_parameter": chosen_name, "finite_differences": finite_differences,
              "first_attention_lora_disabled": {"loss": ablated_loss.item(), "gradient_norm": ablated_norm},
              "ablation_scope": "Diagnostic on 12 training rows only; no model selection or replacement.",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
