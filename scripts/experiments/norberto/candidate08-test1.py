"""Evaluate the original grid-screening candidate-08, without refitting or promotion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from gpt_classifier.data import LABELS, digest, file_digest, read_xlsx, write_json, write_jsonl
from gpt_classifier.evaluation import align, metrics
from norberto_classifier.artifacts import load_bundle
from norberto_classifier.tokenization import tokenize_texts

ROOT = Path(__file__).resolve().parents[3]
GRID = "norberto-lora-grid-20260929-170333"
ADAPTER = Path("artifacts") / GRID / "screening/candidate-08/best_adapter"
RESULTS = Path("results/norberto/experiments/grid-search") / GRID
EXPECTED = {
    ADAPTER / "adapter_model.safetensors": "8e9c8332866fe3c2649382bb6fc4f9970b76471364516e58b7f690bef0e066ca",
    RESULTS / "configs/candidate-08.json": "3d08e96c1c3e171e912634d36edbb4f906b19eda0ea9ec65744691efb22c7866",
    RESULTS / "screening/candidate-08/results.json": "c6776b5b52af80851f75c71930af66156a08ade19effc883eca5850d97fa9054",
}


def verify_source(ep1_root: Path) -> tuple[dict, dict, dict]:
    for relative, expected in EXPECTED.items():
        if file_digest(ep1_root / relative) != expected:
            raise ValueError(f"The original screening artifact changed: {relative}")
    config = json.loads((ep1_root / RESULTS / "configs/candidate-08.json").read_text())
    source = json.loads((ep1_root / RESULTS / "screening/candidate-08/results.json").read_text())
    if (source["candidate_id"] != "candidate-08" or source["adapter_dir"] != str(ADAPTER)
            or source["train_rows"] != 12982 or source["validation_rows"] != 3111
            or source["best_epoch"] != 2 or config["lora_rank"] != 16):
        raise ValueError("Expected the screening adapter, not a CV-fold or final adapter.")
    adapter = ep1_root / ADAPTER
    hashes = {str(relative): file_digest(ep1_root / relative) for relative in EXPECTED}
    hashes.update({str(ADAPTER / path.name): file_digest(path) for path in adapter.iterdir() if path.is_file()})
    return config, source, hashes


def evaluate(args: argparse.Namespace) -> dict:
    config, source, hashes = verify_source(args.ep1_root)
    bundle = load_bundle(ROOT / "models/norberto")
    if args.preflight:
        return {"status": "ready", "model": "candidate-08-grid-screening", "adapter": str(args.ep1_root / ADAPTER),
                "adapter_sha256": hashes[str(ADAPTER / "adapter_model.safetensors")],
                "training_rows": source["train_rows"], "screening_accuracy": source["accuracy"],
                "selected_epoch": source["best_epoch"], "inference_performed": False}
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --preflight is used.")
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory.")
    records, info = read_xlsx(args.input)
    if len(records) != 900:
        raise ValueError("All 900 original Test1 rows are required.")

    import torch
    from peft import PeftModel
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, set_seed

    if args.device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but CUDA/ROCm is unavailable.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = "cuda" if args.device == "gpu" else "cpu"
    set_seed(config["seed"])
    torch.set_num_threads(config["threads"])
    adapter = args.ep1_root / ADAPTER
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True, trust_remote_code=False)
    base = AutoModelForSequenceClassification.from_pretrained(
        bundle.base_model, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={label: i for i, label in enumerate(LABELS)},
        attn_implementation=config["attention"], reference_compile=False, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False, local_files_only=True)
    model.requires_grad_(False)
    model.to(device).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Inference must have no trainable parameters.")
    features, tokenization = tokenize_texts([row.text for row in records], tokenizer, config)
    loader = DataLoader(features, batch_size=config["eval_batch_size"], shuffle=False, num_workers=0,
                        collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))
    outputs, started = [], time.monotonic()
    with torch.inference_mode():
        for batch in loader:
            logits = model(**{name: value.to(device) for name, value in batch.items()}).logits.float()
            if logits.ndim != 2 or logits.shape[1] != 3 or not torch.isfinite(logits).all():
                raise FloatingPointError("Invalid candidate-08 logits.")
            outputs.extend(logits.softmax(-1).cpu().tolist())
    predictions = [{"excel_row": row.excel_row, "text_sha256": digest(row.text),
                    "label": LABELS[max(range(3), key=lambda index: values[index])],
                    "probabilities": dict(zip(LABELS, values))} for row, values in zip(records, outputs)]
    if len(outputs) != len(records):
        raise ValueError("Incomplete predictions.")
    align(records, predictions)
    if file_digest(args.input) != info["sha256"] or any(file_digest(args.ep1_root / path) != sha for path, sha in hashes.items()):
        raise ValueError("An input or adapter file changed during inference.")
    prediction_path = args.output_dir / "predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    report = {"status": "predicted", "model": "candidate-08-grid-screening", "rows": len(records),
              "training_performed": False, "gold_accessed": False, "model_promoted": False,
              "input": info, "source_files": hashes, "source_screening": source, "config": config,
              "tokenization": tokenization, "inference_seconds": time.monotonic() - started,
              "predictions_sha256": file_digest(prediction_path), "device": args.device,
              "script_sha256": file_digest(Path(__file__))}
    write_json(args.output_dir / "run.json", report)
    # Read gold only after all predictions and their identity are persisted.
    gold, gold_info = read_xlsx(args.gold, labeled=True)
    if len(gold) != 900 or [(r.excel_row, r.text) for r in gold] != [(r.excel_row, r.text) for r in records]:
        raise ValueError("Test1 gold row IDs/texts differ from the prediction input.")
    scores = metrics(gold, predictions)
    scores["correct"] = sum(truth.label == row["label"] for truth, row in zip(gold, predictions))
    write_json(args.output_dir / "metrics.json", scores)
    report.update(status="completed", gold_accessed=True, gold_read_after_predictions=True,
                  gold_sha256=gold_info["sha256"], metrics=scores)
    write_json(args.output_dir / "run.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep1-root", type=Path, default=ROOT.parent / "EP1")
    parser.add_argument("--input", type=Path, default=ROOT / "data/test1.xlsx")
    parser.add_argument("--gold", type=Path, default=ROOT / "data/test1-rotulado-para-teste-estatistico.xlsx")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight", action="store_true", help="Verify the screening adapter without inference or gold access.")
    args = parser.parse_args()
    try:
        result = evaluate(args)
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
