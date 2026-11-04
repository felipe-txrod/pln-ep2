"""Adapter inference with validated row alignment and optional workbook export."""
from __future__ import annotations

import json
from pathlib import Path

from gpt_classifier.data import LABELS, digest, file_digest, read_xlsx, write_json, write_jsonl
from gpt_classifier.delivery import export_test1, export_test2
from .config import Config
from .dataset import TokenizedTexts
from .model import build_model, tokenizer_for


def adapter_identity(adapter: Path) -> str:
    from gpt_classifier.data import canonical
    names = ("adapter_config.json", "adapter_model.safetensors", "experiment.json",
             "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    return digest(canonical({name: file_digest(adapter / name) for name in names if (adapter / name).exists()}))


def predict_records(adapter: Path, cache_dir: Path, records) -> list[dict]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    saved = json.loads((adapter / "experiment.json").read_text(encoding="utf-8"))
    if saved["labels"] != list(LABELS):
        raise ValueError("The adapter has an incompatible label vocabulary.")
    values = dict(saved["config"])
    values["target_modules"] = tuple(values["target_modules"])
    config = Config(**values)
    tokenizer = tokenizer_for(config, cache_dir, saved=adapter)
    model = build_model(config, tokenizer, cache_dir, adapter=adapter)
    model.eval()
    data = TokenizedTexts(records, tokenizer, config.max_length, labeled=False)
    loader = DataLoader(data, batch_size=1, shuffle=False,
                        collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))
    predictions = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for record, batch in zip(records, loader):
            logits = model(**{key: value.cuda() for key, value in batch.items()}).logits.float()[0]
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Nonfinite prediction at Excel row {record.excel_row}.")
            predictions.append({"excel_row": record.excel_row, "text_sha256": digest(record.text),
                                "label": LABELS[int(logits.argmax().item())],
                                "probabilities": logits.softmax(-1).cpu().tolist(),
                                "logits": logits.cpu().tolist()})
    return predictions


def predict(adapter: Path, cache_dir: Path, source: Path, output: Path, *, phase: str = "inference",
            test2_run: Path | None = None, workbook: Path | None = None) -> dict:
    from .hardware import isolated_probe
    if output.exists():
        raise FileExistsError(f"Use a new inference directory: {output}")
    identity = adapter_identity(adapter)
    records, info = read_xlsx(source)  # Deliberately ignores any clarity column.
    official = {}
    root = Path(__file__).resolve().parents[2]
    for name in ("test1.xlsx", "test1-rotulado-para-teste-estatistico.xlsx", "test2.xlsx"):
        path = root / "data" / name
        if path.exists():
            official[name] = file_digest(path)
    is_test1 = info["sha256"] in {official.get("test1.xlsx"), official.get("test1-rotulado-para-teste-estatistico.xlsx")}
    if is_test1 and phase != "statistical":
        raise ValueError("Official test1 requires phase=statistical after a completed test2 run.")
    if phase == "statistical":
        if not is_test1 or test2_run is None:
            raise ValueError("Statistical inference requires official test1 and a completed test2 run.")
        earlier = json.loads((test2_run / "run.json").read_text(encoding="utf-8"))
        if (earlier.get("status") != "completed" or earlier.get("adapter_sha256") != identity
                or earlier.get("input", {}).get("sha256") != official.get("test2.xlsx")):
            raise ValueError("The completed test2 run must use exactly the same saved adapter.")
    check = isolated_probe("fp16")
    if check["status"] != "passed":
        raise RuntimeError(f"Inference GPU kernels failed: {check}")
    output.mkdir(parents=True)
    predictions = predict_records(adapter, cache_dir, records)
    write_jsonl(output / "predictions.jsonl", predictions)
    if workbook is not None:
        if is_test1:
            export_test1(root / "data" / "test1.xlsx", workbook, predictions)
        elif info["sha256"] == official.get("test2.xlsx"):
            export_test2(source, workbook, predictions)
        else:
            raise ValueError("Workbook export is reserved for official test1/test2 templates.")
    report = {"status": "completed", "input": info, "adapter_sha256": identity,
              "phase": phase, "rows": len(predictions), "workbook": str(workbook) if workbook else None,
              "test2_run": str(test2_run) if test2_run else None}
    write_json(output / "run.json", report)
    return report
