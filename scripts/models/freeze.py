"""Publish hash-verified inference artifacts without loading or training models."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["c1", "c234", "c5"]


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def copy_file(source: Path, target: Path, expected: str | None = None) -> str:
    observed = checksum(source)
    if expected is not None and observed != expected:
        raise ValueError(f"Source artifact differs from its frozen SHA-256: {source.name}")
    if target.exists():
        if target.is_symlink() or checksum(target) != observed:
            raise FileExistsError(f"Refusing to replace a different frozen file: {target}")
        return observed
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".freeze-", dir=target.parent)
    os.close(descriptor)
    try:
        if shutil.which("cp"):
            subprocess.run(["cp", "--reflink=auto", "--dereference", "--", str(source), temporary], check=True)
        else:
            shutil.copyfile(source, temporary)
        if checksum(Path(temporary)) != observed or checksum(source) != observed:
            raise ValueError(f"The artifact changed while copying: {source.name}")
        os.chmod(temporary, 0o644)
        os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return observed


def write_json(path: Path, value: dict[str, Any]) -> str:
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.is_symlink() or path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Use another destination for a different model package: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
    return checksum(path)


def export_lexical(name: str, output: Path) -> dict[str, Any]:
    contract = ROOT / "src" / f"{name}_classifier" / "frozen_manifest.json"
    frozen = read_json(contract)
    source = ROOT / "artifacts" / f"{name}-ep1"
    expected = {**frozen["files"], "model.joblib": frozen["artifact_sha256"]}
    for relative, digest in expected.items():
        if checksum(source / relative) != digest:
            raise ValueError(f"The original EP1 {name} artifact changed: {relative}")
    files = {relative: copy_file(source / relative, output / relative, digest)
             for relative, digest in expected.items()}
    files["frozen_manifest.json"] = copy_file(contract, output / "frozen_manifest.json")
    return {"schema_version": 1, "model": name, "format": "joblib",
            "labels": LABELS, "files": files, "training_supported": False}


def export_norberto(output: Path, local_base: bool) -> dict[str, Any]:
    contract = ROOT / "src/norberto_classifier/frozen_manifest.json"
    frozen = read_json(contract)
    source = ROOT / "artifacts/norberto-ep1"
    files = {}
    for relative, digest in frozen["files"].items():
        if relative == "base_model/model.safetensors":
            if local_base:
                copy_file(source / relative, output / relative, digest)
            continue
        files[relative] = copy_file(source / relative, output / relative, digest)
    files["frozen_manifest.json"] = copy_file(contract, output / "frozen_manifest.json")
    reference = {"schema_version": 1, "model_id": frozen["model_id"], "revision": frozen["revision"],
                 "license": "cc-by-nc-sa-4.0",
                 "source": "https://huggingface.co/Itau-Unibanco/NorBERTo-base/tree/" + frozen["revision"],
                 "files": {name.removeprefix("base_model/"): digest
                           for name, digest in frozen["files"].items() if name.startswith("base_model/")}}
    files["base_reference.json"] = write_json(output / "base_reference.json", reference)
    return {"schema_version": 1, "model": "norberto", "format": "peft-safetensors",
            "labels": LABELS, "files": files, "base_reference": "base_reference.json",
            "base_license": "cc-by-nc-sa-4.0", "training_supported": False}


def export_gpt(output: Path) -> dict[str, Any]:
    config_path = ROOT / "configs/gpt.json"
    config = read_json(config_path)
    files = {"config.json": copy_file(config_path, output / "config.json")}
    prompt_path = ROOT / "src/gpt_classifier/prompt.py"
    tree = ast.parse(prompt_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name not in {"SYSTEM_PROMPT", "EMPIRICAL_PROMPT"}:
                continue
            content = ast.literal_eval(node.value)
            filename = "prompt.txt" if name == "SYSTEM_PROMPT" else "empirical_prompt.txt"
            target = output / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.is_symlink() or target.read_text(encoding="utf-8") != content:
                    raise FileExistsError(f"The GPT prompt already differs: {target}")
            else:
                target.write_text(content, encoding="utf-8")
            files[filename] = checksum(target)
    if not {"prompt.txt", "empirical_prompt.txt"} <= files.keys():
        raise ValueError("The GPT source must define both frozen prompt templates.")
    return {"schema_version": 1, "model": "gpt", "format": "remote-service",
            "labels": LABELS, "backend": config["backend"], "remote_model": config["model"],
            "weights_available_locally": False, "files": files,
            "prompt_implementation_sha256": checksum(prompt_path)}


def export_qwen(source: Path, output: Path) -> dict[str, Any]:
    """Preserve the selected final adapter only after verifying training and partition provenance."""
    report = read_json(source / "run.json")
    if (report.get("status") != "completed" or report.get("smoke") is not False
            or report.get("pilot") is not False or report.get("verified_lora_update") is not True):
        raise ValueError("Qwen export requires a completed full training run with verified LoRA updates.")
    adapter = source / "adapter"
    experiment = read_json(adapter / "experiment.json")
    config = report["config"]
    if experiment.get("config") != config or experiment.get("labels") != LABELS:
        raise ValueError("The Qwen adapter configuration differs from its completed run.")
    states = [read_json(path) for path in (source / "checkpoints").glob("*/trainer_state.json")]
    if not states:
        raise ValueError("The Qwen run lacks a final trainer state.")
    final = max(states, key=lambda row: row["global_step"])
    if (final["global_step"] != final["max_steps"] or final["global_step"] != report["optimizer_steps"]
            or not math.isclose(final["epoch"], config["epochs"])):
        raise ValueError("Qwen training did not complete all configured epochs and steps.")
    checkpoint = source / "checkpoints" / Path(report["best_checkpoint"]).name
    if Path(final["best_model_checkpoint"]).name != checkpoint.name:
        raise ValueError("The final trainer state names another selected Qwen checkpoint.")
    weights = checksum(adapter / "adapter_model.safetensors")
    if checksum(checkpoint / "adapter_model.safetensors") != weights:
        raise ValueError("The Qwen export differs from the checkpoint selected in validation.")
    partition = report.get("partition")
    if (not isinstance(partition, dict) or experiment.get("partition") != partition
            or partition.get("method") != "frozen_ep1_development_holdout"
            or report.get("train_rows") != 16093 or report.get("validation_rows") != 3999
            or partition.get("holdout_used_for_fit") is not False):
        raise ValueError("The selected Qwen must use the frozen 16,093/3,999 EP1 partition.")
    source_sha = "0e9219233664ac675fc47982bbc822acd3bb15d67fb7bfc26b67bec047e46818"
    split_sha = "0f78e7a809c7b7584db05b33cf03a73ff586d5de02ba824c16562d5070d41494"
    expected = {"data/split.json": partition["split_manifest_sha256"],
                "data/train.jsonl": partition["train_records_sha256"],
                "data/validation.jsonl": partition["validation_records_sha256"],
                "data/frozen-split.csv": split_sha}
    if partition["source_sha256"] != source_sha or partition["split_sha256"] != split_sha:
        raise ValueError("Qwen training names a different source corpus or frozen split.")
    for name, digest in expected.items():
        if checksum(source / name) != digest:
            raise ValueError(f"The Qwen training data changed: {name}")
    with (source / "data/frozen-split.csv").open(newline="") as stream:
        split = list(csv.DictReader(stream))
    split_manifest = read_json(source / "data/split.json")
    for role, filename, field in (("development", "train.jsonl", "train_rows"),
                                  ("holdout", "validation.jsonl", "validation_rows")):
        with (source / "data" / filename).open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        ids = [int(row["excel_row"]) for row in split if row["role"] == role]
        if [row["excel_row"] for row in records] != ids or split_manifest[field] != ids:
            raise ValueError("Qwen prepared row IDs differ from the frozen split.")
    scores = read_json(source / "validation-metrics.json")
    if scores != report["validation_metrics"]:
        raise ValueError("Qwen validation metrics differ from the completed training report.")
    files = {}
    names = ("adapter_model.safetensors", "adapter_config.json", "experiment.json", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json",
             "merges.txt", "chat_template.jinja", "README.md")
    for name in ("adapter_model.safetensors", "adapter_config.json", "experiment.json",
                 "tokenizer.json", "tokenizer_config.json"):
        if not (adapter / name).is_file():
            raise ValueError(f"The selected Qwen adapter lacks an inference file: {name}")
    for name in names:
        path = adapter / name
        if path.is_file():
            relative = "adapter/" + name
            files[relative] = copy_file(path, output / relative)
    identity_names = ("adapter_config.json", "adapter_model.safetensors", "experiment.json",
                      "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    identity = {name: checksum(adapter / name) for name in identity_names if (adapter / name).is_file()}
    adapter_sha = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest()
    metadata = {key: value for key, value in report.items() if key != "update_seconds"}
    metadata["source_training_run_sha256"] = checksum(source / "run.json")
    metadata["validation_predictions_sha256"] = checksum(source / "validation-predictions.json")
    files["training.json"] = write_json(output / "training.json", metadata)
    files["validation-metrics.json"] = copy_file(source / "validation-metrics.json", output / "validation-metrics.json")
    for name in ("split.json", "frozen-split.csv"):
        files["partition/" + name] = copy_file(source / "data" / name, output / "partition" / name)
    reference = {"model_id": config["model_id"], "revision": config["revision"],
                 "cache_dir": ".cache/qwen-model", "versioned_base_weights": False,
                 "restore_command": "bash scripts/qwen/container.sh download --config configs/qwen-fast.json"}
    files["base_reference.json"] = write_json(output / "base_reference.json", reference)
    return {"schema_version": 1, "model": "qwen", "format": "peft-safetensors", "labels": LABELS,
            "adapter_sha256": adapter_sha, "adapter_weights_sha256": weights,
            "source_training_run": str(source), "source_training_run_sha256": checksum(source / "run.json"),
            "base_reference": "base_reference.json", "training_rows": 16093, "validation_rows": 3999,
            "split_sha256": split_sha, "files": files, "training_supported": False}


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze inference files without loading model weights")
    parser.add_argument("--model", required=True, choices=("svm", "rl", "norberto", "gpt", "qwen"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--training-run", type=Path, help="Completed Qwen EP1 training run to preserve")
    parser.add_argument("--with-local-base", action="store_true",
                        help="Copy the ignored NorBERTo base for local offline use")
    args = parser.parse_args()
    try:
        output = args.output if args.output is not None else ROOT / "models" / args.model
        if args.with_local_base and args.model != "norberto":
            raise ValueError("--with-local-base applies only to NorBERTo.")
        if (args.model == "qwen") != (args.training_run is not None):
            raise ValueError("--training-run is required only when exporting Qwen.")
        if args.model in {"svm", "rl"}:
            manifest = export_lexical(args.model, output)
        elif args.model == "norberto":
            manifest = export_norberto(output, args.with_local_base)
        elif args.model == "qwen":
            manifest = export_qwen(args.training_run, output)
        else:
            manifest = export_gpt(output)
        write_json(output / "manifest.json", manifest)
        print(json.dumps({"model": args.model, "files": len(manifest["files"]),
                          "format": manifest["format"], "model_execution_performed": False}, indent=2))
        return 0
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
