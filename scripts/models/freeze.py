"""Publish hash-verified inference artifacts without loading or training models."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze inference files without loading model weights")
    parser.add_argument("--model", required=True, choices=("svm", "rl", "norberto", "gpt"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--with-local-base", action="store_true",
                        help="Copy the ignored NorBERTo base for local offline use")
    args = parser.parse_args()
    try:
        output = args.output if args.output is not None else ROOT / "models" / args.model
        if args.with_local_base and args.model != "norberto":
            raise ValueError("--with-local-base applies only to NorBERTo.")
        if args.model in {"svm", "rl"}:
            manifest = export_lexical(args.model, output)
        elif args.model == "norberto":
            manifest = export_norberto(output, args.with_local_base)
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
