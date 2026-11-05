"""Resolve the selected Qwen package and verify its inference-file checksums."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

DEFAULT_BUNDLE = Path(__file__).resolve().parents[2] / "models/qwen"


def resolve_adapter(adapter: Path | None = None) -> Path:
    """Use the saved default without falling back to historical run directories."""
    selected = DEFAULT_BUNDLE / "adapter"
    if adapter is not None and Path(adapter).resolve() != selected.resolve():
        return Path(adapter)
    manifest_path = DEFAULT_BUNDLE / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"The selected Qwen package is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model") != "qwen" or manifest.get("labels") != ["c1", "c234", "c5"]:
        raise ValueError("The selected Qwen package has an incompatible model or label contract.")
    files = manifest.get("files")
    required = {"adapter/adapter_model.safetensors", "adapter/adapter_config.json",
                "adapter/experiment.json", "adapter/tokenizer.json", "adapter/tokenizer_config.json"}
    if not isinstance(files, dict) or not required <= set(files):
        raise ValueError("The selected Qwen package lacks required inference checksums.")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("The Qwen manifest contains a path outside its package.")
        path = DEFAULT_BUNDLE / relative
        with path.open("rb") as stream:
            observed = hashlib.file_digest(stream, "sha256").hexdigest()
        if observed != expected:
            raise ValueError(f"The selected Qwen package file changed: {name}")
    return selected
