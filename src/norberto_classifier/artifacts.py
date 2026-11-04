"""Resolve and verify a local, immutable copy of the approved EP1 model."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "Itau-Unibanco/NorBERTo-base"
MODEL_REVISION = "db73446f89c96044863ea05a39f680524b84bccb"
ADAPTER_SHA256 = "d693f6e2995d117ad04a5692817527764fcccee6bfb2dde99120baf9fdd2d385"
LABELS = ("c1", "c234", "c5")


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def default_artifact_dir() -> Path:
    configured = os.environ.get("NORBERTO_ARTIFACT_DIR")
    if configured:
        return Path(configured).expanduser()
    project = Path(__file__).resolve().parents[2]
    published = project / "models/norberto"
    return published if (published / "manifest.json").is_file() else project / "artifacts/norberto-ep1"


@dataclass(frozen=True)
class FrozenBundle:
    root: Path
    config: dict[str, Any]
    metadata: dict[str, Any]
    identity_sha256: str

    @property
    def base_model(self) -> Path:
        return self.root / "base_model"

    @property
    def adapter(self) -> Path:
        return self.root / "final_model"


def load_bundle(artifact_dir: Path | None = None) -> FrozenBundle:
    """Verify copied files without importing ML libraries or accessing the GPU."""
    root = (artifact_dir if artifact_dir is not None else default_artifact_dir()).resolve()
    manifest_path = Path(__file__).with_name("frozen_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["model_id"] != MODEL_ID or manifest["revision"] != MODEL_REVISION:
        raise ValueError("The frozen NorBERTo manifest has an unsupported base model.")
    if manifest["files"].get("final_model/adapter_model.safetensors") != ADAPTER_SHA256:
        raise ValueError("Only the approved EP1 delivery adapter may be loaded.")
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.resolve().is_relative_to(root) or not path.is_file():
            if relative == "base_model/model.safetensors" and root == default_artifact_dir():
                raise FileNotFoundError(
                    "Missing pinned NorBERTo base. Run python3 scripts/prepare-model-base.py "
                    "to restore it without training.")
            raise FileNotFoundError(f"Missing local frozen-model file: {path}")
        if file_digest(path) != expected:
            raise ValueError(f"Frozen-model file differs from the EP1 snapshot: {relative}")
    config = json.loads((root / "approved_config.json").read_text(encoding="utf-8"))
    saved = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if saved.get("mode") != "final" or saved.get("status") != "complete":
        raise ValueError("The EP1 artifact is not a completed final model.")
    if config != saved.get("approved_config") or saved.get("labels") != list(LABELS):
        raise ValueError("Frozen inference configuration or label order changed.")
    if (config["model_name"] != MODEL_ID or config["revision"] != MODEL_REVISION
            or config["max_length"] != 512 or config["truncation"] != "head_tail"
            or config["fp16"] is not False or config["attention"] != "sdpa"):
        raise ValueError("The approved NorBERTo-512 inference contract changed.")
    return FrozenBundle(root, config, saved, file_digest(manifest_path))
