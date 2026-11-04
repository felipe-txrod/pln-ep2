"""Restore a pinned public base using file hashes, without loading model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore the immutable NorBERTo base without model execution")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        package = ROOT / "models/norberto"
        reference = json.loads((package / "base_reference.json").read_text(encoding="utf-8"))
        destination = package / "base_model"
        if (reference["model_id"] != "Itau-Unibanco/NorBERTo-base"
                or reference["revision"] != "db73446f89c96044863ea05a39f680524b84bccb"):
            raise ValueError("The public base reference differs from the immutable EP1 model.")
        contract = json.loads((ROOT / "src/norberto_classifier/frozen_manifest.json").read_text(encoding="utf-8"))
        expected_files = {name.removeprefix("base_model/"): digest
                          for name, digest in contract["files"].items() if name.startswith("base_model/")}
        if reference["files"] != expected_files:
            raise ValueError("The public base hashes differ from the frozen inference contract.")
        for relative, expected in reference["files"].items():
            target = destination / relative
            if Path(relative).is_absolute() or not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Invalid base-model relative path.")
            if target.exists():
                if target.is_symlink() or checksum(target) != expected:
                    raise ValueError(f"The local public base file differs from its pinned hash: {relative}")
                print(f"Verified: {relative}", flush=True)
                continue
            if args.verify_only:
                raise FileNotFoundError(f"Missing public base file: {relative}")
            url = ("https://huggingface.co/" + reference["model_id"] + "/resolve/"
                   + reference["revision"] + "/" + urllib.parse.quote(relative))
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".base-download-", dir=target.parent)
            print(f"Downloading pinned public base file: {relative}", flush=True)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                if checksum(Path(temporary)) != expected:
                    raise ValueError(f"Public base download failed its SHA-256 check: {relative}")
                os.chmod(temporary, 0o644)
                os.link(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
        print("The pinned public base is available locally; no model was loaded.")
        return 0
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
