"""Apply the selected frozen ensemble and publish the official Test2 workbook."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/ensemble-soft/model.json"
MODEL_SHA256 = "f39a2200ecbaf124b95c78e249087f2f12302aa4f46768db756d828863301bec"
WEIGHTS = {"svm": 0.75, "rl": 0.0, "norberto": 0.25}
LEXICAL_VERSIONS = {"numpy": "2.3.3", "scipy": "1.16.2", "scikit-learn": "1.7.2", "joblib": "1.5.2", "openpyxl": "3.1.5"}
NEURAL_VERSIONS = {"torch": "2.8.0", "transformers": "4.57.1", "peft": "0.17.1", "accelerate": "1.10.1"}


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def require_versions(*, neural: bool, device: str = "cpu") -> dict[str, str]:
    versions = {}
    expected = {**LEXICAL_VERSIONS, **(NEURAL_VERSIONS if neural else {})}
    profile = " --with-rocm" if neural and device == "gpu" else "" if neural else " --minimal"
    setup = f"bash scripts/ensemble/setup-local.sh{profile}"
    for name, version in expected.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Missing dependency {name}. Run {setup} "
                "or set EP2_DELIVERY_PYTHON to an existing compatible environment.") from exc
        if installed.split("+", 1)[0] != version:
            raise RuntimeError(
                f"{name}=={version} is required; found {installed}. "
                f"Run {setup} in a separate delivery environment.")
        versions[name] = installed
    return versions


def check_device(device: str) -> dict[str, Any]:
    import torch

    if device == "cpu":
        return {"device": "cpu", "torch": torch.__version__, "hip": torch.version.hip}
    if not torch.version.hip:
        raise RuntimeError(
            f"PyTorch {torch.__version__} is not a ROCm build. System ROCm alone is insufficient. "
            "Run bash scripts/ensemble/setup-local.sh --with-rocm or select a ROCm-enabled Python with EP2_DELIVERY_PYTHON.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The ROCm PyTorch build cannot access an AMD GPU. Check /dev/kfd, /dev/dri, "
            "the render/video group permissions, and HSA_OVERRIDE_GFX_VERSION.")
    return {"device": "gpu", "name": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "hip": torch.version.hip}


def read_source(source: Path):
    from openpyxl import load_workbook
    from gpt_classifier.data import read_xlsx

    records, info = read_xlsx(source)
    if len(records) != 897 or info["columns"] != ["resp_text", "clarity"]:
        raise ValueError("Official Test2 delivery requires exactly 897 rows and columns resp_text, clarity.")
    workbook = load_workbook(source, read_only=True, data_only=False)
    try:
        if any(row[0] is not None for row in workbook.worksheets[0].iter_rows(
                min_row=2, min_col=2, max_col=2, values_only=True)):
            raise ValueError("The input clarity column must be blank; a labeled workbook is not a delivery source.")
    finally:
        workbook.close()
    return records, info


def read_completed(run: Path, records, source: dict[str, Any]) -> list[dict[str, Any]]:
    from ensemble_classifier.files import align, read_json, read_predictions, validate_scores
    from gpt_classifier.data import LABELS, file_digest

    manifest = read_json(run / "run.json")
    if (manifest.get("status") != "completed" or manifest.get("rows") != 897
            or manifest.get("gold_accessed") is not False
            or manifest.get("base_training_performed") is not False
            or manifest.get("calibration_fitted") is not False
            or manifest.get("artifact_sha256") != MODEL_SHA256
            or manifest.get("weights") != WEIGHTS
            or manifest.get("labels") != list(LABELS)
            or manifest.get("input", {}).get("sha256") != source["sha256"]):
        raise ValueError("The saved run is incomplete or does not match the selected frozen Test2 delivery.")
    predictions = run / "predictions.jsonl"
    if file_digest(predictions) != manifest.get("predictions_sha256"):
        raise ValueError("Saved predictions differ from their completed-run checksum.")
    rows = read_predictions(predictions)
    align(records, rows)
    for row in rows:
        if not isinstance(row.get("probabilities"), dict):
            raise ValueError("Saved ensemble probabilities are required.")
        validate_scores(row, list(LABELS))
        winner = max(LABELS, key=lambda label: row["probabilities"][label])
        if row["label"] != winner:
            raise ValueError("A saved label differs from the probability argmax.")
    return rows


def verify_workbook(source: Path, target: Path, predictions: list[dict[str, Any]]) -> None:
    from gpt_classifier.data import digest, read_xlsx

    original, _ = read_xlsx(source)
    saved, info = read_xlsx(target, labeled=True)
    if len(saved) != 897 or info["columns"] != ["resp_text", "clarity"]:
        raise ValueError("The output workbook has an invalid delivery format.")
    for before, after, prediction in zip(original, saved, predictions, strict=True):
        if (before.excel_row != after.excel_row or before.text != after.text
                or prediction["excel_row"] != after.excel_row
                or prediction["text_sha256"] != digest(after.text)
                or prediction["label"] != after.label):
            raise ValueError("The output workbook differs from the validated predictions or original texts.")
    with ZipFile(source) as old, ZipFile(target) as new:
        if old.namelist() != new.namelist() or old.comment != new.comment:
            raise ValueError("The workbook archive structure changed.")
        changed = [name for name in old.namelist() if old.read(name) != new.read(name)]
        if len(changed) != 1 or not changed[0].startswith("xl/worksheets/"):
            raise ValueError("The output must change only the original worksheet XML.")


def validate_run_destination(run: Path, source: Path, output: Path) -> None:
    """Allow replacement only of an empty directory or this delivery's own run."""
    from ensemble_classifier.files import read_json
    from gpt_classifier.data import file_digest

    if any(path.is_relative_to(run) for path in (source, output, MODEL)):
        raise ValueError("The run directory must not contain the input, output or frozen model.")
    if not run.exists():
        return
    if not run.is_dir():
        raise ValueError("The run destination must be a directory.")
    if not any(run.iterdir()):
        return
    marker = run / "plan.json"
    if not marker.exists():
        marker = run / "run.json"
    if not marker.exists():
        raise ValueError("Refusing to replace an unrecognized run directory; select a dedicated --run path.")
    identity = read_json(marker)
    if (identity.get("artifact_sha256") != MODEL_SHA256
            or identity.get("input", {}).get("sha256") != file_digest(source)):
        raise ValueError("Refusing to replace a run belonging to another input or model; select another --run path.")


def replace_run(staged: Path, run: Path) -> None:
    """Publish a validated run, restoring the previous directory on rename failure."""
    previous = staged.parent / "previous"
    if run.exists():
        run.rename(previous)
    try:
        staged.rename(run)
    except OSError:
        if previous.exists():
            previous.rename(run)
        raise


def generate(args: argparse.Namespace) -> dict[str, Any]:
    versions = require_versions(neural=not args.export_only, device=args.device)
    from ensemble_classifier.soft_pipeline import load_soft_artifact, predict_soft
    from gpt_classifier.backend import file_lock
    from gpt_classifier.data import file_digest, write_json
    from gpt_classifier.delivery import export_test2

    artifact = load_soft_artifact(MODEL)
    if file_digest(MODEL) != MODEL_SHA256 or artifact["weights"] != WEIGHTS:
        raise ValueError("The selected original 75/25 ensemble changed.")
    source, run, output = (path.resolve() for path in (args.input, args.run, args.output))
    if source == output or output.suffix.lower() != ".xlsx":
        raise ValueError("Use a separate XLSX output path.")
    records, info = read_source(source)
    device = None
    if not args.export_only:
        from norberto_classifier.artifacts import load_bundle
        load_bundle()
        device = check_device(args.device)
    if args.check:
        return {"status": "ready", "python": sys.executable, "versions": versions,
                "device": device, "rows": len(records), "weights": WEIGHTS,
                "input": str(source), "output": str(output), "inference_performed": False}
    run.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(file_lock(run.parent / f".{run.name}.delivery.lock"))
        stack.enter_context(file_lock(output.parent / f".{output.name}.delivery.lock"))
        validate_run_destination(run, source, output)
        working_run = run
        if not args.export_only:
            staging = stack.enter_context(TemporaryDirectory(prefix=f".{run.name}-", dir=run.parent))
            working_run = Path(staging) / "run"
            print(f"Using {sys.executable}; device={device}", flush=True)
            print("Recomputing all member predictions without cached scores; no training or calibration. "
                  "NorBERTo inference may take a few minutes.", flush=True)
            predict_soft(MODEL, source, working_run, device=args.device, live_norberto=True)
        rows = read_completed(working_run, records, info)
        if file_digest(source) != info["sha256"]:
            raise ValueError("The source workbook changed during prediction.")
        export_directory = stack.enter_context(TemporaryDirectory(prefix=f".{output.stem}-", dir=output.parent))
        staged_output = Path(export_directory) / output.name
        export_test2(source, staged_output, rows)
        verify_workbook(source, staged_output, rows)
        report = {"status": "completed", "rows": 897, "weights": WEIGHTS,
                  "model_sha256": MODEL_SHA256, "input_sha256": info["sha256"],
                  "predictions_sha256": file_digest(working_run / "predictions.jsonl"),
                  "output": str(output), "output_sha256": file_digest(staged_output),
                  "source_unchanged": file_digest(source) == info["sha256"],
                  "format_and_order_verified": True, "versions": versions,
                  "training_performed": False, "calibration_fitted": False,
                  "remote_calls": 0, "inference_performed": not args.export_only,
                  "cached_scores_reused": args.export_only}
        if not report["source_unchanged"]:
            raise ValueError("The source workbook changed during export.")
        write_json(working_run / "delivery.json", report)
        if not args.export_only:
            replace_run(working_run, run)
        os.replace(staged_output, output)
        print(f"Delivery saved, replacing any previous output: {output}", flush=True)
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=project_path, default=ROOT / "data/test2.xlsx")
    parser.add_argument("--run", type=project_path, default=ROOT / "results/soft-voting-test2-final")
    parser.add_argument("--output", type=project_path, default=ROOT / "deliveries/test2-soft-voting.xlsx")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="cpu",
                        help="Inference device; CPU is the default and GPU must be requested explicitly")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="Validate native dependencies, model, source and device without inference")
    modes.add_argument("--export-only", action="store_true", help="Overwrite the workbook from a completed prediction run without new inference or importing PyTorch")
    args = parser.parse_args(argv)
    try:
        result = generate(args)
    except (OSError, ImportError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Delivery failed: {exc}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0
