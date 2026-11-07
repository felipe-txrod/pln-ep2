"""Exercise native launchers and delivery integrity without neural inference."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from openpyxl import Workbook
import pytest

from ep2_delivery import cli
from gpt_classifier.data import LABELS, digest, file_digest, read_xlsx, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def saved_run(tmp_path):
    source = tmp_path / "test2.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["resp_text", "clarity"])
    for i in range(897):
        sheet.append([f"Synthetic response {i}", None])
    sheet.freeze_panes = "A2"
    sheet.column_dimensions["A"].width = 70
    workbook.save(source)
    workbook.close()
    records, info = cli.read_source(source)
    run = tmp_path / "predictions"
    run.mkdir()
    rows = []
    for i, record in enumerate(records):
        label = LABELS[i % 3]
        rows.append({"excel_row": record.excel_row, "text_sha256": digest(record.text),
                     "label": label, "probabilities": {c: float(c == label) for c in LABELS}})
    write_jsonl(run / "predictions.jsonl", rows)
    manifest = {"status": "completed", "rows": 897, "gold_accessed": False,
                "base_training_performed": False, "calibration_fitted": False,
                "artifact_sha256": cli.MODEL_SHA256, "weights": cli.WEIGHTS,
                "labels": list(LABELS), "input": info,
                "predictions_sha256": file_digest(run / "predictions.jsonl")}
    write_json(run / "run.json", manifest)
    return SimpleNamespace(source=source, records=records, info=info, run=run,
                           rows=rows, manifest=manifest, output=tmp_path / "delivery.xlsx")


def export_args(saved):
    return SimpleNamespace(input=saved.source, run=saved.run, output=saved.output,
                           device="gpu", check=False, export_only=True)


def test_export_only_reopens_exact_897_labels_and_preserves_other_zip_entries(saved_run):
    saved = saved_run
    before = file_digest(saved.source)
    with patch("ep2_delivery.cli.require_versions", return_value={}), \
         patch("ensemble_classifier.soft_pipeline.load_soft_artifact", return_value={"weights": cli.WEIGHTS}), \
         patch("ep2_delivery.cli.check_device", side_effect=AssertionError("GPU must not be used")), \
         patch("ensemble_classifier.soft_pipeline.predict_soft", side_effect=AssertionError("No new inference")):
        first = cli.generate(export_args(saved))
        saved.output.write_bytes(b"An outdated output must be replaced.")
        second = cli.generate(export_args(saved))
    assert first["output_sha256"] == second["output_sha256"]
    assert first["source_unchanged"] is True
    assert second["inference_performed"] is False
    assert file_digest(saved.source) == before
    checked, _ = read_xlsx(saved.output, labeled=True)
    assert [r.label for r in checked] == [r["label"] for r in saved.rows]
    assert len(checked) == 897
    with ZipFile(saved.source) as old, ZipFile(saved.output) as new:
        changed = [name for name in old.namelist() if old.read(name) != new.read(name)]
        assert len(changed) == 1 and changed[0].startswith("xl/worksheets/")
    assert json.loads((saved.run / "delivery.json").read_text())["format_and_order_verified"] is True


@pytest.fixture
def native_delivery(saved_run, monkeypatch):
    monkeypatch.setattr(cli, "require_versions", lambda **kwargs: {})
    monkeypatch.setattr("ensemble_classifier.soft_pipeline.load_soft_artifact", lambda path: {"weights": cli.WEIGHTS})
    monkeypatch.setattr("norberto_classifier.artifacts.load_bundle", lambda: None)
    monkeypatch.setattr(cli, "check_device", lambda device: {"device": device})
    calls = []

    def predict(model, source, destination, **kwargs):
        assert destination != saved_run.run
        assert not destination.exists()
        assert kwargs == {"device": "cpu", "live_norberto": True}
        calls.append(destination)
        destination.mkdir()
        rows = [dict(row) for row in saved_run.rows]
        label = LABELS[len(calls) % 3]
        rows[0] = {**rows[0], "label": label,
                   "probabilities": {c: float(c == label) for c in LABELS}}
        write_jsonl(destination / "predictions.jsonl", rows)
        write_json(destination / "run.json", {
            **saved_run.manifest, "predictions_sha256": file_digest(destination / "predictions.jsonl")})
        member = destination / "members/svm"
        member.mkdir(parents=True)
        (member / "fresh.txt").write_text(str(len(calls)))

    monkeypatch.setattr("ensemble_classifier.soft_pipeline.predict_soft", predict)
    args = export_args(saved_run)
    args.device = "cpu"
    args.export_only = False
    return args, calls


def test_rerun_recomputes_all_members_and_replaces_run_and_workbook(saved_run, native_delivery):
    args, calls = native_delivery
    old_member = saved_run.run / "members/svm"
    old_member.mkdir(parents=True)
    (old_member / "stale.txt").write_text("Old cached scores")
    first = cli.generate(args)
    second = cli.generate(args)
    assert len(calls) == 2 and calls[0] != calls[1]
    assert all(not path.exists() for path in calls)
    assert not (old_member / "stale.txt").exists()
    assert (old_member / "fresh.txt").read_text() == "2"
    assert first["output_sha256"] != second["output_sha256"]
    assert second["cached_scores_reused"] is False
    assert second["inference_performed"] is True
    assert file_digest(saved_run.output) == second["output_sha256"]
    assert read_xlsx(saved_run.output, labeled=True)[0][0].label == LABELS[2]
    assert file_digest(saved_run.source) == saved_run.info["sha256"]


@pytest.mark.parametrize("failure", ["inference", "export", "verification"])
def test_failed_regeneration_preserves_previous_successful_delivery(saved_run, native_delivery, monkeypatch, failure):
    from gpt_classifier.delivery import export_test2

    args, _ = native_delivery
    export_test2(saved_run.source, saved_run.output, saved_run.rows)
    previous = {path: path.read_bytes() for path in saved_run.run.rglob("*") if path.is_file()}
    old_workbook = saved_run.output.read_bytes()

    def fail(*args, **kwargs):
        raise RuntimeError("Synthetic failure")

    target = {"inference": "ensemble_classifier.soft_pipeline.predict_soft",
              "export": "gpt_classifier.delivery.export_test2",
              "verification": "ep2_delivery.cli.verify_workbook"}[failure]
    monkeypatch.setattr(target, fail)
    with pytest.raises(RuntimeError, match="Synthetic failure"):
        cli.generate(args)
    assert saved_run.output.read_bytes() == old_workbook
    assert {path: path.read_bytes() for path in saved_run.run.rglob("*") if path.is_file()} == previous
    assert not list(saved_run.run.parent.glob(f".{saved_run.run.name}-*"))
    assert not list(saved_run.output.parent.glob(f".{saved_run.output.stem}-*"))


def test_unrecognized_run_directory_is_not_replaced(saved_run, native_delivery):
    args, calls = native_delivery
    (saved_run.run / "run.json").unlink()
    unrelated = saved_run.run / "important.txt"
    unrelated.write_text("Keep this file")
    with pytest.raises(ValueError, match="unrecognized"):
        cli.generate(args)
    assert not calls
    assert unrelated.read_text() == "Keep this file"


def test_first_generation_publishes_to_a_new_run_directory(saved_run, native_delivery):
    args, calls = native_delivery
    args.run = saved_run.run.parent / "new-results/delivery"
    result = cli.generate(args)
    assert len(calls) == 1
    assert (args.run / "run.json").exists()
    assert (args.run / "delivery.json").exists()
    assert file_digest(saved_run.output) == result["output_sha256"]


@pytest.mark.parametrize("field", ["artifact_sha256", "input"])
def test_another_experiment_is_not_overwritten(saved_run, native_delivery, field):
    args, calls = native_delivery
    saved_run.manifest[field] = "different-model" if field == "artifact_sha256" else {"sha256": "different-input"}
    write_json(saved_run.run / "run.json", saved_run.manifest)
    previous = (saved_run.run / "run.json").read_bytes()
    with pytest.raises(ValueError, match="another input or model"):
        cli.generate(args)
    assert not calls
    assert (saved_run.run / "run.json").read_bytes() == previous


def test_failed_run_publication_restores_previous_directory(saved_run, tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    staging.mkdir()
    staged = staging / "run"
    staged.mkdir()
    original_rename = Path.rename
    previous = (saved_run.run / "run.json").read_bytes()

    def rename(path, target):
        if path == staged:
            raise OSError("Synthetic rename failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(OSError, match="Synthetic rename failure"):
        cli.replace_run(staged, saved_run.run)
    assert (saved_run.run / "run.json").read_bytes() == previous


@pytest.mark.parametrize("field,value", [
    ("status", "preparing"), ("rows", 900), ("gold_accessed", True),
    ("base_training_performed", True), ("calibration_fitted", True),
    ("artifact_sha256", "0" * 64), ("weights", {"svm": 1.0}),
])
def test_rejects_incomplete_or_different_model_runs(saved_run, field, value):
    saved_run.manifest[field] = value
    write_json(saved_run.run / "run.json", saved_run.manifest)
    with pytest.raises(ValueError, match="incomplete"):
        cli.read_completed(saved_run.run, saved_run.records, saved_run.info)
    assert not saved_run.output.exists()


def test_rejects_changed_predictions_and_wrong_text_order(saved_run):
    saved = saved_run
    write_jsonl(saved.run / "predictions.jsonl", list(reversed(saved.rows)))
    with pytest.raises(ValueError, match="checksum"):
        cli.read_completed(saved.run, saved.records, saved.info)
    saved.manifest["predictions_sha256"] = file_digest(saved.run / "predictions.jsonl")
    write_json(saved.run / "run.json", saved.manifest)
    with pytest.raises(ValueError, match="alignment"):
        cli.read_completed(saved.run, saved.records, saved.info)


def test_rejects_labels_that_disagree_with_saved_probabilities(saved_run):
    saved = saved_run
    saved.rows[0]["label"] = "c5"
    write_jsonl(saved.run / "predictions.jsonl", saved.rows)
    saved.manifest["predictions_sha256"] = file_digest(saved.run / "predictions.jsonl")
    write_json(saved.run / "run.json", saved.manifest)
    with pytest.raises(ValueError, match="argmax"):
        cli.read_completed(saved.run, saved.records, saved.info)


def test_rejects_nonblank_input_without_inference(tmp_path):
    source = tmp_path / "labeled.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["resp_text", "clarity"])
    for i in range(897):
        sheet.append([f"Synthetic {i}", "c1"])
    workbook.save(source)
    workbook.close()
    with pytest.raises(ValueError, match="must be blank"):
        cli.read_source(source)


def test_rejects_other_workbook_changes_even_when_labels_match(saved_run):
    from gpt_classifier.delivery import export_test2

    saved = saved_run
    export_test2(saved.source, saved.output, saved.rows)
    with ZipFile(saved.output, "a") as archive:
        archive.writestr("unexpected.txt", "modified")
    with pytest.raises(ValueError, match="archive structure"):
        cli.verify_workbook(saved.source, saved.output, saved.rows)


def test_cuda_torch_is_rejected_before_gpu_detection():
    fake = SimpleNamespace(__version__="2.8.0+cu128", version=SimpleNamespace(hip=None),
                           cuda=SimpleNamespace(is_available=lambda: pytest.fail("No GPU probe expected")))
    with patch.dict(sys.modules, torch=fake), pytest.raises(RuntimeError, match="not a ROCm build"):
        cli.check_device("gpu")


def test_rocm_build_with_inaccessible_gpu_is_rejected():
    fake = SimpleNamespace(__version__="2.8.0+rocm6.4", version=SimpleNamespace(hip="6.4"),
                           cuda=SimpleNamespace(is_available=lambda: False))
    with patch.dict(sys.modules, torch=fake), pytest.raises(RuntimeError, match="cannot access"):
        cli.check_device("gpu")


@pytest.mark.parametrize("arguments,device", [([], "cpu"), (["--device", "gpu"], "gpu")])
def test_delivery_defaults_to_cpu_and_preserves_explicit_gpu(arguments, device, monkeypatch):
    captured = []
    monkeypatch.setattr(cli, "generate", lambda args: captured.append(args.device) or {"status": "checked"})
    assert cli.main(arguments) == 0
    assert captured == [device]


def test_cpu_preflight_does_not_probe_gpu():
    fake = SimpleNamespace(__version__="2.8.0+cpu", version=SimpleNamespace(hip=None),
                           cuda=SimpleNamespace(is_available=lambda: pytest.fail("CPU must not probe GPU")))
    with patch.dict(sys.modules, torch=fake):
        assert cli.check_device("cpu")["device"] == "cpu"


def test_native_launcher_resolves_project_and_import_path_outside_repository(tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        "print(json.dumps({'args':sys.argv[1:],'cwd':os.getcwd(),"
        "'path':os.environ['PYTHONPATH'],'override':os.environ['HSA_OVERRIDE_GFX_VERSION']}))\n")
    fake_python.chmod(0o755)
    env = {**os.environ, "EP2_DELIVERY_PYTHON": str(fake_python),
           "PYTHONPATH": "/unrelated/python/cache", "HSA_OVERRIDE_GFX_VERSION": "10.3.1"}
    completed = subprocess.run(["bash", str(ROOT / "scripts/ensemble/deliver-test2.sh"), "--check"],
                               cwd=tmp_path, env=env, text=True, capture_output=True, check=True)
    captured = json.loads(completed.stdout)
    assert captured["args"] == ["-u", "-m", "ep2_delivery", "--check"]
    assert captured["cwd"] == str(ROOT)
    assert captured["path"] == str(ROOT / "src")
    assert captured["override"] == "10.3.1"


def test_prediction_paths_are_resolved_from_project_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert cli.project_path("data/test2.xlsx") == ROOT / "data/test2.xlsx"
    assert cli.project_path(str(tmp_path / "output.xlsx")) == tmp_path / "output.xlsx"
