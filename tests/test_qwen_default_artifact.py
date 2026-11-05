"""Keep the selected Qwen package verified and separate from historical adapters."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from ensemble_classifier.config import EnsembleConfig
from qwen_classifier import artifacts
from qwen_classifier.cli import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def saved_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "qwen"
    adapter = bundle / "adapter"
    adapter.mkdir(parents=True)
    files = {}
    for name in ("adapter_model.safetensors", "adapter_config.json", "experiment.json",
                 "tokenizer.json", "tokenizer_config.json"):
        path = adapter / name
        path.write_text("synthetic " + name)
        files["adapter/" + name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (bundle / "manifest.json").write_text(json.dumps({"model": "qwen", "labels": ["c1", "c234", "c5"],
                                                    "files": files}))
    monkeypatch.setattr(artifacts, "DEFAULT_BUNDLE", bundle)
    return bundle


def test_default_resolves_saved_bundle_and_rejects_changed_weights(saved_bundle):
    assert artifacts.resolve_adapter() == saved_bundle / "adapter"
    assert artifacts.resolve_adapter(saved_bundle / "adapter") == saved_bundle / "adapter"
    (saved_bundle / "adapter/adapter_model.safetensors").write_text("changed weights")
    with pytest.raises(ValueError, match="file changed"):
        artifacts.resolve_adapter()


def test_missing_default_never_selects_historical_training_run(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "DEFAULT_BUNDLE", tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="selected Qwen package is missing"):
        artifacts.resolve_adapter()
    explicit = tmp_path / "explicit-override"
    assert artifacts.resolve_adapter(explicit) == explicit


@pytest.mark.parametrize("corruption", ["labels", "missing-checksum", "external-path"])
def test_invalid_package_contract_is_rejected(saved_bundle, corruption):
    path = saved_bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    if corruption == "labels":
        manifest["labels"] = ["other"]
    elif corruption == "missing-checksum":
        del manifest["files"]["adapter/tokenizer.json"]
    else:
        manifest["files"]["../outside"] = "a" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        artifacts.resolve_adapter()


def test_predict_cli_allows_the_selected_default_without_adapter_flag(tmp_path, monkeypatch, capsys):
    import qwen_classifier.inference as inference
    observed = []
    monkeypatch.setattr(inference, "predict", lambda adapter, *args, **kwargs:
                        observed.append(adapter) or {"status": "completed"})
    assert main(["predict", "--input", str(tmp_path / "input.xlsx"),
                 "--output-dir", str(tmp_path / "prediction")]) == 0
    assert observed == [None]
    assert '"status": "completed"' in capsys.readouterr().out


def test_freezer_rejects_incomplete_qwen_before_creating_package(tmp_path):
    path = ROOT / "scripts/models/freeze.py"
    spec = importlib.util.spec_from_file_location("qwen_freezer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "training"
    source.mkdir()
    (source / "run.json").write_text(json.dumps({"status": "running", "smoke": False, "pilot": False}))
    output = tmp_path / "published"
    with pytest.raises(ValueError, match="completed full training"):
        module.export_qwen(source, output)
    assert not output.exists()


def test_comparison_script_uses_verified_package_without_original_checkpoints(saved_bundle):
    path = ROOT / "scripts/qwen/evaluate-test1.py"
    spec = importlib.util.spec_from_file_location("selected_qwen_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {"mode": "qlora", "epochs": 3}
    report = {"status": "completed", "smoke": False, "pilot": False, "verified_lora_update": True,
              "config": config}
    (saved_bundle / "adapter/experiment.json").write_text(json.dumps({"config": config, "labels": ["c1", "c234", "c5"]}))
    (saved_bundle / "training.json").write_text(json.dumps(report))
    manifest_path = saved_bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in ("adapter/experiment.json", "training.json"):
        manifest["files"][name] = hashlib.sha256((saved_bundle / name).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    assert module.verify_training(None) == report


def test_optional_qwen_uses_selected_adapter_without_changing_ensemble_membership():
    for name in ("ensemble.json", "ensemble-test1.json"):
        config = EnsembleConfig.load(ROOT / "configs" / name)
        assert config.models == ("norberto", "svm", "rl") and "qwen" not in config.members
        selected = EnsembleConfig.load(ROOT / "configs" / name, models=["qwen"], tiebreaker="qwen")
        assert selected.members["qwen"]["adapter"] == "models/qwen/adapter"
        assert "qwen-ep1-test" in selected.members["qwen"]["run"]
