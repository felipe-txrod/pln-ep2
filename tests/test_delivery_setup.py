"""Keep heavyweight installations opt-in, including transitive PyTorch installs."""
from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from ep2_delivery import setup


def environment(tmp_path: Path) -> Path:
    folder = tmp_path / "environment"
    (folder / "bin").mkdir(parents=True)
    (folder / "bin/python").touch()
    (folder / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    return folder


def requirement_names(path: Path) -> set[str]:
    names = set()
    for line in path.read_text().splitlines():
        if line.startswith("-r "):
            names |= requirement_names(path.parent / line[3:])
        elif line and not line.startswith(("#", "-c ")):
            names.add(line.split("==")[0])
    return names


def test_default_prepares_cpu_inference_without_rocm():
    options = setup.parse_options([])
    assert options.torch_backend == "cpu" and options.neural is True
    commands = setup.install_commands(options)
    assert "torch==2.8.0+cpu" in commands[0]
    assert commands[0][-1] == "https://download.pytorch.org/whl/cpu"
    assert any("requirements-delivery-neural.txt" in argument for command in commands for argument in command)
    assert not any("rocm" in argument for command in commands for argument in command)


def test_explicit_minimal_never_selects_neural_packages():
    for arguments in [["--minimal"], ["--without-rocm"], ["--without-rocm", "--without-neural"]]:
        options = setup.parse_options(arguments)
        assert options.torch_backend == "keep" and options.neural is False
        commands = setup.install_commands(options)
        assert len(commands) == 2
        assert "--no-deps" in commands[-1]
        assert not any("torch==" in argument or "neural.txt" in argument or "reportlab" in argument
                       for command in commands for argument in command)
    names = requirement_names(setup.ROOT / "requirements-delivery.txt")
    assert names == {"numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl", "openpyxl"}


@pytest.mark.parametrize("flag,backend,package", [
    ("--with-rocm", "rocm", "torch==2.8.0+rocm6.4"),
    ("--with-cpu", "cpu", "torch==2.8.0+cpu"),
])
def test_explicit_backend_uses_only_its_own_index_and_includes_neural(flag, backend, package):
    options = setup.parse_options([flag])
    assert options.torch_backend == backend and options.neural is True
    commands = setup.install_commands(options)
    assert package in commands[0]
    assert "--index-url" in commands[0]
    assert any("requirements-delivery-neural.txt" in argument for command in commands for argument in command)


def test_neural_can_be_excluded_even_with_explicit_backend():
    options = setup.parse_options(["--with-rocm", "--without-neural"])
    assert options.torch_backend == "rocm" and options.neural is False
    assert not any("neural.txt" in argument for command in setup.install_commands(options) for argument in command)


def test_neural_reuses_existing_torch_without_an_install_command():
    for arguments in [["--with-neural"], ["--with-neural", "--without-rocm"]]:
        options = setup.parse_options(arguments)
        assert options.neural is True and options.torch_backend == "keep"
        assert not any("torch==" in argument for command in setup.install_commands(options) for argument in command)
    text = (setup.ROOT / "requirements-delivery-neural.txt").read_text()
    assert "-c requirements-delivery-torch.txt" in text


def test_missing_reused_torch_aborts_before_any_pip_install(tmp_path, monkeypatch):
    folder = environment(tmp_path)
    options = setup.parse_options(["--with-neural", "--venv", str(folder)])
    calls = []

    def reject(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(setup.subprocess, "run", reject)
    with pytest.raises(RuntimeError, match="No package installation was started"):
        setup.execute(options)
    assert len(calls) == 1 and calls[0][1] == "-c"
    assert "import torch" not in calls[0][-1]


def test_default_execution_only_installs_minimal_requirements_and_project(tmp_path, monkeypatch):
    folder = environment(tmp_path)
    calls = []
    monkeypatch.setattr(setup.subprocess, "run", lambda command, **kwargs: calls.append(command))
    setup.execute(setup.parse_options(["--minimal", "--venv", str(folder)]))
    assert len(calls) == 2
    assert str(setup.ROOT / "requirements-delivery.txt") in calls[0]
    assert "--no-deps" in calls[1]


def test_dry_run_creates_nothing_and_never_executes_pip(tmp_path, monkeypatch, capsys):
    destination = tmp_path / "new-environment"
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected process"))
    options = setup.parse_options(["--with-rocm", "--venv", str(destination), "--dry-run"])
    setup.execute(options)
    assert not destination.exists()
    assert "torch==2.8.0+rocm6.4" in capsys.readouterr().out


def test_delivery_setup_has_no_pdf_installation_profile():
    with pytest.raises(SystemExit):
        setup.parse_options(["--with-report"])
    for flags in [[], ["--with-rocm"], ["--with-cpu"], ["--with-neural"]]:
        commands = setup.install_commands(setup.parse_options(flags))
        assert not any("reportlab" in argument for command in commands for argument in command)


def test_rejects_conflicting_backends_and_existing_non_environment(tmp_path):
    with pytest.raises(SystemExit):
        setup.parse_options(["--with-rocm", "--without-rocm"])
    (tmp_path / "user-file.txt").write_text("preserve")
    with pytest.raises(ValueError, match="non-virtual-environment"):
        setup.validate_environment(setup.SetupOptions(environment=tmp_path))
    assert (tmp_path / "user-file.txt").read_text() == "preserve"


def test_existing_environment_visibility_cannot_change_silently(tmp_path):
    folder = environment(tmp_path)
    with pytest.raises(ValueError, match="existing environment is isolated"):
        setup.validate_environment(setup.SetupOptions(environment=folder, system_site_packages=True))


def test_setup_launcher_help_works_without_project_dependencies(tmp_path):
    completed = subprocess.run(
        ["bash", str(setup.ROOT / "scripts/ensemble/setup-local.sh"), "--help"],
        cwd=tmp_path, text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--without-rocm" in completed.stdout and "--dry-run" in completed.stdout
