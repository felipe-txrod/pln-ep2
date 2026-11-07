"""Prepare CPU delivery by default, with explicit ROCm and minimal profiles."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TORCH_CHECK = (
    "import importlib.metadata as m; "
    "v=m.version('torch'); "
    "assert v.split('+',1)[0]=='2.8.0', f'Expected existing torch 2.8.0; found {v}'; "
    "print(f'Keeping installed PyTorch: {v}', flush=True)"
)
ROCM_CHECK = (
    "import importlib.metadata as m; "
    "v=m.version('torch'); "
    "assert v=='2.8.0+rocm6.4', f'Expected torch 2.8.0+rocm6.4; found {v}'; "
    "print(f'Prepared ROCm PyTorch: {v}', flush=True)"
)


@dataclass(frozen=True)
class SetupOptions:
    environment: Path
    torch_backend: str = "cpu"
    neural: bool = True
    system_site_packages: bool = False
    dry_run: bool = False

    @property
    def python(self) -> Path:
        return self.environment / "bin/python"


def parse_options(argv: list[str] | None = None) -> SetupOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--with-rocm", dest="torch_backend", action="store_const", const="rocm",
                         help="Install ROCm PyTorch and enable neural dependencies by default")
    backend.add_argument("--without-rocm", "--minimal", dest="torch_backend", action="store_const", const="keep",
                         help="Install only the lexical/export base; keep existing PyTorch untouched")
    backend.add_argument("--with-cpu", dest="torch_backend", action="store_const", const="cpu",
                         help="Install CPU-only PyTorch and neural dependencies (default)")
    parser.set_defaults(torch_backend=None, neural=None)
    neural = parser.add_mutually_exclusive_group()
    neural.add_argument("--with-neural", dest="neural", action="store_true",
                        help="Install Transformer/PEFT dependencies using an existing or explicitly selected PyTorch")
    neural.add_argument("--without-neural", dest="neural", action="store_false",
                        help="Skip Transformer/PEFT dependencies")
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv-delivery",
                        help="Create or update this virtual environment (relative paths use the project root)")
    parser.add_argument("--system-site-packages", action="store_true",
                        help="Allow a new virtual environment to reuse base-interpreter site packages")
    parser.add_argument("--dry-run", action="store_true", help="Print the installation plan without creating files or downloading packages")
    args = parser.parse_args(argv)
    torch_backend = args.torch_backend
    if torch_backend is None:
        torch_backend = "cpu" if args.neural is None else "keep"
    environment = args.venv.expanduser()
    if not environment.is_absolute():
        environment = ROOT / environment
    return SetupOptions(environment=environment.absolute(), torch_backend=torch_backend,
                        neural=(torch_backend != "keep") if args.neural is None else args.neural,
                        system_site_packages=args.system_site_packages,
                        dry_run=args.dry_run)


def validate_environment(options: SetupOptions) -> None:
    environment = options.environment
    if environment.exists() and any(environment.iterdir()) and not (environment / "pyvenv.cfg").is_file():
        raise ValueError(f"Refusing to use a non-virtual-environment directory: {environment}")
    if options.python.is_file() and options.system_site_packages:
        config = (environment / "pyvenv.cfg").read_text().lower()
        if "include-system-site-packages = true" not in config:
            raise ValueError("The existing environment is isolated. Choose a new --venv to enable --system-site-packages.")


def install_commands(options: SetupOptions) -> list[list[str]]:
    python = str(options.python)
    pip = [python, "-m", "pip", "install", "--disable-pip-version-check"]
    commands = []
    if options.torch_backend == "rocm":
        commands.append([*pip, "torch==2.8.0+rocm6.4", "--index-url", "https://download.pytorch.org/whl/rocm6.4"])
    elif options.torch_backend == "cpu":
        commands.append([*pip, "torch==2.8.0+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"])
    commands.append([*pip, "-r", str(ROOT / "requirements-delivery.txt")])
    if options.neural:
        commands.append([*pip, "-r", str(ROOT / "requirements-delivery-neural.txt")])
    commands.append([*pip, "--no-deps", "-e", str(ROOT)])
    return commands


def execute(options: SetupOptions) -> None:
    validate_environment(options)
    commands = install_commands(options)
    creation = [sys.executable, "-m", "venv"]
    if options.system_site_packages:
        creation.append("--system-site-packages")
    creation.append(str(options.environment))
    print(f"Environment: {options.environment}", flush=True)
    print(f"PyTorch: {options.torch_backend}; neural libraries: {options.neural}", flush=True)
    if options.dry_run:
        if not options.python.is_file():
            print(shlex.join(creation))
        if options.neural and options.torch_backend == "keep":
            print("Precondition: PyTorch 2.8.0 must already be visible in the selected environment; it will not be installed.")
        for command in commands:
            print(shlex.join(command))
        return
    if not options.python.is_file():
        subprocess.run(creation, check=True)
    if options.neural and options.torch_backend == "keep":
        try:
            subprocess.run([str(options.python), "-c", TORCH_CHECK], check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "--with-neural without a backend requires an existing PyTorch 2.8.0. "
                "Select its environment with --venv, reuse system packages with --system-site-packages, "
                "or explicitly request --with-rocm/--with-cpu. No package installation was started.") from exc
    for command in commands:
        print(f"Running: {shlex.join(command)}", flush=True)
        subprocess.run(command, check=True)
    if options.torch_backend != "keep":
        check = ROCM_CHECK if options.torch_backend == "rocm" else TORCH_CHECK
        subprocess.run([str(options.python), "-c", check], check=True)
    if options.neural:
        print("Prepared neural environment. Validate with: bash scripts/ensemble/deliver-test2.sh --check", flush=True)
    else:
        print("Prepared minimal environment for --export-only. Neural inference requires the optional dependencies.", flush=True)
    if options.environment != ROOT / ".venv-delivery":
        print(f"Select this environment with EP2_DELIVERY_PYTHON={shlex.quote(str(options.python))}", flush=True)


def main(argv: list[str] | None = None) -> int:
    options = parse_options(argv)
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        print("Setup requires Python 3.11 or 3.12; set EP2_SETUP_PYTHON accordingly.", file=sys.stderr)
        return 1
    try:
        execute(options)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
