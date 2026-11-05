"""Check relocated roots and launcher arguments without containers or models."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ScriptLayoutTests(unittest.TestCase):
    def capture_launcher(self, relative: str, arguments: list[str], **extra: str) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            fake_docker = temporary / "docker"
            fake_docker.write_text(
                f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
            fake_docker.chmod(0o755)
            ep1 = temporary / "EP1"
            (ep1 / "src/clarity").mkdir(parents=True)
            (ep1 / "src/clarity/finetune.py").touch()
            (ep1 / ".cache/huggingface").mkdir(parents=True)
            env = {**os.environ, "PATH": str(temporary) + os.pathsep + os.environ["PATH"],
                   "NORBERTO_EP1_ROOT": str(ep1), "NORBERTO_IMAGE": "test-image",
                   "CONTAINER_RUNTIME": str(fake_docker), "QWEN_IMAGE": "test-image", **extra}
            completed = subprocess.run(["bash", str(ROOT / relative), *arguments], cwd=temporary,
                                       env=env, text=True, capture_output=True, check=True)
            return json.loads(completed.stdout)

    def test_prediction_exposes_gpu_and_uses_relocated_script(self):
        args = self.capture_launcher("scripts/norberto/test1.sh", ["predict", "--help"])
        self.assertIn(f"{ROOT}:/workspace", args)
        self.assertIn("--device=/dev/kfd", args)
        self.assertIn("scripts/norberto/evaluate-test1.py", args)
        self.assertEqual(args[args.index("--stage") + 1], "predict")
        self.assertFalse(any(":/ep1" in arg for arg in args))

    def test_comparison_has_no_gpu_device_exposure(self):
        args = self.capture_launcher("scripts/norberto/test1.sh", ["compare", "--help"])
        self.assertIn(f"{ROOT}:/workspace", args)
        self.assertFalse(any(arg.startswith("--device=") for arg in args))
        self.assertEqual(args[args.index("--stage") + 1], "compare")

    def test_qwen_mount_root_and_native_override(self):
        args = self.capture_launcher("scripts/qwen/container.sh", ["--help"], QWEN_GFX_OVERRIDE="native")
        self.assertIn(f"{ROOT}:/workspace", args)
        self.assertIn("-u", args)
        self.assertIn("HSA_OVERRIDE_GFX_VERSION", args)
        self.assertEqual(args[-4:], ["python3", "-m", "qwen_classifier", "--help"])

    def test_svm_evaluation_is_cpu_only(self):
        args = self.capture_launcher("scripts/svm/test1.sh", ["--help"])
        self.assertIn(f"{ROOT}:/workspace", args)
        self.assertIn("--cpus=2", args)
        self.assertFalse(any(arg.startswith("--device=") for arg in args))
        self.assertIn("scripts/svm/evaluate-test1.py", args)
        self.assertEqual(args[-1], "--help")

    def test_rl_training_is_cpu_only_and_offline(self):
        args = self.capture_launcher("scripts/rl/train.sh", ["--dry-run"])
        self.assertFalse(any(arg.startswith("--device=") for arg in args))
        self.assertIn("--network=none", args)
        self.assertIn("scripts/rl/train-development.py", args)
        self.assertEqual(args[-1], "--dry-run")

    def test_python_root_expressions_resolve_to_repository(self):
        checked = 0
        for path in (ROOT / "scripts").rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Subscript):
                    continue
                expression = ast.get_source_segment(path.read_text(), node.value)
                if expression and expression.startswith("Path(__file__).resolve().parents["):
                    self.assertIsInstance(node.value.slice, ast.Constant)
                    self.assertIsInstance(node.value.slice.value, int)
                    actual = path.resolve().parents[node.value.slice.value]
                    self.assertEqual(actual, ROOT, str(path.relative_to(ROOT)))
                    checked += 1
        self.assertGreaterEqual(checked, 6)

    def test_base_restore_compatibility_command_preserves_help(self):
        completed = subprocess.run([sys.executable, str(ROOT / "scripts/prepare-model-base.py"), "--help"],
                                   text=True, capture_output=True, check=True)
        self.assertIn("--verify-only", completed.stdout)


if __name__ == "__main__":
    unittest.main()
