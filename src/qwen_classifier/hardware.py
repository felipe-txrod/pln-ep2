"""Run real accelerator operations in disposable subprocesses, never on CPU."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from gpt_classifier.data import canonical, digest, write_json


def environment() -> dict:
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "numpy", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    pci_devices = []
    for path in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
        if not re.fullmatch(r"card\d+", path.parent.name):
            continue
        pci_devices.append({name: (path / name).read_text().strip()
                            for name in ("vendor", "device") if (path / name).exists()})
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "pci_devices": pci_devices,
            "device_nodes": {name: Path(name).exists() for name in ("/dev/kfd", "/dev/dri")},
            "variables": {name: os.environ.get(name) for name in (
                "HSA_OVERRIDE_GFX_VERSION", "AMDGPU_TARGETS", "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF")},
            "implementation_sha256": digest(canonical({
                str(path.relative_to(Path(__file__).parents[1])): digest(path.read_text(encoding="utf-8"))
                for path in [*sorted(Path(__file__).parent.glob("*.py")),
                             Path(__file__).parents[1] / "gpt_classifier" / "data.py",
                             Path(__file__).parents[1] / "gpt_classifier" / "delivery.py"]}))}


def runtime_identity() -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU is available. Expose /dev/kfd and /dev/dri and install ROCm PyTorch.")
    properties = torch.cuda.get_device_properties(0)
    return {"name": properties.name, "architecture": getattr(properties, "gcnArchName", "unknown"),
            "hip": torch.version.hip, "cuda": torch.version.cuda,
            "total_vram_mib": properties.total_memory / 2**20}


def memory() -> dict:
    import torch
    return {"peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "free_vram_mib": torch.cuda.mem_get_info()[0] / 2**20}


def _operation(kind: str) -> dict:
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(42)
    device = runtime_identity()
    if not device["hip"]:
        raise RuntimeError("This experiment requires ROCm-enabled PyTorch, not a CUDA or CPU wheel.")
    torch.cuda.reset_peak_memory_stats()
    if kind == "fp16":
        a = torch.randn(128, 128, dtype=torch.float32) / 8
        b = torch.randn(128, 128, dtype=torch.float32) / 8
        actual = a.half().cuda() @ b.half().cuda()
        expected = a.half().float() @ b.half().float()
        torch.testing.assert_close(actual.float().cpu(), expected, atol=0.002, rtol=0.02)
        q = torch.randn(1, 4, 32, 32, dtype=torch.float16, device="cuda", requires_grad=True)
        output = functional.scaled_dot_product_attention(q, q, q, is_causal=True)
        reference_q = q.detach().float().cpu().requires_grad_()
        reference = functional.scaled_dot_product_attention(reference_q, reference_q, reference_q, is_causal=True)
        torch.testing.assert_close(output.float().cpu(), reference, atol=0.008, rtol=0.03)
        output.float().square().mean().backward()
        reference.square().mean().backward()
        torch.testing.assert_close(q.grad.float().cpu(), reference_q.grad, atol=0.0001, rtol=0.08)
        details = {"gemm_max_abs_error": (actual.float().cpu() - expected).abs().max().item(),
                   "sdpa_forward_backward": True}
    else:
        import bitsandbytes as bnb
        # Exercise nested NF4 and input gradients, as required by checkpointed LoRA.
        layer = bnb.nn.Linear4bit(128, 128, bias=False, compute_dtype=torch.float16,
                                 compress_statistics=True, quant_type="nf4").cuda()
        adapter_a = torch.nn.Parameter(torch.randn(128, 4, device="cuda", dtype=torch.float32) * 0.01)
        adapter_b = torch.nn.Parameter(torch.zeros(4, 128, device="cuda", dtype=torch.float32))
        optimizer = torch.optim.AdamW([adapter_a, adapter_b], lr=0.01, foreach=False)
        x = torch.randn(2, 16, 128, device="cuda", dtype=torch.float16, requires_grad=True)
        y = layer(x) + ((x.float() @ adapter_a) @ adapter_b).half()
        state = layer.weight.quant_state
        if state is None or state.quant_type != "nf4" or not state.nested:
            raise RuntimeError("The backend did not create nested NF4 GPU weights.")
        dequantized = bnb.functional.dequantize_4bit(layer.weight.data, state)
        reference = functional.linear(x.detach(), dequantized.half())
        torch.testing.assert_close(y, reference, atol=0.02, rtol=0.05)
        y.float().square().mean().backward()
        for gradient in (x.grad, adapter_a.grad, adapter_b.grad):
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError("NF4 backward produced missing or nonfinite gradients.")
        before = adapter_b.detach().clone()
        optimizer.step()
        if torch.equal(before, adapter_b):
            raise RuntimeError("The NF4 LoRA surrogate did not update its adapter.")
        details = {"nf4": True, "double_quantization": True, "backward": True, "optimizer_update": True}
    torch.cuda.synchronize()
    return {"status": "passed", "device": device, **details, **memory()}


def isolated_probe(kind: str, *, timeout: int = 180) -> dict:
    with tempfile.TemporaryDirectory(prefix="ep2-probe-") as temporary:
        output = Path(temporary) / "probe.json"
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "qwen_classifier.hardware", kind, str(output)],
                capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return {"status": "failed", "error": f"{kind} kernel probe exceeded {timeout} seconds."}
        if output.exists():
            result = json.loads(output.read_text())
        else:
            result = {"status": "failed", "error": "Worker exited before writing a report."}
        return {**result, "returncode": completed.returncode, "stderr_tail": completed.stderr[-4000:]}


def probe() -> dict:
    current = environment()
    fp16 = isolated_probe("fp16")
    nf4 = isolated_probe("nf4") if fp16["status"] == "passed" else {
        "status": "skipped", "error": "FP16 GPU kernels failed; NF4 cannot be validated."}
    return {"environment": current, "environment_sha256": digest(canonical(current)),
            "fp16": fp16, "qlora": nf4,
            "recommended_mode": "qlora" if nf4["status"] == "passed" else
                                "fp16" if fp16["status"] == "passed" else None,
            "support_caveat": "Kernel success is empirical compatibility, not official gfx1032 or Fedora support."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("fp16", "nf4"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        report = _operation(args.kind)
    except Exception as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    write_json(args.output, report)
    raise SystemExit(0 if report["status"] == "passed" else 1)
