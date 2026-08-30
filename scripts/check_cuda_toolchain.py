#!/usr/bin/env python3
"""Fast CUDA build-toolchain probe for an allocated Lovelace GPU node."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _version_line(command: str) -> str:
    """Return the first non-empty line from a version command."""
    completed = subprocess.run([command, "--version"], check=False, capture_output=True, text=True, timeout=15)
    output = completed.stdout or completed.stderr
    return next((line.strip() for line in output.splitlines() if line.strip()), "version unavailable")


def _report(status: str, name: str, detail: str) -> bool:
    print(f"[{status}] {name}: {detail}")
    return status != "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-gpu-model", default=None)
    parser.add_argument("--expected-gpus", type=int, default=None)
    parser.add_argument("--require-deepspeed", action="store_true")
    args = parser.parse_args()

    passed = True
    nvcc = shutil.which("nvcc")
    if nvcc:
        passed &= _report("PASS", "nvcc", f"{nvcc} ({_version_line(nvcc)})")
    else:
        passed &= _report("FAIL", "nvcc", "not found on PATH")

    cuda_home = os.environ.get("CUDA_HOME")
    cuda_home_nvcc = Path(cuda_home) / "bin" / "nvcc" if cuda_home else None
    if cuda_home_nvcc and cuda_home_nvcc.is_file():
        passed &= _report("PASS", "CUDA_HOME", f"{cuda_home} contains bin/nvcc")
    else:
        detail = "unset" if not cuda_home else f"{cuda_home} does not contain bin/nvcc"
        passed &= _report("FAIL", "CUDA_HOME", detail)

    compiler = shutil.which("clang") or shutil.which("gcc")
    if compiler:
        passed &= _report("PASS", "C/C++ compiler", f"{compiler} ({_version_line(compiler)})")
    else:
        passed &= _report("FAIL", "C/C++ compiler", "neither clang nor gcc is on PATH")

    try:
        import torch
    except Exception as exc:
        _report("FAIL", "torch", f"{type(exc).__name__}: {exc}")
        return 1

    passed &= _report("PASS", "torch", f"{torch.__version__}; CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        _report("FAIL", "CUDA runtime", "torch.cuda.is_available() is false")
        return 1

    device_count = torch.cuda.device_count()
    devices = [torch.cuda.get_device_name(index) for index in range(device_count)]
    passed &= _report("PASS", "CUDA devices", ", ".join(devices))
    if args.expected_gpus is not None and device_count != args.expected_gpus:
        passed &= _report("FAIL", "CUDA device count", f"expected {args.expected_gpus}, observed {device_count}")
    if args.expected_gpu_model and not all(args.expected_gpu_model.lower() in device.lower() for device in devices):
        passed &= _report("FAIL", "GPU model", f"expected {args.expected_gpu_model!r}, observed {devices}")

    if os.environ.get("DS_IGNORE_CUDA_DETECTION") == "1":
        _report("WARN", "DS_IGNORE_CUDA_DETECTION", "enabled; unset it when testing CUDA-op builds")

    if args.require_deepspeed:
        try:
            import deepspeed

            accelerator = deepspeed.accelerator.get_accelerator().device_name()
            passed &= _report("PASS", "deepspeed", f"{deepspeed.__version__}; accelerator {accelerator}")
        except Exception as exc:
            passed &= _report("FAIL", "deepspeed", f"{type(exc).__name__}: {exc}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
