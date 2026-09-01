#!/usr/bin/env python3
"""Validate that a private PyTorch runtime has the required CUDA backends."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


def _normalise(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def is_import_below_environment(import_path: str | Path, environment: str | Path) -> bool:
    """Return whether an imported package file belongs to the private environment."""
    resolved_import = _normalise(import_path)
    resolved_environment = _normalise(environment)
    return resolved_import == resolved_environment or resolved_environment in resolved_import.parents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-environment", required=True)
    parser.add_argument("--expected-version", default="2.8.0")
    parser.add_argument("--expected-cuda", default="12.4")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-gloo", action="store_true")
    parser.add_argument("--require-nccl", action="store_true")
    parser.add_argument(
        "--smoke-process-groups",
        action="store_true",
        help="Create one-rank NCCL and Gloo groups and execute all-reduce operations.",
    )
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    expected_environment = _normalise(args.expected_environment)
    torch_file = _normalise(torch.__file__)
    private_import = is_import_below_environment(torch_file, expected_environment)

    checks = {
        "private_import": (
            private_import,
            f"torch resolves to {torch_file}; expected it below {expected_environment}",
        ),
        "torch_version": (
            torch.__version__.startswith(args.expected_version),
            f"expected {args.expected_version}, observed {torch.__version__}",
        ),
        "torch_cuda_version": (
            torch.version.cuda == args.expected_cuda,
            f"expected {args.expected_cuda}, observed {torch.version.cuda}",
        ),
    }
    if args.require_cuda:
        checks["cuda_available"] = (
            torch.cuda.is_available(),
            "torch.cuda.is_available() must be true",
        )
    if args.require_gloo:
        checks["gloo_available"] = (
            dist.is_gloo_available(),
            "private PyTorch must be built with Gloo",
        )
    if args.require_nccl:
        checks["nccl_available"] = (
            dist.is_nccl_available(),
            "private PyTorch must be built with NCCL",
        )

    passed = True
    for name, (ok, detail) in checks.items():
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        passed &= ok

    if torch.cuda.is_available():
        devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        print(f"[INFO] CUDA devices: {', '.join(devices)}")

    if args.smoke_process_groups:
        prerequisites = torch.cuda.is_available() and dist.is_gloo_available() and dist.is_nccl_available()
        if not prerequisites:
            print("[FAIL] process_group_smoke: CUDA, Gloo, and NCCL must all be available")
            passed = False
        else:
            try:
                torch.cuda.set_device(0)
                with tempfile.TemporaryDirectory(prefix="hpc-ai-torch-store-") as temporary_directory:
                    store_path = Path(temporary_directory) / "store"
                    dist.init_process_group(
                        backend="nccl",
                        init_method=f"file://{store_path}",
                        rank=0,
                        world_size=1,
                    )
                    try:
                        gloo_group = dist.new_group([0], backend="gloo")
                        gpu_value = torch.ones(1, device="cuda")
                        cpu_value = torch.ones(1)
                        dist.all_reduce(gpu_value)
                        dist.all_reduce(cpu_value, group=gloo_group)
                    finally:
                        if dist.is_initialized():
                            dist.destroy_process_group()
                print("[PASS] process_group_smoke: NCCL and Gloo all-reduce completed")
            except Exception as exc:
                print(f"[FAIL] process_group_smoke: {type(exc).__name__}: {exc}")
                passed = False
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
