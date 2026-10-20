#!/usr/bin/env python3
"""Bounded two-GPU diagnostic; no model weights or benchmark evidence."""

import argparse
import importlib
import json
import os
import platform
import sys
import tempfile
from datetime import timedelta
from pathlib import Path


def worker(rank, rendezvous):
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    matrix = torch.ones((64, 64), device=device)
    assert torch.all(matrix @ matrix == 64).item(), "CUDA computation failed"
    torch.cuda.synchronize(device)
    for backend in ("gloo", "nccl"):
        dist.init_process_group(
            backend,
            init_method=f"file://{rendezvous}-{backend}",
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=60),
        )
        try:
            value = torch.tensor([rank + 1.0], device=device if backend == "nccl" else "cpu")
            dist.all_reduce(value)
            assert value.item() == 3.0, f"{backend} all_reduce failed"
        finally:
            dist.destroy_process_group()
    print(f"Rank {rank}: CUDA, Gloo and NCCL passed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "python": sys.executable,
        "host": platform.node(),
        "library_path": os.environ.get("LD_LIBRARY_PATH"),
        "benchmark_evidence": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        import site

        site.addsitedir(os.environ["HPC_RECOVERY_SITE"])
        import torch
        import torch.distributed as dist
        import torch.multiprocessing as mp
        import vllm

        report.update(
            torch_version=torch.__version__,
            torch_path=torch.__file__,
            cuda_version=torch.version.cuda,
            vllm_version=vllm.__version__,
            vllm_path=vllm.__file__,
            gpu_count=torch.cuda.device_count(),
        )
        if torch.cuda.device_count() != 2:
            raise RuntimeError("Expected exactly two allocated GPUs")
        if not dist.is_gloo_available() or not dist.is_nccl_available():
            raise RuntimeError("Gloo and NCCL must both be available")
        importlib.import_module("vllm._C")
        report["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(2)]
        with tempfile.TemporaryDirectory(prefix="hpc-recovery-") as directory:
            mp.spawn(worker, args=(str(Path(directory) / "rendezvous"),), nprocs=2, join=True)
        report["status"] = "passed"
        report["checks"] = ["vllm._C import", "CUDA computation", "Gloo CPU all_reduce", "NCCL GPU all_reduce"]
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Recovery smoke test passed: {args.output}")


if __name__ == "__main__":
    main()
