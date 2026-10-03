"""Untimed input preparation and placement evidence for scaling studies."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any


def scaling_runtime_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in ("workloads/scaling.py", "workloads/transformer.py", "config/scaling.py", "profiling/measurement.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


class GlobalTrainingInputs:
    """The same ordered global examples, partitioned without duplication.

    Allocate and copy the entire small synthetic input stream before warm-up.
    Neither RNG, host-to-device copies nor hashing enter the measured loop.
    A fresh batch is consumed on every optimizer update, including warm-up.
    """

    def __init__(self, torch, config, runtime, device, vocab_size: int):
        p = config.workload.parameters
        self.micro_batch = p["per_gpu_batch_size"]
        self.accumulation = p["gradient_accumulation_steps"]
        global_batch = p["global_batch_size"]
        if global_batch != self.micro_batch * self.accumulation * runtime.world_size:
            raise ValueError("Observed world size violates the controlled global-batch contract")
        updates = p["warmup_steps"] + config.runtime.max_steps
        generator = torch.Generator(device="cpu").manual_seed(config.runtime.seed)
        inputs = torch.randint(0, vocab_size, (updates, global_batch, p["sequence_length"]), generator=generator)
        self.sha256 = hashlib.sha256(inputs.numpy().tobytes()).hexdigest()
        rank_batch = self.micro_batch * self.accumulation
        start = runtime.rank * rank_batch
        self.inputs = inputs[:, start : start + rank_batch].contiguous().to(device)
        self.index = 0
        self.batches = updates * self.accumulation

    def next_batch(self):
        if self.index >= self.batches:
            raise RuntimeError("Controlled global input stream exhausted")
        update, micro = divmod(self.index, self.accumulation)
        self.index += 1
        start = micro * self.micro_batch
        tokens = self.inputs[update, start : start + self.micro_batch]
        return {"input_ids": tokens, "labels": tokens}


def capture_training_placement(torch: Any, runtime: Any) -> dict[str, Any]:
    from hpc_ai_perf.workloads.environment import _capture_gpu_topology

    props = torch.cuda.get_device_properties(runtime.local_rank)
    placement = {
        "rank": runtime.rank,
        "local_rank": runtime.local_rank,
        "hostname": socket.gethostname(),
        "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    ranks = [placement]
    if runtime.distributed:
        ranks = [None] * runtime.world_size
        torch.distributed.all_gather_object(ranks, placement)
    if runtime.rank != 0:
        return {}
    metrics = {**_capture_gpu_topology(), "training_rank_placement": json.dumps(ranks, sort_keys=True)}
    # Scheduler context is diagnostic and may be unavailable on non-Slurm hosts.
    try:
        result = subprocess.run(
            ["squeue", "--noheader", "--nodes", socket.gethostname(), "--format=%i|%t|%b|%C"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        metrics["scheduler_node_jobs_at_start"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        metrics["scheduler_node_jobs_at_start"] = f"unavailable: {type(exc).__name__}"
    return metrics
