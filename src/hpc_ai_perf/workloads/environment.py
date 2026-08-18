from __future__ import annotations

import importlib.util
import os
import time
from dataclasses import dataclass
from typing import Any

from hpc_ai_perf.config import ExperimentConfig
from hpc_ai_perf.workloads.base import Workload, WorkloadRunOutput


@dataclass(frozen=True)
class EnvironmentValidationWorkload(Workload):
    """Execute the concrete CUDA, distributed, and serving preflight checks."""

    name: str = "environment_validation"

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        target = str(config.workload.parameters.get("validation_target", ""))
        handlers = {
            "cuda_pytorch": self._cuda_pytorch,
            "ddp_nccl_allreduce": self._ddp_nccl_allreduce,
            "fsdp_smoke": self._fsdp_smoke,
            "deepspeed_zero_smoke": self._deepspeed_smoke,
            "vllm_serving": self._vllm_smoke,
            "tensorrt_llm": self._tensorrt_llm_smoke,
        }
        try:
            handler = handlers[target]
        except KeyError as exc:
            raise ValueError(f"Unsupported environment validation target: {target}") from exc
        output = handler(config)
        output["validation_target"] = target
        output["successful_run"] = True
        return WorkloadRunOutput(metrics=output)

    @staticmethod
    def _cuda_pytorch(config: ExperimentConfig) -> dict[str, int | float | str | bool | None]:
        torch = _torch_with_cuda(config)
        expected = int(config.workload.parameters.get("expected_gpus", config.hardware.gpus))
        visible = torch.cuda.device_count()
        if visible != expected:
            raise RuntimeError(f"Expected {expected} visible CUDA GPUs, observed {visible}")
        start = time.perf_counter()
        left = torch.ones(1024, device="cuda")
        right = left * 2
        torch.cuda.synchronize()
        return {
            "gpu_detected": True,
            "visible_gpu_count": visible,
            "cuda_version": str(torch.version.cuda),
            "pytorch_version": str(torch.__version__),
            "smoke_test_runtime_seconds": time.perf_counter() - start,
            "tensor_check": float(right.sum().item()),
        }

    @staticmethod
    def _ddp_nccl_allreduce(config: ExperimentConfig) -> dict[str, int | float | str | bool | None]:
        torch = _torch_with_cuda(config)
        dist, rank, local_rank, world_size = _init_process_group(torch, config)
        expected = int(config.workload.parameters.get("expected_gpus", config.hardware.gpus))
        if world_size != expected:
            raise RuntimeError(f"Expected distributed world size {expected}, observed {world_size}")
        tensor_bytes = int(config.workload.parameters.get("allreduce_bytes", 64 * 1024 * 1024))
        values = torch.ones(max(1, tensor_bytes // 4), dtype=torch.float32, device=local_rank)
        for _ in range(3):
            dist.all_reduce(values)
        torch.cuda.synchronize(local_rank)
        start = time.perf_counter()
        iterations = 10
        for _ in range(iterations):
            dist.all_reduce(values)
        torch.cuda.synchronize(local_rank)
        elapsed = time.perf_counter() - start
        expected_value = float(world_size ** (iterations + 3))
        observed = float(values[0].item())
        if observed != expected_value:
            raise RuntimeError(f"AllReduce produced {observed}, expected {expected_value}")
        algorithmic_bytes = 2 * (world_size - 1) / world_size * tensor_bytes * iterations
        return {
            "allreduce_success": True,
            "allreduce_time_seconds": elapsed / iterations,
            "nccl_bandwidth_gbps": algorithmic_bytes / max(elapsed, 1e-12) / 1e9,
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
        }

    @staticmethod
    def _fsdp_smoke(config: ExperimentConfig) -> dict[str, int | float | str | bool | None]:
        torch = _torch_with_cuda(config)
        _, rank, local_rank, world_size = _init_process_group(torch, config)
        from torch.distributed.fsdp import FullyShardedDataParallel

        model = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.ReLU(), torch.nn.Linear(256, 2)).to(local_rank)
        model = FullyShardedDataParallel(model, device_id=torch.device("cuda", local_rank), use_orig_params=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        inputs = torch.randn(8, 128, device=local_rank)
        labels = torch.randint(0, 2, (8,), device=local_rank)
        torch.cuda.reset_peak_memory_stats(local_rank)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(inputs), labels)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(local_rank)
        return {
            "fsdp_init": True,
            "single_train_step": True,
            "loss": float(loss.detach().cpu()),
            "memory_used_gb": torch.cuda.max_memory_allocated(local_rank) / (1024**3),
            "world_size": world_size,
            "rank": rank,
        }

    @staticmethod
    def _deepspeed_smoke(config: ExperimentConfig) -> dict[str, int | float | str | bool | None]:
        torch = _torch_with_cuda(config)
        if importlib.util.find_spec("deepspeed") is None:
            raise RuntimeError("DeepSpeed is not installed")
        import deepspeed

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        expected = config.distributed.nodes * config.distributed.gpus_per_node
        if world_size != expected:
            raise RuntimeError(f"Expected DeepSpeed world size {expected}, observed {world_size}")
        torch.cuda.set_device(local_rank)
        model = torch.nn.Linear(128, 2).to(local_rank)
        ds_config = {
            "train_micro_batch_size_per_gpu": 8,
            "zero_optimization": {"stage": 2},
            "bf16": {"enabled": True},
        }
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
        inputs = torch.randn(8, 128, device=local_rank)
        labels = torch.randint(0, 2, (8,), device=local_rank)
        loss = torch.nn.functional.cross_entropy(engine(inputs), labels)
        engine.backward(loss)
        engine.step()
        return {
            "deepspeed_launch": True,
            "zero_init": True,
            "single_train_step": True,
            "loss": float(loss.detach().cpu()),
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
        }

    @staticmethod
    def _vllm_smoke(config: ExperimentConfig) -> dict[str, int | float | str | bool | None]:
        _torch_with_cuda(config)
        if importlib.util.find_spec("vllm") is None:
            raise RuntimeError("vLLM is not installed")
        import vllm

        return {"vllm_available": True, "vllm_version": str(getattr(vllm, "__version__", "unknown"))}

    @staticmethod
    def _tensorrt_llm_smoke(config: ExperimentConfig) -> dict[str, int | float | str | bool | None]:
        _torch_with_cuda(config)
        if importlib.util.find_spec("tensorrt_llm") is None:
            raise RuntimeError("TensorRT-LLM is not installed")
        import tensorrt_llm

        return {
            "tensorrt_llm_available": True,
            "tensorrt_llm_version": str(getattr(tensorrt_llm, "__version__", "unknown")),
        }


def _torch_with_cuda(config: ExperimentConfig) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Environment validation requires PyTorch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(f"{config.experiment_id} requires CUDA; CPU fallback is forbidden")
    return torch


def _init_process_group(torch: Any, config: ExperimentConfig) -> tuple[Any, int, int, int]:
    import torch.distributed as dist

    expected = config.distributed.nodes * config.distributed.gpus_per_node
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if expected > 1 and world_size != expected:
        raise RuntimeError(
            f"Expected torchrun world size {expected}, observed {world_size}; "
            "use the generated distributed Slurm script"
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=config.distributed.backend, init_method="env://")
    return dist, rank, local_rank, world_size
