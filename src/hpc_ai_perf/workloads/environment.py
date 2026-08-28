from __future__ import annotations

import importlib.util
import os
import time
from dataclasses import dataclass
from typing import Any

from hpc_ai_perf.config import ExperimentConfig
from hpc_ai_perf.workloads.base import Workload, WorkloadRunOutput
from hpc_ai_perf.workloads.tensorrt_llm import validate_tensorrt_llm_environment


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
        expected_checks = config.workload.parameters.get("expected_checks", [])
        missing_checks = [str(check) for check in expected_checks if not output.get(str(check))]
        if missing_checks:
            raise RuntimeError(f"Environment validation did not satisfy checks: {', '.join(missing_checks)}")
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
            "cuda_visible": True,
            "torch_import": True,
            "gpu_count": visible,
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
            "torchrun_launch": world_size > 1,
            "process_group_init": dist.is_initialized(),
            "allreduce": True,
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

        device = torch.device("cuda", local_rank)
        model = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.ReLU(), torch.nn.Linear(256, 2)).to(device)
        model = FullyShardedDataParallel(model, device_id=device, use_orig_params=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        inputs = torch.randn(8, 128, device=device)
        labels = torch.randint(0, 2, (8,), device=device)
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(inputs), labels)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        return {
            "torchrun_launch": world_size > 1,
            "fsdp_init": True,
            "single_train_step": True,
            "loss": float(loss.detach().cpu()),
            "memory_used_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
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
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        model = torch.nn.Linear(128, 2).to(device)
        ds_config = {
            "train_micro_batch_size_per_gpu": 8,
            "zero_optimization": {"stage": 2},
            "bf16": {"enabled": True},
        }
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
        inputs = torch.randn(8, 128, device=device)
        labels = torch.randint(0, 2, (8,), device=device)
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
        torch = _torch_with_cuda(config)
        if importlib.util.find_spec("vllm") is None:
            raise RuntimeError("vLLM is not installed")
        try:
            import vllm
            from vllm import LLM, SamplingParams
        except (ImportError, OSError) as exc:
            torch_cuda = getattr(getattr(torch, "version", None), "cuda", "unknown")
            raise RuntimeError(
                "vLLM import failed before engine startup; the installed vLLM wheel is likely incompatible with "
                f"the active CUDA/PyTorch runtime (torch CUDA {torch_cuda}). Reinstall or build vLLM for the "
                f"cluster CUDA/driver stack. Original import error: {exc}"
            ) from exc

        parameters = config.workload.parameters
        model_name = str(parameters["hf_model_name"])
        model_revision = str(parameters["hf_model_revision"])
        max_new_tokens = int(parameters.get("max_new_tokens", 2))
        prompt_token_id = int(parameters.get("prompt_token_id", 15496))
        dtype = "bfloat16" if config.runtime.precision == "bf16" else "float16"
        engine = LLM(
            model=model_name,
            revision=model_revision,
            dtype=dtype,
            tensor_parallel_size=1,
            max_model_len=64,
            enforce_eager=True,
        )
        sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        prompt_token_ids = [[prompt_token_id]]
        engine.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling, use_tqdm=False)
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = engine.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - started, 1e-12)
        generated_tokens = sum(len(candidate.token_ids) for output in outputs for candidate in output.outputs)
        if generated_tokens < 1:
            raise RuntimeError("vLLM engine returned no generated tokens")
        return {
            "vllm_available": True,
            "vllm_import": True,
            "vllm_version": str(getattr(vllm, "__version__", "unknown")),
            "engine_start": True,
            "request_served": True,
            "requests": len(outputs),
            "generated_tokens": generated_tokens,
            "total_tokens": generated_tokens,
            "throughput_unit": "generated_tokens_per_second",
            "measured_seconds": elapsed,
            "gpu_seconds": elapsed,
            "gpu_hours": elapsed / 3600,
            "throughput_tokens_sec": generated_tokens / elapsed,
            "latency_p50_ms": elapsed * 1000,
            "latency_p95_ms": elapsed * 1000,
            "time_to_first_token_p50_ms": elapsed * 1000,
            "framework_observed": "vllm",
            "hf_model_name": model_name,
            "hf_model_revision": model_revision,
        }

    @staticmethod
    def _tensorrt_llm_smoke(config: ExperimentConfig) -> dict[str, int | float | str | bool | None]:
        torch = _torch_with_cuda(config)
        if importlib.util.find_spec("tensorrt_llm") is None:
            raise RuntimeError("TensorRT-LLM is not installed")
        import tensorrt_llm

        return validate_tensorrt_llm_environment(config, torch=torch, tensorrt_llm=tensorrt_llm)


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
