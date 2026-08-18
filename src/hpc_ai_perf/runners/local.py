from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

from hpc_ai_perf.config import ExperimentConfig
from hpc_ai_perf.profiling import Profiler, build_profiler
from hpc_ai_perf.results import ExperimentResult
from hpc_ai_perf.workloads import WorkloadRegistry, default_registry


class LocalBenchmarkRunner:
    """Run a workload in the current Python process for local validation.

    This runner is used by local smoke tests, Mac dev experiments, and Slurm jobs
    that invoke `python -m hpc_ai_perf.cli run`. It deliberately stays
    single-process: distributed orchestration is represented in configs and
    Slurm rendering, while this class focuses on reliable execution and
    structured result creation.
    """

    def __init__(self, registry: WorkloadRegistry | None = None, profiler: Profiler | None = None) -> None:
        """Create a runner with injectable workload registry and profiler.

        Tests can pass small fake registries or profilers, while normal CLI usage
        gets the default workload registry and a null profiler. This keeps the
        runner easy to test without special hardware.
        """
        self._registry = registry or default_registry()
        self._profiler = profiler

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Execute one config and always return a structured result record.

        Workload preparation, profiler start, and workload execution are all
        inside one guarded block. If anything fails, the exception is captured as
        a readable error string and the result is marked `failed`, which is far
        more useful on HPC than losing the run to an unstructured traceback.
        """
        started_at = _now()
        start = time.perf_counter()
        metrics: dict[str, int | float | str | bool | None] = {}
        error: str | None = None
        status = "completed"
        profiler = self._profiler

        try:
            workload = self._registry.get(config.workload.name)
            profiler = profiler or build_profiler(config)
            # Preparation is allowed to perform dependency checks or lightweight
            # setup before the measured workload starts.
            workload.prepare(config)
            profiler.start(config)
            output = workload.run(config)
            metrics.update(output.metrics)
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            # Stop profilers even after workload failure so partial diagnostic
            # metrics can still be written to the raw result record.
            if profiler is not None:
                try:
                    profiler_metrics = profiler.stop()
                    metrics.update(profiler_metrics)
                    _promote_standard_profiler_metrics(metrics, profiler_metrics)
                except Exception as exc:
                    status = "failed"
                    profiler_error = f"{type(exc).__name__}: {exc}"
                    error = f"{error}; profiler stop failed: {profiler_error}" if error else profiler_error
            cleanup_error = _cleanup_process_group()
            if cleanup_error is not None:
                status = "failed"
                error = f"{error}; distributed cleanup failed: {cleanup_error}" if error else cleanup_error

        completed_at = _now()
        runtime_seconds = time.perf_counter() - start
        return ExperimentResult.from_config(
            config,
            status=status,  # type: ignore[arg-type]
            started_at=started_at,
            completed_at=completed_at,
            runtime_seconds=runtime_seconds,
            metrics=metrics,
            environment=_environment(config),
            error=error,
        )


def _now() -> str:
    """Return a timezone-aware UTC timestamp for result records."""
    return datetime.now(UTC).isoformat()


def _environment(config: ExperimentConfig) -> dict[str, int | float | str | bool | None]:
    """Capture lightweight host metadata alongside every run."""
    environment: dict[str, int | float | str | bool | None] = {
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "requested_platform": config.hardware.platform,
        "requested_gpu_model": config.hardware.gpu_model,
        "command": " ".join(sys.argv),
        "config_sha256": hashlib.sha256(
            json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        environment.update(
            {
                "torch_version": str(torch.__version__),
                "cuda_version": str(torch.version.cuda),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
            }
        )
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            device_index = min(local_rank, torch.cuda.device_count() - 1)
            properties = torch.cuda.get_device_properties(device_index)
            environment.update(
                {
                    "observed_gpu_model": str(properties.name),
                    "observed_gpu_memory_gb": float(properties.total_memory / (1024**3)),
                    "gpu_compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
            if hasattr(torch.cuda, "nccl") and hasattr(torch.cuda.nccl, "version"):
                environment["nccl_version"] = str(torch.cuda.nccl.version())
            environment["cudnn_version"] = str(torch.backends.cudnn.version())
    except ImportError:
        environment["torch_version"] = None

    environment["git_commit"] = _git_value(["git", "rev-parse", "HEAD"])
    environment["git_dirty"] = bool(_git_value(["git", "status", "--porcelain"]))
    environment.update(_nvidia_smi_metadata())
    for package in ["datasets", "deepspeed", "transformers", "vllm"]:
        try:
            environment[f"{package}_version"] = version(package)
        except PackageNotFoundError:
            environment[f"{package}_version"] = None
    return environment


def _cleanup_process_group() -> str | None:
    """Destroy a process group created by a distributed experiment."""
    try:
        import torch.distributed as dist
    except ImportError:
        return None
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception as exc:  # pragma: no cover - depends on cluster failure state
        return f"{type(exc).__name__}: {exc}"
    return None


def _git_value(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _nvidia_smi_metadata() -> dict[str, str | None]:
    """Capture immutable GPU identity and driver details when available."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
        if not rows:
            return {}
        values = [value.strip() for value in rows[min(local_rank, len(rows) - 1)].split(",")]
        if len(values) != 2:
            return {}
        return {"observed_gpu_uuid": values[0], "nvidia_driver_version": values[1]}
    except (OSError, subprocess.SubprocessError):
        return {}


def _promote_standard_profiler_metrics(
    metrics: dict[str, int | float | str | bool | None],
    profiler_metrics: dict[str, int | float | str | bool | None],
) -> None:
    """Expose standard profiler fields at the stable top-level metric names."""
    standard_names = {
        "energy_joules_estimate",
        "gpu_utilization",
        "memory_reserved_gb",
        "memory_used_gb",
        "power_draw_watts",
    }
    for key, value in profiler_metrics.items():
        suffix = key.rsplit(".", 1)[-1]
        if suffix in standard_names:
            metrics[suffix] = value
