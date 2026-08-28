from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from hpc_ai_perf.config import ExperimentConfig
from hpc_ai_perf.profiling.base import CompositeProfiler, NullProfiler, Profiler

SUPPORTED_PROFILERS = {
    "dataloader_timing",
    "nccl",
    "nvidia_smi",
    "serving_latency",
    "torch_memory",
    "torch_profiler",
}
WORKLOAD_PROFILER_REQUIRED_METRICS = {
    "dataloader_timing": {"data_loading_seconds", "cpu_to_gpu_transfer_seconds"},
    "serving_latency": {"latency_p50_ms", "latency_p95_ms"},
}


class WorkloadMetricProfiler(Profiler):
    """Declare metrics that are collected inside the workload itself."""

    def __init__(self, name: str) -> None:
        self.name = name

    def stop(self) -> dict[str, int | float | str | bool | None]:
        return {"collector_active": True}


class TorchMemoryProfiler(Profiler):
    """Collect CUDA peak-memory counters around a workload run."""

    name = "torch_memory"

    def __init__(self) -> None:
        self._torch: Any | None = None
        self._device: Any | None = None

    def start(self, config: ExperimentConfig) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised on cluster extras
            raise RuntimeError("torch_memory profiling requires PyTorch") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("torch_memory profiling requires CUDA")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self._torch = torch
        self._device = torch.device("cuda", local_rank)
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"torch_memory profiling requested LOCAL_RANK={local_rank}, "
                f"but only {torch.cuda.device_count()} CUDA device(s) are visible"
            )
        torch.cuda.set_device(self._device)
        torch.cuda.reset_peak_memory_stats(self._device)
        torch.cuda.synchronize(self._device)

    def stop(self) -> dict[str, int | float | str | bool | None]:
        if self._torch is None or self._device is None:
            return {}
        self._torch.cuda.synchronize(self._device)
        allocated = self._torch.cuda.max_memory_allocated(self._device) / (1024**3)
        reserved = self._torch.cuda.max_memory_reserved(self._device) / (1024**3)
        return {
            "memory_used_gb": allocated,
            "memory_reserved_gb": reserved,
        }


class NvidiaSmiProfiler(Profiler):
    """Sample utilization, memory, and power through the nvidia-smi CLI."""

    name = "nvidia_smi"
    _QUERY = "utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm"

    def __init__(self, interval_seconds: float = 0.2) -> None:
        self._interval_seconds = interval_seconds
        self._samples: list[tuple[float, float, float, float, float]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._gpu_count = 1

    def start(self, config: ExperimentConfig) -> None:
        if shutil.which("nvidia-smi") is None:
            raise RuntimeError("nvidia_smi profiling requested, but nvidia-smi is not available")
        self._samples.clear()
        self._stop_event.clear()
        self._started_at = time.perf_counter()
        self._gpu_count = max(1, config.hardware.gpus)
        self._thread = threading.Thread(target=self._sample_loop, name="nvidia-smi-profiler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, int | float | str | bool | None]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval_seconds * 3))
        duration = max(time.perf_counter() - self._started_at, 0.0)
        if not self._samples:
            raise RuntimeError("nvidia-smi returned no valid samples")
        utilizations = [sample[0] for sample in self._samples]
        memory_mib = [sample[1] for sample in self._samples]
        power_watts = [sample[2] for sample in self._samples]
        temperatures = [sample[3] for sample in self._samples]
        clocks_mhz = [sample[4] for sample in self._samples]
        average_power = sum(power_watts) / len(power_watts)
        return {
            "gpu_utilization": sum(utilizations) / len(utilizations),
            "gpu_utilization_max": max(utilizations),
            "memory_used_gb": max(memory_mib) / 1024,
            "power_draw_watts": average_power,
            "power_draw_max_watts": max(power_watts),
            "energy_joules_estimate": average_power * duration,
            "temperature_max_c": max(temperatures),
            "sm_clock_mean_mhz": sum(clocks_mhz) / len(clocks_mhz),
            "profiled_gpu_count": self._gpu_count,
            "samples": len(self._samples),
        }

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--query-gpu={self._QUERY}",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
                if len(rows) >= self._gpu_count:
                    selected_rows = rows[: self._gpu_count]
                    parsed = [[float(value.strip()) for value in row.split(",")] for row in selected_rows]
                    if parsed and all(len(values) == 5 for values in parsed):
                        self._samples.append(
                            (
                                sum(values[0] for values in parsed) / len(parsed),
                                max(values[1] for values in parsed),
                                sum(values[2] for values in parsed),
                                max(values[3] for values in parsed),
                                sum(values[4] for values in parsed) / len(parsed),
                            )
                        )
            except (OSError, subprocess.SubprocessError, ValueError):
                # A transient sample failure should not discard otherwise valid samples.
                pass
            self._stop_event.wait(self._interval_seconds)


class TorchProfiler(Profiler):
    """Capture a PyTorch CPU/CUDA trace for profiling-focused experiments."""

    name = "torch_profiler"

    def __init__(self, output_dir: str | Path, experiment_id: str) -> None:
        self._path = Path(output_dir) / "profiler" / f"{experiment_id}.rank{_rank()}.json"
        self._profiler: Any | None = None

    def start(self, config: ExperimentConfig) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised on cluster extras
            raise RuntimeError("torch_profiler requires PyTorch") from exc
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler = torch.profiler.profile(
            activities=activities,
            profile_memory=True,
            record_shapes=True,
            with_stack=False,
        )
        self._profiler.start()

    def stop(self) -> dict[str, int | float | str | bool | None]:
        if self._profiler is None:
            return {}
        self._profiler.stop()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._profiler.export_chrome_trace(str(self._path))
        events = self._profiler.key_averages()
        cpu_us = sum(float(getattr(event, "self_cpu_time_total", 0.0)) for event in events)
        cuda_us = sum(float(getattr(event, "self_cuda_time_total", 0.0)) for event in events)
        return {
            "trace_path": str(self._path),
            "self_cpu_time_ms": cpu_us / 1000,
            "self_cuda_time_ms": cuda_us / 1000,
        }


def build_profiler(config: ExperimentConfig) -> Profiler:
    """Build exactly the profiler set declared by an experiment config."""
    if not config.profiling.enabled:
        return NullProfiler()

    unknown = sorted(set(config.profiling.profilers) - SUPPORTED_PROFILERS)
    if unknown:
        raise ValueError(f"Unsupported profiler(s): {', '.join(unknown)}")

    profilers: list[Profiler] = []
    for name in config.profiling.profilers:
        if name == "nvidia_smi":
            interval = float(config.profiling.parameters.get("sample_interval_seconds", 0.2))
            profilers.append(NvidiaSmiProfiler(interval))
        elif name == "torch_memory":
            profilers.append(TorchMemoryProfiler())
        elif name == "torch_profiler":
            profilers.append(TorchProfiler(config.runtime.output_dir, config.experiment_id))
        else:
            # NCCL, serving-latency, and DataLoader timings are emitted by the
            # execution path where their timing boundaries are well defined.
            profilers.append(WorkloadMetricProfiler(name))
    return CompositeProfiler(profilers) if profilers else NullProfiler()


def missing_workload_profiler_metrics(
    config: ExperimentConfig, metrics: dict[str, int | float | str | bool | None]
) -> list[str]:
    """Return required in-workload metrics absent from a completed adapter run."""
    requirements = {name: set(required) for name, required in WORKLOAD_PROFILER_REQUIRED_METRICS.items()}
    if config.hardware.gpus > 1 and config.mode in {"training", "profiling"}:
        requirements["nccl"] = {"allreduce_time_seconds", "nccl_bandwidth_gbps"}

    missing: list[str] = []
    for profiler_name in config.profiling.profilers:
        for metric_name in sorted(requirements.get(profiler_name, set())):
            if metrics.get(metric_name) is None:
                missing.append(f"{profiler_name} requires workload metric {metric_name}")
    return missing


def _rank() -> int:
    return int(__import__("os").environ.get("RANK", "0"))
