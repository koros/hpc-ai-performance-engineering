from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _NvidiaSmiSample:
    timestamp: float
    utilization: float
    memory_mib: float
    power_watts: float
    temperature_c: float
    sm_clock_mhz: float


class NvidiaSmiProfiler(Profiler):
    """Sample utilization, memory, and power through the nvidia-smi CLI."""

    name = "nvidia_smi"
    measurement_stop_priority = 100
    _QUERY = "uuid,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm"

    def __init__(self, interval_seconds: float = 0.2) -> None:
        self._interval_seconds = interval_seconds
        self._samples: list[_NvidiaSmiSample] = []
        self._samples_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._collection_stopped_at: float | None = None
        self._measurement_started_at: float | None = None
        self._measurement_stopped_at: float | None = None
        self._gpu_count = 1
        self._target_gpu_uuids: tuple[str, ...] = ()
        self._profiled_gpu_uuids: tuple[str, ...] = ()
        self._mapping_method = ""

    def start(self, config: ExperimentConfig) -> None:
        if shutil.which("nvidia-smi") is None:
            raise RuntimeError("nvidia_smi profiling requested, but nvidia-smi is not available")
        self._samples.clear()
        self._stop_event.clear()
        self._started_at = time.perf_counter()
        self._collection_stopped_at = None
        self._measurement_started_at = None
        self._measurement_stopped_at = None
        self._gpu_count = max(1, config.hardware.gpus)
        self._target_gpu_uuids = visible_cuda_device_uuids(self._gpu_count)
        self._profiled_gpu_uuids = ()
        self._mapping_method = ""
        # Resolve and sample the devices synchronously so an ambiguous mapping
        # fails before the benchmark starts instead of silently profiling an
        # unrelated GPU on a shared node.
        self._sample_once()
        self._thread = threading.Thread(target=self._sample_loop, name="nvidia-smi-profiler", daemon=True)
        self._thread.start()

    def begin_measurement(self) -> None:
        if self._measurement_started_at is not None:
            raise RuntimeError("nvidia-smi measured region was opened more than once")
        self._measurement_started_at = time.perf_counter()

    def end_measurement(self) -> None:
        if self._measurement_started_at is None or self._measurement_stopped_at is not None:
            return
        # Freeze the boundary before stopping the sampling thread. Any query
        # already in flight may finish later, but timestamp filtering excludes
        # it from measured-region metrics.
        self._measurement_stopped_at = time.perf_counter()
        self._halt_sampling()

    def stop(self) -> dict[str, int | float | str | bool | None]:
        self._halt_sampling()
        stopped_at = self._collection_stopped_at or time.perf_counter()
        with self._samples_lock:
            samples = list(self._samples)
        if not samples:
            raise RuntimeError("nvidia-smi returned no valid samples")

        full_window = _summarize_nvidia_smi_window(samples, self._started_at, stopped_at)
        metrics: dict[str, int | float | str | bool | None] = {
            # Backward-compatible aliases retain their historical full-window
            # meaning. Derived energy analysis uses only the explicitly scoped
            # measured-region fields added below.
            "gpu_utilization": full_window["gpu_utilization"],
            "gpu_utilization_max": full_window["gpu_utilization_max"],
            "memory_used_gb": full_window["memory_used_gb"],
            "power_draw_watts": full_window["power_draw_watts"],
            "power_draw_max_watts": full_window["power_draw_max_watts"],
            "energy_joules_estimate": full_window["energy_joules"],
            "temperature_max_c": full_window["temperature_max_c"],
            "sm_clock_mean_mhz": full_window["sm_clock_mean_mhz"],
            "gpu_utilization_full_window": full_window["gpu_utilization"],
            "memory_used_gb_full_window": full_window["memory_used_gb"],
            "power_draw_watts_full_window": full_window["power_draw_watts"],
            "power_draw_max_watts_full_window": full_window["power_draw_max_watts"],
            "energy_joules_full_window": full_window["energy_joules"],
            "full_window_duration_seconds": full_window["duration_seconds"],
            "profiled_gpu_count": self._gpu_count,
            "profiled_gpu_uuids": ",".join(self._profiled_gpu_uuids),
            "gpu_mapping_method": self._mapping_method,
            "telemetry_schema_version": "3",
            "memory_metric_scope": "max_per_gpu",
            "power_metric_scope": "sum_visible_gpus",
            "utilization_metric_scope": "mean_visible_gpus",
            "samples": len(samples),
            "measurement_status": "not_recorded",
            "energy_scope": "full_window",
        }

        if self._measurement_started_at is not None and self._measurement_stopped_at is not None:
            measured = _summarize_nvidia_smi_window(
                samples,
                self._measurement_started_at,
                self._measurement_stopped_at,
                require_two_observations=True,
            )
            metrics["measurement_duration_seconds"] = max(
                self._measurement_stopped_at - self._measurement_started_at, 0.0
            )
            metrics["measurement_samples"] = measured["source_samples"]
            if measured["sufficient"]:
                metrics.update(
                    {
                        "gpu_utilization_measured_region": measured["gpu_utilization"],
                        "gpu_utilization_max_measured_region": measured["gpu_utilization_max"],
                        "memory_used_gb_measured_region": measured["memory_used_gb"],
                        "power_draw_watts_measured_region": measured["power_draw_watts"],
                        "power_draw_max_watts_measured_region": measured["power_draw_max_watts"],
                        "energy_joules_measured_region": measured["energy_joules"],
                        "temperature_max_c_measured_region": measured["temperature_max_c"],
                        "sm_clock_mean_mhz_measured_region": measured["sm_clock_mean_mhz"],
                        "measurement_status": "sufficient",
                        "energy_scope": "measured_region",
                    }
                )
            else:
                metrics["measurement_status"] = "insufficient_samples"
        return metrics

    def _halt_sampling(self) -> None:
        if self._collection_stopped_at is None:
            self._collection_stopped_at = time.perf_counter()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval_seconds * 3))

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            # A transient sample failure should not discard otherwise valid samples.
            with suppress(OSError, subprocess.SubprocessError, ValueError):
                self._sample_once()
            self._stop_event.wait(self._interval_seconds)

    def _sample_once(self) -> None:
        rows = query_nvidia_smi_rows(self._QUERY, timeout=10)
        selected_rows, mapping_method = select_nvidia_smi_rows(
            rows,
            target_gpu_uuids=self._target_gpu_uuids,
            expected_gpu_count=self._gpu_count,
        )
        parsed: list[tuple[str, list[float]]] = []
        for row in selected_rows:
            values = [value.strip() for value in row.split(",")]
            if len(values) != 6:
                raise ValueError(f"nvidia-smi returned an unexpected telemetry row: {row}")
            parsed.append((values[0], [float(value) for value in values[1:]]))
        if not parsed:
            raise ValueError("nvidia-smi returned no telemetry rows for the allocated CUDA devices")
        if self._mapping_method and self._mapping_method != mapping_method:
            raise ValueError("nvidia-smi GPU mapping method changed during profiling")
        self._mapping_method = mapping_method
        self._profiled_gpu_uuids = tuple(uuid for uuid, _ in parsed)
        samples = [values for _, values in parsed]
        sample = _NvidiaSmiSample(
            timestamp=time.perf_counter(),
            utilization=sum(values[0] for values in samples) / len(samples),
            memory_mib=max(values[1] for values in samples),
            power_watts=sum(values[2] for values in samples),
            temperature_c=max(values[3] for values in samples),
            sm_clock_mhz=sum(values[4] for values in samples) / len(samples),
        )
        with self._samples_lock:
            self._samples.append(sample)


def _summarize_nvidia_smi_window(
    samples: list[_NvidiaSmiSample],
    started_at: float,
    stopped_at: float,
    *,
    require_two_observations: bool = False,
) -> dict[str, float | int | bool]:
    """Summarize timestamped samples and integrate power over an exact interval."""
    duration = max(stopped_at - started_at, 0.0)
    ordered = sorted(samples, key=lambda sample: sample.timestamp)
    source = [sample for sample in ordered if started_at <= sample.timestamp <= stopped_at]
    previous = [sample for sample in ordered if sample.timestamp < started_at]
    following = [sample for sample in ordered if sample.timestamp > stopped_at]
    anchors = list(source)
    if previous:
        anchors.insert(0, previous[-1])
    if following:
        anchors.append(following[0])
    if not anchors:
        anchors = ordered[:1]

    # A boundary projection uses the nearest sampled value while integration
    # between real observations uses the trapezoidal rule.
    projected: list[tuple[float, _NvidiaSmiSample]] = []
    if anchors:
        start_value = max(
            (sample for sample in ordered if sample.timestamp <= started_at),
            default=anchors[0],
            key=lambda item: item.timestamp,
        )
        end_value = max(
            (sample for sample in ordered if sample.timestamp <= stopped_at),
            default=anchors[-1],
            key=lambda item: item.timestamp,
        )
        projected.append((started_at, start_value))
        projected.extend((sample.timestamp, sample) for sample in source if started_at < sample.timestamp < stopped_at)
        projected.append((stopped_at, end_value))

    observation_count = len({sample.timestamp for sample in anchors})
    sufficient = bool(projected) and duration > 0 and (not require_two_observations or observation_count >= 2)
    relevant = source or anchors
    if not relevant:
        raise RuntimeError("nvidia-smi returned no samples in the profiling window")
    energy_joules = 0.0
    for (left_time, left), (right_time, right) in zip(projected, projected[1:], strict=False):
        energy_joules += (left.power_watts + right.power_watts) * 0.5 * max(right_time - left_time, 0.0)
    return {
        "sufficient": sufficient,
        "source_samples": observation_count,
        "duration_seconds": duration,
        "gpu_utilization": sum(sample.utilization for sample in relevant) / len(relevant),
        "gpu_utilization_max": max(sample.utilization for sample in relevant),
        "memory_used_gb": max(sample.memory_mib for sample in relevant) / 1024,
        "power_draw_watts": energy_joules / duration if duration > 0 else 0.0,
        "power_draw_max_watts": max(sample.power_watts for sample in relevant),
        "energy_joules": energy_joules,
        "temperature_max_c": max(sample.temperature_c for sample in relevant),
        "sm_clock_mean_mhz": sum(sample.sm_clock_mhz for sample in relevant) / len(relevant),
    }


def visible_cuda_device_uuids(expected_gpu_count: int | None = None) -> tuple[str, ...]:
    """Return UUIDs for CUDA-visible devices in CUDA runtime order when exposed by Torch."""
    try:
        import torch
    except ImportError:
        return ()
    if not torch.cuda.is_available():
        return ()
    visible_count = int(torch.cuda.device_count())
    requested_count = visible_count if expected_gpu_count is None else expected_gpu_count
    if visible_count < requested_count:
        raise RuntimeError(
            f"CUDA exposes {visible_count} device(s), but telemetry requires {requested_count} allocated device(s)"
        )
    uuids: list[str] = []
    for index in range(requested_count):
        value = getattr(torch.cuda.get_device_properties(index), "uuid", None)
        normalized = _format_cuda_uuid(value)
        if normalized is None:
            return _uuid_tokens_from_cuda_visible_devices(requested_count)
        uuids.append(normalized)
    return tuple(uuids)


def query_nvidia_smi_rows(query: str, *, timeout: int = 5) -> list[str]:
    """Query nvidia-smi and return non-empty CSV rows."""
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return [row.strip() for row in completed.stdout.splitlines() if row.strip()]


def select_nvidia_smi_rows(
    rows: list[str], *, target_gpu_uuids: tuple[str, ...], expected_gpu_count: int
) -> tuple[list[str], str]:
    """Select only allocated GPU rows, preferring an explicit CUDA-to-NVML UUID match."""
    if target_gpu_uuids:
        by_uuid = {_normalize_gpu_uuid(row.split(",", 1)[0]): row for row in rows}
        missing = [uuid for uuid in target_gpu_uuids if _normalize_gpu_uuid(uuid) not in by_uuid]
        if missing:
            raise ValueError(f"nvidia-smi did not return allocated CUDA GPU UUID(s): {', '.join(missing)}")
        selected = [by_uuid[_normalize_gpu_uuid(uuid)] for uuid in target_gpu_uuids]
        return selected, "cuda_uuid"
    if len(rows) == expected_gpu_count:
        # Some Slurm/cgroup deployments filter nvidia-smi to exactly the job's
        # devices but do not expose UUIDs through Torch. Exact row-count equality
        # is the only safe fallback; never take the first N rows from a larger set.
        return rows, "cgroup_visible_rows"
    raise ValueError(
        "cannot prove nvidia-smi device ownership: Torch exposed no CUDA UUIDs "
        f"and nvidia-smi returned {len(rows)} row(s) for {expected_gpu_count} requested GPU(s)"
    )


def nvidia_smi_device_metadata() -> dict[str, str | None]:
    """Capture the current CUDA rank's immutable UUID and driver using verified mapping."""
    try:
        import torch

        visible_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if visible_count < 1:
            return {}
        target_uuids = visible_cuda_device_uuids(visible_count)
        rows = query_nvidia_smi_rows("uuid,driver_version")
        selected, mapping_method = select_nvidia_smi_rows(
            rows,
            target_gpu_uuids=target_uuids,
            expected_gpu_count=visible_count,
        )
        local_rank = min(int(os.environ.get("LOCAL_RANK", "0")), len(selected) - 1)
        values = [value.strip() for value in selected[local_rank].split(",")]
        if len(values) != 2:
            return {}
        return {
            "observed_gpu_uuid": values[0],
            "nvidia_driver_version": values[1],
            "observed_gpu_uuid_mapping_method": mapping_method,
        }
    except (ImportError, OSError, RuntimeError, subprocess.SubprocessError, ValueError):
        return {}


def _uuid_tokens_from_cuda_visible_devices(expected_gpu_count: int) -> tuple[str, ...]:
    tokens = [token.strip() for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if token.strip()]
    if len(tokens) < expected_gpu_count:
        return ()
    selected = tokens[:expected_gpu_count]
    if all(token.upper().startswith(("GPU-", "MIG-")) for token in selected):
        return tuple(selected)
    return ()


def _format_cuda_uuid(value: object) -> str | None:
    if isinstance(value, bytes) and len(value) == 16:
        hex_value = value.hex()
        return f"GPU-{hex_value[:8]}-{hex_value[8:12]}-{hex_value[12:16]}-{hex_value[16:20]}-{hex_value[20:]}"
    text = str(value or "").strip()
    if text.upper().startswith(("GPU-", "MIG-")):
        return text
    return None


def _normalize_gpu_uuid(value: str) -> str:
    return value.strip().casefold()


class TorchProfiler(Profiler):
    """Capture a PyTorch CPU/CUDA trace for profiling-focused experiments."""

    name = "torch_profiler"

    def __init__(self, output_dir: str | Path, experiment_id: str, *, repetition_number: int | None = None) -> None:
        trial = repetition_number
        if trial is None:
            raw_trial = os.environ.get("SLURM_ARRAY_TASK_ID")
            trial = int(raw_trial) if raw_trial and raw_trial.isdigit() else None
        trial_name = f"r{trial:03d}" if trial is not None else "single"
        self._path = Path(output_dir) / "profiler" / experiment_id / trial_name / f"rank{_rank()}.json"
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


def build_profiler(config: ExperimentConfig, *, repetition_number: int | None = None) -> Profiler:
    """Build exactly the profiler set declared by an experiment config."""
    if not config.profiling.enabled:
        return NullProfiler()

    unknown = sorted(set(config.profiling.profilers) - SUPPORTED_PROFILERS)
    if unknown:
        raise ValueError(f"Unsupported profiler(s): {', '.join(unknown)}")

    profilers: list[Profiler] = []
    rank = int(os.environ.get("RANK", "0"))
    for name in config.profiling.profilers:
        if name == "nvidia_smi":
            # nvidia-smi observes every GPU allocated to the job. One sampler
            # per distributed rank would duplicate node-wide queries and add
            # avoidable measurement overhead on four-/eight-GPU workloads.
            # Rank zero records the complete visible allocation for the
            # canonical result.
            if rank != 0:
                continue
            interval = float(config.profiling.parameters.get("sample_interval_seconds", 0.2))
            profilers.append(NvidiaSmiProfiler(interval))
        elif name == "torch_memory":
            profilers.append(TorchMemoryProfiler())
        elif name == "torch_profiler":
            artifact_output_dir = os.environ.get("HPC_AI_ARTIFACT_OUTPUT_DIR", config.runtime.output_dir)
            profilers.append(
                TorchProfiler(artifact_output_dir, config.experiment_id, repetition_number=repetition_number)
            )
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
