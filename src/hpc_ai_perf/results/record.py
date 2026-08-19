from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from hpc_ai_perf.config import ExperimentConfig, Precision

ResultStatus = Literal["completed", "failed"]


@dataclass(frozen=True)
class ExperimentResult:
    """Structured raw result record written after every workload run.

    Raw result JSON files are the durable evidence produced by local, dev, smoke,
    and HPC jobs. The top-level fields capture the stable schema used by
    aggregation and validation, while `metrics`, `environment`, and `config`
    preserve detailed run-specific context for later analysis.
    """

    schema_version: str
    run_id: str
    repetition_number: int
    experiment_id: str
    experiment: str
    workload: str
    model: str
    mode: str
    gpus: int
    precision: Precision
    status: ResultStatus
    started_at: str
    completed_at: str
    runtime_seconds: float
    throughput_unit: str | None
    warmup_iterations: int | None
    measured_iterations: int | None
    model_revision: str | None
    dataset_revision: str | None
    git_commit: str | None
    git_dirty: bool | None
    config_sha256: str | None
    command: str | None
    metrics: dict[str, int | float | str | bool | None] = field(default_factory=dict)
    environment: dict[str, int | float | str | bool | None] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    throughput_tokens_sec: float | None = None
    throughput_samples_sec: float | None = None
    gpu_utilization: float | None = None
    memory_used_gb: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    error: str | None = None

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        *,
        status: ResultStatus,
        started_at: str,
        completed_at: str,
        runtime_seconds: float,
        metrics: dict[str, int | float | str | bool | None] | None = None,
        environment: dict[str, int | float | str | bool | None] | None = None,
        repetition_number: int | None = None,
        error: str | None = None,
    ) -> ExperimentResult:
        """Build a result record by combining config metadata with run outputs.

        The runner supplies timestamps, status, runtime, metrics, environment,
        and any error string. This constructor copies stable identifying fields
        from the config so each JSON result is self-contained even if the config
        file is moved or edited later.
        """
        metrics = metrics or {}
        environment = environment or {}
        resolved_repetition = _repetition_number(config, environment, repetition_number)
        return cls(
            schema_version="1.2.0",
            run_id=_run_id(config.experiment_id, started_at, resolved_repetition),
            repetition_number=resolved_repetition,
            experiment_id=config.experiment_id,
            experiment=config.experiment,
            workload=config.workload.name,
            model=config.workload.model,
            mode=config.mode,
            gpus=config.hardware.gpus,
            precision=config.runtime.precision,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            runtime_seconds=runtime_seconds,
            throughput_unit=_throughput_unit(config, metrics),
            warmup_iterations=_iteration_count(
                metrics,
                ("warmup_steps", "warmup_batches", "warmup_rounds"),
                fallback=_default_warmup_iterations(config),
            ),
            measured_iterations=_iteration_count(
                metrics,
                ("steps", "batches", "rounds"),
                fallback=config.runtime.max_steps,
            ),
            model_revision=str(
                config.workload.parameters.get("hf_model_revision") or config.workload.model_revision or ""
            )
            or None,
            dataset_revision=str(config.metadata.get("dataset_revision") or config.workload.dataset or "") or None,
            git_commit=_optional_str(environment.get("git_commit")),
            git_dirty=_optional_bool(environment.get("git_dirty")),
            config_sha256=_optional_str(environment.get("config_sha256")),
            command=_optional_str(environment.get("command")),
            throughput_tokens_sec=_optional_float(metrics.get("throughput_tokens_sec")),
            throughput_samples_sec=_optional_float(metrics.get("throughput_samples_sec")),
            gpu_utilization=_optional_float(metrics.get("gpu_utilization")),
            memory_used_gb=_optional_float(metrics.get("memory_used_gb")),
            latency_p50_ms=_optional_float(metrics.get("latency_p50_ms")),
            latency_p95_ms=_optional_float(metrics.get("latency_p95_ms")),
            metrics=metrics,
            environment=environment,
            config=config.to_dict(),
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the immutable result object into JSON-serializable data."""
        return asdict(self)

    def write_json(self, output_dir: str | Path, *, preserve_trial: bool = False) -> Path:
        """Write a result, optionally preserving it as an immutable trial.

        Local smoke runs retain the predictable ``<experiment_id>.json`` path.
        Matrix execution uses ``preserve_trial=True`` and writes a unique file
        under ``<experiment_id>/<run_id>.json`` so repeated jobs cannot replace
        earlier research evidence.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if preserve_trial:
            output_path = output_path / self.experiment_id
            output_path.mkdir(parents=True, exist_ok=True)
            result_path = output_path / f"{self.run_id}.json"
        else:
            result_path = output_path / f"{self.experiment_id}.json"
        result_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result_path


def _optional_float(value: object) -> float | None:
    """Return numeric metric values as floats and ignore missing/non-numeric values."""
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _throughput_unit(config: ExperimentConfig, metrics: dict[str, int | float | str | bool | None]) -> str | None:
    observed = _optional_str(metrics.get("throughput_unit"))
    if observed:
        return observed
    if config.workload.name == "synthetic":
        return "synthetic_tokens_per_second"
    if config.workload.name == "environment_validation":
        target = config.workload.parameters.get("validation_target")
        return "generated_tokens_per_second" if target == "vllm_serving" else None
    if config.mode in {"training", "profiling"}:
        return "training_tokens_per_second"
    if config.mode == "inference" and config.workload.name == "distilbert":
        return "input_tokens_per_second"
    if config.mode == "inference":
        return "generated_tokens_per_second"
    return None


def _iteration_count(
    metrics: dict[str, int | float | str | bool | None],
    names: tuple[str, ...],
    *,
    fallback: int | None,
) -> int | None:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, int) and value >= 0:
            return value
    return fallback if isinstance(fallback, int) and fallback >= 0 else None


def _default_warmup_iterations(config: ExperimentConfig) -> int | None:
    measured = config.runtime.max_steps
    if measured is None:
        return None
    configured = config.workload.parameters.get("warmup_steps")
    if configured is not None:
        return max(0, int(configured))
    if measured <= 1:
        return 0
    return min(5, max(1, measured // 10))


def _repetition_number(
    config: ExperimentConfig,
    environment: dict[str, int | float | str | bool | None],
    explicit: int | None,
) -> int:
    candidates = (
        explicit,
        environment.get("slurm_array_task_id"),
        config.workload.parameters.get("repeat_index"),
        1,
    )
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            repetition = int(candidate)
        except (TypeError, ValueError):
            continue
        if repetition >= 1:
            return repetition
    raise ValueError("repetition_number must be at least 1")


def _run_id(experiment_id: str, started_at: str, repetition_number: int) -> str:
    timestamp = re.sub(r"[^0-9]", "", started_at)[:20] or "unknown"
    return f"{experiment_id}-{timestamp}-r{repetition_number:03d}-{uuid.uuid4().hex[:8]}"
