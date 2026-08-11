from __future__ import annotations

import json
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
        error: str | None = None,
    ) -> ExperimentResult:
        """Build a result record by combining config metadata with run outputs.

        The runner supplies timestamps, status, runtime, metrics, environment,
        and any error string. This constructor copies stable identifying fields
        from the config so each JSON result is self-contained even if the config
        file is moved or edited later.
        """
        metrics = metrics or {}
        return cls(
            schema_version="1.0.0",
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
            throughput_tokens_sec=_optional_float(metrics.get("throughput_tokens_sec")),
            throughput_samples_sec=_optional_float(metrics.get("throughput_samples_sec")),
            gpu_utilization=_optional_float(metrics.get("gpu_utilization")),
            memory_used_gb=_optional_float(metrics.get("memory_used_gb")),
            latency_p50_ms=_optional_float(metrics.get("latency_p50_ms")),
            latency_p95_ms=_optional_float(metrics.get("latency_p95_ms")),
            metrics=metrics,
            environment=environment or {},
            config=config.to_dict(),
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the immutable result object into JSON-serializable data."""
        return asdict(self)

    def write_json(self, output_dir: str | Path) -> Path:
        """Write this result as pretty-printed JSON named by experiment ID.

        Existing results for the same experiment ID are overwritten by design:
        the project keeps the latest local/dev/smoke run at a predictable path,
        while Slurm stdout/stderr logs keep job-specific scheduler context.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
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
