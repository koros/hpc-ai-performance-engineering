from __future__ import annotations

import platform
import socket
import time
from datetime import UTC, datetime

from hpc_ai_perf.config import ExperimentConfig
from hpc_ai_perf.profiling import NullProfiler, Profiler
from hpc_ai_perf.results import ExperimentResult
from hpc_ai_perf.workloads import WorkloadRegistry, default_registry


class LocalBenchmarkRunner:
    """Run a workload in the current Python process for local validation."""

    def __init__(self, registry: WorkloadRegistry | None = None, profiler: Profiler | None = None) -> None:
        self._registry = registry or default_registry()
        self._profiler = profiler or NullProfiler()

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        workload = self._registry.get(config.workload.name)
        started_at = _now()
        start = time.perf_counter()
        metrics: dict[str, int | float | str | bool | None] = {}
        error: str | None = None
        status = "completed"

        try:
            workload.prepare(config)
            self._profiler.start(config)
            output = workload.run(config)
            metrics.update(output.metrics)
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            metrics.update(self._profiler.stop())

        completed_at = _now()
        runtime_seconds = time.perf_counter() - start
        return ExperimentResult.from_config(
            config,
            status=status,  # type: ignore[arg-type]
            started_at=started_at,
            completed_at=completed_at,
            runtime_seconds=runtime_seconds,
            metrics=metrics,
            environment=_environment(),
            error=error,
        )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _environment() -> dict[str, int | float | str | bool | None]:
    return {
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

