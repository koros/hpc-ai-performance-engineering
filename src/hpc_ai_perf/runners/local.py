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
        self._profiler = profiler or NullProfiler()

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Execute one config and always return a structured result record.

        Workload preparation, profiler start, and workload execution are all
        inside one guarded block. If anything fails, the exception is captured as
        a readable error string and the result is marked `failed`, which is far
        more useful on HPC than losing the run to an unstructured traceback.
        """
        workload = self._registry.get(config.workload.name)
        started_at = _now()
        start = time.perf_counter()
        metrics: dict[str, int | float | str | bool | None] = {}
        error: str | None = None
        status = "completed"

        try:
            # Preparation is allowed to perform dependency checks or lightweight
            # setup before the measured workload starts.
            workload.prepare(config)
            self._profiler.start(config)
            output = workload.run(config)
            metrics.update(output.metrics)
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            # Stop profilers even after workload failure so partial diagnostic
            # metrics can still be written to the raw result record.
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
    """Return a timezone-aware UTC timestamp for result records."""
    return datetime.now(UTC).isoformat()


def _environment() -> dict[str, int | float | str | bool | None]:
    """Capture lightweight host metadata alongside every run."""
    return {
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
