from __future__ import annotations

from abc import ABC

from hpc_ai_perf.config import ExperimentConfig


class Profiler(ABC):
    """Base profiler interface used by the local runner.

    Profilers are optional metric collectors that surround a workload run. They
    can represent simple in-process probes, external command wrappers, or future
    GPU tooling adapters. The base implementation is intentionally no-op so
    smoke and dev workflows can run without platform-specific profilers.
    """

    name = "profiler"

    def start(self, config: ExperimentConfig) -> None:
        """Start collecting metrics for a configured run."""

    def stop(self) -> dict[str, int | float | str | bool | None]:
        """Stop collecting metrics and return machine-readable scalar values."""
        return {}


class NullProfiler(Profiler):
    """No-op profiler used when profiling is disabled or unavailable."""

    name = "none"


class CompositeProfiler(Profiler):
    """Run multiple profiler adapters as one profiler.

    The composite prefixes returned metric names with each profiler's name so
    independent adapters can report similar metric keys without overwriting one
    another in the final result record.
    """

    def __init__(self, profilers: list[Profiler]) -> None:
        """Store profiler adapters in the order they should start and stop."""
        self._profilers = profilers

    def start(self, config: ExperimentConfig) -> None:
        """Start every profiler before the workload enters its measured region."""
        for profiler in self._profilers:
            profiler.start(config)

    def stop(self) -> dict[str, int | float | str | bool | None]:
        """Stop every profiler and merge their scalar metrics with name prefixes."""
        metrics: dict[str, int | float | str | bool | None] = {}
        for profiler in self._profilers:
            for key, value in profiler.stop().items():
                metrics[f"{profiler.name}.{key}"] = value
        return metrics
