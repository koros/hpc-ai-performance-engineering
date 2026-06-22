from __future__ import annotations

from abc import ABC

from hpc_ai_perf.config import ExperimentConfig


class Profiler(ABC):
    """Base profiler interface."""

    name = "profiler"

    def start(self, config: ExperimentConfig) -> None:
        """Start profiling."""

    def stop(self) -> dict[str, int | float | str | bool | None]:
        """Stop profiling and return machine-readable metrics."""
        return {}


class NullProfiler(Profiler):
    name = "none"


class CompositeProfiler(Profiler):
    def __init__(self, profilers: list[Profiler]) -> None:
        self._profilers = profilers

    def start(self, config: ExperimentConfig) -> None:
        for profiler in self._profilers:
            profiler.start(config)

    def stop(self) -> dict[str, int | float | str | bool | None]:
        metrics: dict[str, int | float | str | bool | None] = {}
        for profiler in self._profilers:
            for key, value in profiler.stop().items():
                metrics[f"{profiler.name}.{key}"] = value
        return metrics

