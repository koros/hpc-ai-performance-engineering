from __future__ import annotations

from abc import ABC
from contextlib import suppress

from hpc_ai_perf.config import ExperimentConfig


class Profiler(ABC):
    """Base profiler interface used by the local runner.

    Profilers are optional metric collectors that surround a workload run. They
    can represent simple in-process probes, external command wrappers, or future
    GPU tooling adapters. The base implementation is intentionally no-op so
    smoke and dev workflows can run without platform-specific profilers.
    """

    name = "profiler"
    measurement_stop_priority = 0

    def start(self, config: ExperimentConfig) -> None:
        """Start collecting metrics for a configured run."""

    def begin_measurement(self) -> None:
        """Mark the beginning of the workload's comparable measured region."""

    def end_measurement(self) -> None:
        """Freeze the end of the workload's comparable measured region."""

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
        started: list[Profiler] = []
        try:
            for profiler in self._profilers:
                profiler.start(config)
                started.append(profiler)
        except Exception:
            # A later collector can fail after an earlier one has started a
            # sampling thread. Stop those collectors immediately before the
            # runner records the preflight failure.
            for profiler in reversed(started):
                with suppress(Exception):
                    profiler.stop()
            self._profilers = []
            raise

    def begin_measurement(self) -> None:
        """Open the measured region on every profiler adapter."""
        begun: list[Profiler] = []
        try:
            for profiler in self._profilers:
                profiler.begin_measurement()
                begun.append(profiler)
        except Exception:
            for profiler in sorted(begun, key=lambda item: item.measurement_stop_priority, reverse=True):
                with suppress(Exception):
                    profiler.end_measurement()
            raise

    def end_measurement(self) -> None:
        """Freeze measurement collectors before expensive profiler finalization."""
        failures: list[str] = []
        for profiler in sorted(self._profilers, key=lambda item: item.measurement_stop_priority, reverse=True):
            try:
                profiler.end_measurement()
            except Exception as exc:
                failures.append(f"{profiler.name}: {type(exc).__name__}: {exc}")
        if failures:
            raise RuntimeError("; ".join(failures))

    def stop(self) -> dict[str, int | float | str | bool | None]:
        """Stop every profiler and merge their scalar metrics with name prefixes."""
        metrics: dict[str, int | float | str | bool | None] = {}
        failures: list[str] = []
        for profiler in reversed(self._profilers):
            try:
                for key, value in profiler.stop().items():
                    metrics[f"{profiler.name}.{key}"] = value
            except Exception as exc:
                failures.append(f"{profiler.name}: {type(exc).__name__}: {exc}")
        if failures:
            raise RuntimeError("; ".join(failures))
        return metrics
