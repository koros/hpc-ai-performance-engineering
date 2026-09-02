from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

from hpc_ai_perf.profiling.base import Profiler

_ACTIVE_PROFILER: ContextVar[Profiler | None] = ContextVar("hpc_ai_perf_active_profiler", default=None)


def bind_measurement_profiler(profiler: Profiler) -> Token[Profiler | None]:
    """Bind the runner's profiler so workloads can expose exact timing boundaries."""
    return _ACTIVE_PROFILER.set(profiler)


def reset_measurement_profiler(token: Token[Profiler | None]) -> None:
    """Restore the previous profiler binding after one workload run."""
    _ACTIVE_PROFILER.reset(token)


@contextmanager
def measured_region() -> Iterator[None]:
    """Notify active collectors only while comparable workload work executes."""
    profiler = _ACTIVE_PROFILER.get()
    if profiler is None:
        yield
        return
    profiler.begin_measurement()
    try:
        yield
    finally:
        profiler.end_measurement()
