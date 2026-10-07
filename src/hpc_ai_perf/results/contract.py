from __future__ import annotations

import re
from typing import Literal

SampleScalar = int | float | str | bool | None
SampleSeries = list[SampleScalar]
OutcomeClass = Literal["success", "oom", "unsupported", "failed"]


class OutOfMemoryWorkloadError(RuntimeError):
    """Raised when a workload cannot fit the requested hardware shape."""


class UnsupportedWorkloadError(RuntimeError):
    """Raised when the requested workload/backend combination is unavailable."""


def classify_outcome(status: str, error: BaseException | str | None = None) -> OutcomeClass:
    """Classify a run while preserving the legacy completed/failed status field."""
    if status == "completed":
        return "success"
    if isinstance(error, OutOfMemoryWorkloadError):
        return "oom"
    if isinstance(error, UnsupportedWorkloadError):
        return "unsupported"
    message = str(error or "").lower()
    if re.search(r"out of memory|cuda out of memory|cublas_status_alloc_failed|oom", message):
        return "oom"
    if re.search(r"unsupported|not installed|no execution adapter|requires .* package|does not support", message):
        return "unsupported"
    return "failed"
