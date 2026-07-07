from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from hpc_ai_perf.config import ExperimentConfig


@dataclass(frozen=True)
class WorkloadRunOutput:
    metrics: dict[str, int | float | str | bool | None] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)


class Workload(ABC):
    """Base interface for training, inference, and profiling workloads."""

    name: str

    def prepare(self, config: ExperimentConfig) -> None:
        """Prepare datasets, model files, or local state before execution."""

    @abstractmethod
    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        """Execute the configured workload and return machine-readable metrics."""
