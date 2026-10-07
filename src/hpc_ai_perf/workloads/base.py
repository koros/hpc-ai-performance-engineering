from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from hpc_ai_perf.config import ExperimentConfig
from hpc_ai_perf.results.contract import OutcomeClass, SampleSeries


@dataclass(frozen=True)
class WorkloadRunOutput:
    """Container returned by every workload after execution.

    Workloads should keep their outputs machine-readable. `metrics` contains
    numeric or scalar values that can be aggregated into CSV analysis tables,
    while `artifacts` records paths to any files produced by the run. The class
    is frozen so downstream result writing can treat the output as a stable
    snapshot of what the workload observed.
    """

    metrics: dict[str, int | float | str | bool | None] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    samples: dict[str, SampleSeries] = field(default_factory=dict)
    outcome: OutcomeClass | None = None


class Workload(ABC):
    """Base interface for training, inference, and profiling workloads.

    The runner interacts with workloads only through this interface. Concrete
    workload classes are responsible for interpreting the experiment parameters
    relevant to their model or synthetic benchmark, running the work, and
    returning metrics in a consistent `WorkloadRunOutput` object.
    """

    name: str

    def prepare(self, config: ExperimentConfig) -> None:
        """Prepare datasets, model files, or local state before execution.

        The default implementation is intentionally empty because many smoke and
        synthetic workloads do not need a separate setup step. Heavier workloads
        can override this hook to download models, validate input assets, or
        initialise local caches before `run` starts measuring execution.
        """

    @abstractmethod
    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        """Execute the configured workload and return machine-readable metrics.

        Implementations should raise meaningful exceptions for unrecoverable
        setup or dependency failures. The local runner is responsible for
        converting those failures into structured failed result records, which
        keeps raw output files useful during HPC smoke testing.
        """
