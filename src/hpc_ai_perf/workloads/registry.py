from __future__ import annotations

from dataclasses import dataclass, field

from hpc_ai_perf.workloads.base import Workload
from hpc_ai_perf.workloads.environment import EnvironmentValidationWorkload
from hpc_ai_perf.workloads.long_context import QwenLongContextWorkload
from hpc_ai_perf.workloads.lowbit import LowBitServingWorkload
from hpc_ai_perf.workloads.synthetic import SyntheticWorkload
from hpc_ai_perf.workloads.transformer import transformer_workloads


@dataclass
class WorkloadRegistry:
    """Mutable lookup table that maps workload names to runnable adapters.

    Experiment configs refer to workloads by short names such as `synthetic`,
    `gpt2_small`, or `distilbert`. The registry is the bridge between those
    declarative config values and the Python objects that know how to execute
    them.
    """

    _workloads: dict[str, Workload] = field(default_factory=dict)

    def register(self, workload: Workload) -> None:
        """Register one workload adapter under its declared unique name.

        Duplicate names are rejected because a config should always resolve to a
        single, predictable implementation. That matters for reproducibility:
        rerunning `workload.name == "gpt2_small"` should never silently switch to
        a different adapter because of registration order.
        """
        if workload.name in self._workloads:
            raise ValueError(f"Workload already registered: {workload.name}")
        self._workloads[workload.name] = workload

    def get(self, name: str) -> Workload:
        """Return the workload adapter for a config workload name.

        The error message includes available names so a typo in a JSON config can
        be diagnosed from the CLI output or Slurm log without opening the source
        code.
        """
        try:
            return self._workloads[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._workloads)) or "none"
            raise KeyError(f"Unknown workload '{name}'. Available workloads: {available}") from exc

    def names(self) -> list[str]:
        """Return registered workload names in stable sorted order for reports."""
        return sorted(self._workloads)


def default_registry() -> WorkloadRegistry:
    """Create the standard registry used by the local benchmark runner.

    The registry is built when requested rather than as a global singleton. This
    keeps tests isolated, avoids import-time side effects, and lets optional
    transformer workload dependencies be checked by the adapters when they are
    actually executed.
    """
    registry = WorkloadRegistry()
    registry.register(SyntheticWorkload())
    registry.register(EnvironmentValidationWorkload())
    registry.register(LowBitServingWorkload())
    registry.register(QwenLongContextWorkload())
    for workload in transformer_workloads():
        registry.register(workload)
    return registry
