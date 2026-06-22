from __future__ import annotations

from dataclasses import dataclass, field

from hpc_ai_perf.workloads.base import Workload
from hpc_ai_perf.workloads.synthetic import SyntheticWorkload


@dataclass
class WorkloadRegistry:
    _workloads: dict[str, Workload] = field(default_factory=dict)

    def register(self, workload: Workload) -> None:
        if workload.name in self._workloads:
            raise ValueError(f"Workload already registered: {workload.name}")
        self._workloads[workload.name] = workload

    def get(self, name: str) -> Workload:
        try:
            return self._workloads[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._workloads)) or "none"
            raise KeyError(f"Unknown workload '{name}'. Available workloads: {available}") from exc

    def names(self) -> list[str]:
        return sorted(self._workloads)


def default_registry() -> WorkloadRegistry:
    registry = WorkloadRegistry()
    registry.register(SyntheticWorkload())
    return registry

