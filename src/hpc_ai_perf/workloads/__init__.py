"""Workload abstractions and registry."""

from hpc_ai_perf.workloads.base import Workload, WorkloadRunOutput
from hpc_ai_perf.workloads.registry import WorkloadRegistry, default_registry

__all__ = ["Workload", "WorkloadRegistry", "WorkloadRunOutput", "default_registry"]
