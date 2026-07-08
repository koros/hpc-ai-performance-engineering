"""Workload abstractions and registry."""

from hpc_ai_perf.workloads.base import Workload, WorkloadRunOutput
from hpc_ai_perf.workloads.registry import WorkloadRegistry, default_registry
from hpc_ai_perf.workloads.transformer import TransformerWorkload, transformer_workloads

__all__ = [
    "TransformerWorkload",
    "Workload",
    "WorkloadRegistry",
    "WorkloadRunOutput",
    "default_registry",
    "transformer_workloads",
]
