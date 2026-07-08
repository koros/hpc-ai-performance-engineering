"""Workload abstractions and registry."""

from hpc_ai_perf.workloads.base import Workload, WorkloadRunOutput
from hpc_ai_perf.workloads.registry import WorkloadRegistry, default_registry
from hpc_ai_perf.workloads.transformer import (
    DistilBERTWorkload,
    GPT2SmallWorkload,
    TinyLlamaWorkload,
    TransformerWorkload,
    transformer_workloads,
)

__all__ = [
    "DistilBERTWorkload",
    "GPT2SmallWorkload",
    "TinyLlamaWorkload",
    "TransformerWorkload",
    "Workload",
    "WorkloadRegistry",
    "WorkloadRunOutput",
    "default_registry",
    "transformer_workloads",
]
