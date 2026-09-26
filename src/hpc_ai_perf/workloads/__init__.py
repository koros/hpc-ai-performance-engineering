"""Workload abstractions and registry."""

from hpc_ai_perf.workloads.base import Workload, WorkloadRunOutput
from hpc_ai_perf.workloads.environment import EnvironmentValidationWorkload
from hpc_ai_perf.workloads.long_context import QwenLongContextWorkload
from hpc_ai_perf.workloads.registry import WorkloadRegistry, default_registry
from hpc_ai_perf.workloads.transformer import (
    DistilBERTWorkload,
    GPT2SmallWorkload,
    Qwen25SevenBTrainingWorkload,
    TinyLlamaWorkload,
    TransformerWorkload,
    transformer_workloads,
)

__all__ = [
    "DistilBERTWorkload",
    "EnvironmentValidationWorkload",
    "GPT2SmallWorkload",
    "QwenLongContextWorkload",
    "Qwen25SevenBTrainingWorkload",
    "TinyLlamaWorkload",
    "TransformerWorkload",
    "Workload",
    "WorkloadRegistry",
    "WorkloadRunOutput",
    "default_registry",
    "transformer_workloads",
]
