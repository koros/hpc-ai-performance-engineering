from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec

from hpc_ai_perf.config import ExperimentConfig
from hpc_ai_perf.workloads.base import Workload, WorkloadRunOutput


@dataclass(frozen=True)
class TransformerWorkload(Workload):
    """Placeholder adapter for transformer workloads that need GPU-capable dependencies."""

    name: str
    display_name: str
    required_packages: tuple[str, ...] = ("torch", "transformers")

    def prepare(self, config: ExperimentConfig) -> None:
        missing = [package for package in self.required_packages if find_spec(package) is None]
        if missing:
            package_list = ", ".join(missing)
            raise RuntimeError(
                f"{self.display_name} requires optional dependencies: {package_list}. "
                "Install them with: python -m pip install -e '.[workloads]'"
            )

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        raise NotImplementedError(
            f"{self.display_name} adapter is scaffolded but execution is not implemented yet for {config.mode}."
        )


def transformer_workloads() -> list[TransformerWorkload]:
    return [
        TransformerWorkload(name="gpt2_small", display_name="GPT-2 Small"),
        TransformerWorkload(name="distilbert", display_name="DistilBERT"),
        TransformerWorkload(name="tinyllama", display_name="TinyLlama"),
    ]
