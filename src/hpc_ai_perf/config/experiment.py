from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

ExperimentMode = Literal["training", "inference", "profiling"]
Precision = Literal["fp32", "fp16", "bf16", "fp8", "int8", "int4"]
DistributedStrategy = Literal["none", "ddp", "fsdp", "deepspeed_zero1", "deepspeed_zero2", "deepspeed_zero3"]


@dataclass(frozen=True)
class HardwareConfig:
    """Hardware description requested or observed for an experiment.

    Matrix configs use this block to describe the intended target platform, GPU
    count, GPU model, and interconnect. Local/dev configs keep `gpus` at zero so
    their results are clearly separated from real H100/L40S measurements.
    """

    platform: str = "local"
    gpus: int = 0
    gpu_model: str | None = None
    interconnect: str | None = None


@dataclass(frozen=True)
class WorkloadConfig:
    """Model and dataset settings consumed by a workload adapter.

    `name` selects the registered adapter, `model` records the human-readable or
    canonical model target, `dataset` records the intended data source, and
    `parameters` carries adapter-specific knobs such as sequence length,
    micro-batch size, prompt text, or Hugging Face model overrides.
    """

    name: str
    model: str
    dataset: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DistributedConfig:
    """Distributed execution shape encoded in an experiment config.

    The local runner currently executes single-process workloads, but the Slurm
    renderer and matrix tracker still need these fields to describe DDP, FSDP,
    DeepSpeed, node count, GPUs per node, and communication backend choices.
    """

    strategy: DistributedStrategy = "none"
    nodes: int = 1
    gpus_per_node: int = 0
    backend: str = "nccl"


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime controls shared across workloads.

    Precision, maximum step count, seed, and output directory live here because
    they affect execution rather than the conceptual workload definition. The
    same shape works for local smoke runs and generated HPC jobs.
    """

    precision: Precision = "bf16"
    max_steps: int | None = None
    seed: int = 1337
    output_dir: str = "results/raw"


@dataclass(frozen=True)
class ProfilingConfig:
    """Profiler selection and profiler-specific parameters.

    Profiling is optional because early smoke and dev runs should remain cheap.
    When enabled, this section records which profiler adapters should be used
    and any extra settings needed by platform tools such as Nsight or nvidia-smi.
    """

    enabled: bool = False
    profilers: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level immutable representation of one runnable experiment config.

    The JSON/YAML files are parsed into this dataclass so the rest of the
    project can use typed attributes instead of repeatedly indexing nested
    dictionaries. The object remains intentionally close to the file schema,
    which keeps rendered Slurm scripts, local runs, validation, and reports all
    aligned with the same source data.
    """

    experiment_id: str
    experiment: str
    mode: ExperimentMode
    workload: WorkloadConfig
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Create an experiment config from a parsed JSON/YAML mapping.

        The method performs lightweight structural construction and relies on
        the dataclass constructors to reject missing required nested fields. Full
        project validation lives in `hpc_ai_perf.validation`, where errors can be
        collected across all configs instead of failing at the first file.
        """
        return cls(
            experiment_id=str(data["experiment_id"]),
            experiment=str(data["experiment"]),
            mode=data["mode"],
            workload=WorkloadConfig(**data["workload"]),
            hardware=HardwareConfig(**data.get("hardware", {})),
            distributed=DistributedConfig(**data.get("distributed", {})),
            runtime=RuntimeConfig(**data.get("runtime", {})),
            profiling=ProfilingConfig(**data.get("profiling", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the config back to a plain dictionary for serialization."""
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from JSON or YAML.

    JSON is the normal concrete-config format in this repository. YAML is
    supported for templates or hand-authored variants when the optional `yaml`
    extra is installed. The returned object is always an `ExperimentConfig` so
    downstream code does not need to care which file format was used.
    """
    config_path = Path(path)
    raw_text = config_path.read_text(encoding="utf-8")

    if config_path.suffix.lower() in {".yaml", ".yml"}:
        # Import PyYAML lazily so JSON-only workflows do not require the yaml extra.
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Install the yaml extra to load YAML configs: pip install -e '.[yaml]'") from exc
        data = yaml.safe_load(raw_text)
    else:
        data = json.loads(raw_text)

    if not isinstance(data, dict):
        raise ValueError(f"Experiment config must be a mapping: {config_path}")

    return ExperimentConfig.from_mapping(data)
