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
    platform: str = "local"
    gpus: int = 0
    gpu_model: str | None = None
    interconnect: str | None = None


@dataclass(frozen=True)
class WorkloadConfig:
    name: str
    model: str
    dataset: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DistributedConfig:
    strategy: DistributedStrategy = "none"
    nodes: int = 1
    gpus_per_node: int = 0
    backend: str = "nccl"


@dataclass(frozen=True)
class RuntimeConfig:
    precision: Precision = "bf16"
    max_steps: int | None = None
    seed: int = 1337
    output_dir: str = "results/raw"


@dataclass(frozen=True)
class ProfilingConfig:
    enabled: bool = False
    profilers: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
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
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from JSON or YAML."""
    config_path = Path(path)
    raw_text = config_path.read_text(encoding="utf-8")

    if config_path.suffix.lower() in {".yaml", ".yml"}:
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
