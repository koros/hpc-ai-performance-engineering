from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote

from hpc_ai_perf.config import ExperimentConfig


@dataclass(frozen=True)
class SlurmJob:
    job_name: str
    command: str
    partition: str | None = None
    nodes: int = 1
    gpus_per_node: int = 1
    cpus_per_task: int = 8
    time_limit: str = "01:00:00"
    account: str | None = None
    modules: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        config_path: str | Path,
        *,
        partition: str | None = None,
        account: str | None = None,
        time_limit: str | None = None,
        cpus_per_task: int = 8,
        modules: list[str] | None = None,
        environment: dict[str, str] | None = None,
    ) -> SlurmJob:
        gpus_per_node = config.distributed.gpus_per_node or config.hardware.gpus
        command = f"python -m hpc_ai_perf.cli run --config {quote(str(config_path))}"
        return cls(
            job_name=_job_name(config),
            command=command,
            partition=partition,
            nodes=config.distributed.nodes,
            gpus_per_node=gpus_per_node,
            cpus_per_task=cpus_per_task,
            time_limit=time_limit or _time_limit(config),
            account=account,
            modules=modules or [],
            environment=environment or {},
        )

    def render(self) -> str:
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --nodes={self.nodes}",
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
            f"#SBATCH --time={self.time_limit}",
            "#SBATCH --output=results/raw/%x-%j.out",
            "#SBATCH --error=results/raw/%x-%j.err",
        ]
        if self.gpus_per_node > 0:
            lines.insert(4, f"#SBATCH --gres=gpu:{self.gpus_per_node}")
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        if self.account:
            lines.append(f"#SBATCH --account={self.account}")
        lines.append("")
        lines.append("set -euo pipefail")
        for module in self.modules:
            lines.append(f"module load {quote(module)}")
        for key, value in self.environment.items():
            lines.append(f"export {key}={quote(value)}")
        lines.append("")
        lines.append(self.command)
        lines.append("")
        return "\n".join(lines)


def render_slurm_script(
    config: ExperimentConfig,
    config_path: str | Path,
    *,
    partition: str | None = None,
    account: str | None = None,
    time_limit: str | None = None,
    cpus_per_task: int = 8,
    modules: list[str] | None = None,
    environment: dict[str, str] | None = None,
) -> str:
    return SlurmJob.from_config(
        config,
        config_path,
        partition=partition,
        account=account,
        time_limit=time_limit,
        cpus_per_task=cpus_per_task,
        modules=modules,
        environment=environment,
    ).render()


def write_slurm_script(script: str, output_dir: str | Path, experiment_id: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    script_path = output_path / f"{experiment_id}.sbatch"
    script_path.write_text(script, encoding="utf-8")
    return script_path


def _job_name(config: ExperimentConfig) -> str:
    return f"hpc-ai-{config.experiment_id}".lower().replace("_", "-")


def _time_limit(config: ExperimentConfig) -> str:
    gpu_hours = config.metadata.get("gpu_hours_estimate")
    if isinstance(gpu_hours, int | float) and config.hardware.gpus > 0:
        wall_hours = max(1, round((float(gpu_hours) / config.hardware.gpus) + 0.5))
        return f"{wall_hours:02d}:00:00"
    return "01:00:00"
