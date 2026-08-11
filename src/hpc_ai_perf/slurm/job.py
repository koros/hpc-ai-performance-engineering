from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote

from hpc_ai_perf.config import ExperimentConfig


@dataclass(frozen=True)
class SlurmJob:
    """Renderable representation of one `sbatch` script.

    This dataclass separates scheduler intent from string rendering. The CLI and
    bulk renderer fill these fields from experiment configs plus optional Slurm
    profile defaults, and `render` turns the final values into a script that can
    be reviewed before submission.
    """

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
    setup_commands: list[str] = field(default_factory=list)

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
        setup_commands: list[str] | None = None,
    ) -> SlurmJob:
        """Create a Slurm job description from an experiment config.

        GPU count comes from the distributed section when available, otherwise
        from the hardware section. The command always invokes the local runner
        against the same config path, which keeps generated scripts simple and
        reproducible.
        """
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
            setup_commands=setup_commands or [],
        )

    def render(self) -> str:
        """Render the job description as a bash `sbatch` script.

        The script intentionally uses relative output paths and setup commands
        because the project workflow assumes jobs are submitted from the repo
        root. Optional directives such as partition and account are omitted when
        not provided so clusters that do not require them can still accept the
        script.
        """
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
        for command in self.setup_commands:
            lines.append(command)
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
    setup_commands: list[str] | None = None,
) -> str:
    """Convenience wrapper that builds and renders one Slurm script."""
    return SlurmJob.from_config(
        config,
        config_path,
        partition=partition,
        account=account,
        time_limit=time_limit,
        cpus_per_task=cpus_per_task,
        modules=modules,
        environment=environment,
        setup_commands=setup_commands,
    ).render()


def write_slurm_script(script: str, output_dir: str | Path, experiment_id: str) -> Path:
    """Write a rendered Slurm script to `<output_dir>/<experiment_id>.sbatch`."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    script_path = output_path / f"{experiment_id}.sbatch"
    script_path.write_text(script, encoding="utf-8")
    return script_path


def _job_name(config: ExperimentConfig) -> str:
    """Create a scheduler-safe job name from the experiment ID."""
    return f"hpc-ai-{config.experiment_id}".lower().replace("_", "-")


def _time_limit(config: ExperimentConfig) -> str:
    """Estimate a conservative wall-clock limit from configured GPU-hours.

    GPU-hour estimates describe total accelerator time. Dividing by the number
    of GPUs gives approximate wall time; adding half an hour before rounding
    gives the generated script a small buffer for startup and variance.
    """
    gpu_hours = config.metadata.get("gpu_hours_estimate")
    if isinstance(gpu_hours, int | float) and config.hardware.gpus > 0:
        wall_hours = max(1, round((float(gpu_hours) / config.hardware.gpus) + 0.5))
        return f"{wall_hours:02d}:00:00"
    return "01:00:00"
