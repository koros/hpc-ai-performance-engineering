from __future__ import annotations

from dataclasses import dataclass, field


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

    def render(self) -> str:
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --nodes={self.nodes}",
            f"#SBATCH --gres=gpu:{self.gpus_per_node}",
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
            f"#SBATCH --time={self.time_limit}",
            "#SBATCH --output=results/raw/%x-%j.out",
            "#SBATCH --error=results/raw/%x-%j.err",
        ]
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        if self.account:
            lines.append(f"#SBATCH --account={self.account}")
        lines.append("")
        lines.append("set -euo pipefail")
        for module in self.modules:
            lines.append(f"module load {module}")
        for key, value in self.environment.items():
            lines.append(f"export {key}={value}")
        lines.append("")
        lines.append(self.command)
        lines.append("")
        return "\n".join(lines)
