from __future__ import annotations

from pathlib import Path


def build_submission_plan(script_dir: str | Path, *, batch_name: str | None = None) -> str:
    scripts = sorted(Path(script_dir).glob("*.sbatch"))
    title = batch_name or Path(script_dir).name.replace("_", " ").title()

    lines = [
        f"# Slurm Submission Plan: {title}",
        "",
        f"Script directory: `{Path(script_dir)}`",
        f"Total jobs: {len(scripts)}",
        "",
    ]
    if not scripts:
        lines.extend(
            [
                "No `.sbatch` scripts were found.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "## Review Checklist",
            "",
            "- Confirm `#SBATCH --partition` is correct, and add `#SBATCH --account` only if the cluster requires it.",
            "- Confirm module, virtual environment, and environment-variable setup is correct.",
            "- Confirm output/error paths point to writable storage.",
            "- Submit Batch 1 before Batch 2 and inspect logs before continuing.",
            "",
            "## Commands",
            "",
            "```bash",
        ]
    )
    for script in scripts:
        lines.append(f"sbatch {script}")
    lines.extend(
        [
            "```",
            "",
            "## Jobs",
            "",
            "| Order | Script | Command |",
            "| ---: | --- | --- |",
        ]
    )
    for index, script in enumerate(scripts, start=1):
        lines.append(f"| {index} | `{script.name}` | `sbatch {script}` |")
    lines.append("")
    return "\n".join(lines)


def write_submission_plan(
    script_dir: str | Path,
    output_path: str | Path,
    *,
    batch_name: str | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_submission_plan(script_dir, batch_name=batch_name), encoding="utf-8")
    return output
