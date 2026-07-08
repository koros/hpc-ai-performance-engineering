from __future__ import annotations

import csv
import re
from datetime import UTC, datetime
from pathlib import Path

JOB_STATUS_FIELDS = [
    "batch_name",
    "experiment_id",
    "script_path",
    "job_id",
    "state",
    "submitted_at",
    "updated_at",
    "notes",
]
SBATCH_JOB_ID_PATTERN = re.compile(r"Submitted batch job\s+(?P<job_id>\d+)")


def build_job_status_rows(script_dir: str | Path, *, batch_name: str | None = None) -> list[dict[str, str]]:
    scripts = sorted(Path(script_dir).glob("*.sbatch"))
    title = batch_name or Path(script_dir).name.replace("_", " ").title()
    timestamp = _utc_now()
    return [
        {
            "batch_name": title,
            "experiment_id": script.stem,
            "script_path": str(script),
            "job_id": "",
            "state": "PLANNED",
            "submitted_at": "",
            "updated_at": timestamp,
            "notes": "",
        }
        for script in scripts
    ]


def write_job_status_csv(script_dir: str | Path, output_path: str | Path, *, batch_name: str | None = None) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(output, build_job_status_rows(script_dir, batch_name=batch_name))
    return output


def record_job_submission(
    status_path: str | Path,
    *,
    experiment_id: str | None = None,
    script_path: str | Path | None = None,
    job_id: str | None = None,
    sbatch_output: str | None = None,
    state: str = "SUBMITTED",
    submitted_at: str | None = None,
    notes: str | None = None,
) -> Path:
    resolved_job_id = job_id or _parse_sbatch_job_id(sbatch_output or "")
    if not resolved_job_id:
        raise ValueError("Provide --job-id or --sbatch-output containing 'Submitted batch job <id>'.")
    if not experiment_id and not script_path:
        raise ValueError("Provide --experiment-id or --script-path to identify the status row.")

    status = Path(status_path)
    rows = _read_rows(status)
    match = _find_status_row(rows, experiment_id=experiment_id, script_path=script_path)
    timestamp = _utc_now()
    match["job_id"] = resolved_job_id
    match["state"] = state
    match["submitted_at"] = submitted_at or match.get("submitted_at") or timestamp
    match["updated_at"] = timestamp
    if notes is not None:
        match["notes"] = notes
    _write_rows(status, rows)
    return status


def build_job_status_report(status_path: str | Path) -> str:
    rows = _read_rows(Path(status_path))
    state_counts: dict[str, int] = {}
    for row in rows:
        state = row.get("state", "UNKNOWN") or "UNKNOWN"
        state_counts[state] = state_counts.get(state, 0) + 1

    lines = [
        "# Slurm Job Status",
        "",
        f"Status file: `{Path(status_path)}`",
        f"Total jobs: {len(rows)}",
        "",
        "## State Summary",
        "",
        "| State | Jobs |",
        "| --- | ---: |",
    ]
    for state, count in sorted(state_counts.items()):
        lines.append(f"| {state} | {count} |")

    lines.extend(
        [
            "",
            "## Jobs",
            "",
            "| Experiment | Job ID | State | Script | Updated | Notes |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {experiment_id} | {job_id} | {state} | `{script_path}` | {updated_at} | {notes} |".format(
                experiment_id=row.get("experiment_id", ""),
                job_id=row.get("job_id", ""),
                state=row.get("state", ""),
                script_path=row.get("script_path", ""),
                updated_at=row.get("updated_at", ""),
                notes=row.get("notes", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_job_status_report(status_path: str | Path, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_job_status_report(status_path), encoding="utf-8")
    return output


def _find_status_row(
    rows: list[dict[str, str]],
    *,
    experiment_id: str | None,
    script_path: str | Path | None,
) -> dict[str, str]:
    script_value = str(script_path) if script_path is not None else None
    for row in rows:
        if experiment_id and row.get("experiment_id") == experiment_id:
            return row
        if script_value and row.get("script_path") == script_value:
            return row
    identifier = experiment_id or script_value
    raise ValueError(f"No Slurm status row found for {identifier}.")


def _parse_sbatch_job_id(output: str) -> str:
    match = SBATCH_JOB_ID_PATTERN.search(output)
    return match.group("job_id") if match else ""


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=JOB_STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
