"""Persistent, result-first lifecycle state for one platform experiment wave."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hpc_ai_perf.slurm.bundle import validate_result_bundle_provenance
from hpc_ai_perf.validation import validate_result_file


@dataclass(frozen=True)
class ExperimentResultStatus:
    """Validated result coverage for one experiment in a bundle attempt."""

    experiment_id: str
    required_trials: int
    completed_trials: tuple[int, ...]
    failed_trials: tuple[int, ...]
    invalid_records: int

    @property
    def missing_trials(self) -> tuple[int, ...]:
        completed = set(self.completed_trials)
        return tuple(trial for trial in range(1, self.required_trials + 1) if trial not in completed)

    @property
    def complete(self) -> bool:
        return not self.missing_trials and not self.failed_trials and self.invalid_records == 0


@dataclass(frozen=True)
class BundleResultStatus:
    """Coverage summary for all configs frozen into a submission bundle."""

    bundle_dir: Path
    experiments: tuple[ExperimentResultStatus, ...]

    @property
    def complete(self) -> bool:
        return all(experiment.complete for experiment in self.experiments)

    @property
    def expected_trials(self) -> int:
        return sum(experiment.required_trials for experiment in self.experiments)

    @property
    def completed_trials(self) -> int:
        return sum(len(experiment.completed_trials) for experiment in self.experiments)

    @property
    def retry_experiment_ids(self) -> tuple[str, ...]:
        return tuple(experiment.experiment_id for experiment in self.experiments if not experiment.complete)


def wave_state_path(repository: str | Path, platform_id: str, wave: str) -> Path:
    """Return the ignored state file that removes bundle-path shell variables."""
    return Path(repository) / "build" / "platform_waves" / platform_id / f"wave_{wave}.json"


def append_wave_attempt(
    state_path: str | Path,
    *,
    platform_id: str,
    wave: str,
    bundle_dir: str | Path,
    bundle_id: str,
    manifest_sha256: str,
    status_path: str | Path,
    experiment_ids: list[str],
) -> dict[str, Any]:
    """Persist one bundle attempt before Slurm submission starts."""
    path = Path(state_path)
    state = _load_state(path, platform_id=platform_id, wave=wave)
    attempt = {
        "bundle_dir": str(Path(bundle_dir).resolve()),
        "bundle_id": bundle_id,
        "manifest_sha256": manifest_sha256,
        "status_path": str(Path(status_path).resolve()),
        "experiment_ids": sorted(experiment_ids),
        "created_at": _utc_now(),
        "collected_at": None,
        "imported_paths": [],
    }
    state["attempts"].append(attempt)
    _write_json(path, state)
    return state


def current_wave_attempt(state_path: str | Path) -> dict[str, Any]:
    """Return the latest saved attempt or explain how to start a wave."""
    path = Path(state_path)
    if not path.is_file():
        raise FileNotFoundError(f"No saved platform-wave state at {path}; run submit-platform-wave first")
    state = json.loads(path.read_text(encoding="utf-8"))
    attempts = state.get("attempts") if isinstance(state, dict) else None
    if not isinstance(attempts, list) or not attempts:
        raise ValueError(f"Platform-wave state has no submission attempts: {path}")
    attempt = attempts[-1]
    if not isinstance(attempt, dict):
        raise ValueError(f"Platform-wave attempt is invalid: {path}")
    return attempt


def load_wave_state(state_path: str | Path) -> dict[str, Any]:
    """Load all attempts so collection can combine a clean retry with prior work."""
    path = Path(state_path)
    if not path.is_file():
        raise FileNotFoundError(f"No saved platform-wave state at {path}; run submit-platform-wave first")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("attempts"), list):
        raise ValueError(f"Platform-wave state is invalid: {path}")
    return data


def mark_current_attempt_collected(state_path: str | Path, imported_paths: list[str]) -> dict[str, Any]:
    """Record the exact evidence paths imported by the latest clean attempt."""
    path = Path(state_path)
    state = json.loads(path.read_text(encoding="utf-8"))
    attempts = state.get("attempts") if isinstance(state, dict) else None
    if not isinstance(attempts, list) or not attempts or not isinstance(attempts[-1], dict):
        raise ValueError(f"Platform-wave state has no valid current attempt: {path}")
    attempts[-1]["collected_at"] = _utc_now()
    attempts[-1]["imported_paths"] = sorted(set(imported_paths))
    _write_json(path, state)
    return state


def mark_wave_collected(state_path: str | Path, imported_paths: list[str]) -> dict[str, Any]:
    """Record all evidence imported across attempts for this logical wave."""
    path = Path(state_path)
    state = load_wave_state(path)
    state["collected_at"] = _utc_now()
    state["imported_paths"] = sorted(set(imported_paths))
    _write_json(path, state)
    return state


def summarize_bundle_results(bundle_dir: str | Path) -> BundleResultStatus:
    """Summarize validated trial coverage without copying or mutating evidence."""
    root = Path(bundle_dir)
    manifest_path = root / "results" / "SUBMISSION_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Bundle result manifest is missing: {manifest_path}")
    manifest = _load_manifest(manifest_path)
    statuses: list[ExperimentResultStatus] = []
    configs = manifest["configs"]
    assert isinstance(configs, dict)
    for experiment_id, entry in sorted(configs.items()):
        if not isinstance(entry, dict):
            raise ValueError(f"Bundle manifest config entry is invalid: {experiment_id}")
        required_trials = int(entry.get("required_trials", 1))
        completed: set[int] = set()
        failed: set[int] = set()
        invalid = 0
        for result_path in sorted((root / "results" / experiment_id).glob("*.json")):
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                invalid += 1
                continue
            if not isinstance(result, dict):
                invalid += 1
                continue
            issues = validate_result_file(result_path)
            issues.extend(_bundle_validation_issues(result, result_path))
            repetition = result.get("repetition_number")
            if issues or not isinstance(repetition, int) or repetition < 1:
                invalid += 1
                continue
            if result.get("status") == "completed" and result.get("error") is None:
                if repetition in completed:
                    invalid += 1
                else:
                    completed.add(repetition)
            else:
                failed.add(repetition)
        statuses.append(
            ExperimentResultStatus(
                experiment_id=str(experiment_id),
                required_trials=required_trials,
                completed_trials=tuple(sorted(completed)),
                failed_trials=tuple(sorted(failed)),
                invalid_records=invalid,
            )
        )
    return BundleResultStatus(root, tuple(statuses))


def format_bundle_result_status(status: BundleResultStatus) -> str:
    """Render a concise, action-oriented status report for terminal and live waves."""
    lines = [
        f"Bundle: {status.bundle_dir}",
        f"Validated completed trials: {status.completed_trials}/{status.expected_trials}",
        "",
        "| Experiment | Completed | Required | Failed | Invalid | Missing |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for experiment in status.experiments:
        missing = ",".join(str(value) for value in experiment.missing_trials) or "-"
        lines.append(
            f"| {experiment.experiment_id} | {len(experiment.completed_trials)} | {experiment.required_trials} | "
            f"{len(experiment.failed_trials)} | {experiment.invalid_records} | {missing} |"
        )
    if status.complete:
        lines.extend(["", "All expected trials are present and provenance-valid."])
    else:
        retry_ids = ", ".join(status.retry_experiment_ids)
        lines.extend(["", f"Recovery bundle required for: {retry_ids}"])
    return "\n".join(lines)


def _bundle_validation_issues(result: dict[str, Any], result_path: Path) -> list[str]:
    return validate_result_bundle_provenance(result, result_path)


def _load_state(path: Path, *, platform_id: str, wave: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": "1.0",
            "platform_id": platform_id,
            "wave": str(wave),
            "attempts": [],
            "collected_at": None,
            "imported_paths": [],
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Platform-wave state must be a JSON object: {path}")
    if data.get("platform_id") != platform_id or str(data.get("wave")) != str(wave):
        raise ValueError(f"Platform-wave state does not match {platform_id} wave {wave}: {path}")
    if not isinstance(data.get("attempts"), list):
        raise ValueError(f"Platform-wave state attempts must be a list: {path}")
    return data


def _load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("configs"), dict):
        raise ValueError(f"Bundle result manifest is invalid: {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file_obj:
        json.dump(data, file_obj, indent=2)
        file_obj.write("\n")
        temporary = Path(file_obj.name)
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
