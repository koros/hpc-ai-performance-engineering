from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hpc_ai_perf.cli import _materialize_platform_batch, _render_platform_slurm
from hpc_ai_perf.config import (
    discover_experiment_configs,
    load_experiment_config,
    load_platform_conditions,
    load_platform_profile,
    materialize_platform_configs,
    resolve_platform_config,
)
from hpc_ai_perf.results import ExperimentResult, flatten_result_record
from hpc_ai_perf.tracking import build_platform_progress_rows, track_platform_progress
from hpc_ai_perf.validation import validate_result_records


def test_platform_profiles_load() -> None:
    l40s = load_platform_profile("platforms/l40s.json")
    h100 = load_platform_profile("platforms/h100.json")

    assert l40s.platform_id == "l40s"
    assert l40s.partition == "gpu_l40s"
    assert l40s.output_root == "results/raw/l40s"
    assert h100.platform_id == "h100"
    assert h100.max_gpus == 4


def test_execution_matrix_expands_ranges_and_keeps_platforms_separate() -> None:
    l40s = load_platform_conditions("platforms/execution_matrix.json", "l40s", wave="0")
    h100 = load_platform_conditions("platforms/execution_matrix.json", "h100", wave="0")

    assert [condition.condition_id for condition in l40s] == [f"E{index:03d}@l40s" for index in range(1, 7)]
    assert [condition.condition_id for condition in h100 if condition.status == "planned"] == [
        f"E{index:03d}@h100" for index in range(1, 8)
    ]


def test_deferred_platform_conditions_are_tracked() -> None:
    conditions = load_platform_conditions(
        "platforms/execution_matrix.json",
        "l40s",
        experiment_ids={"E007", "E121", "E131"},
    )

    assert {(condition.condition_id, condition.status) for condition in conditions} == {
        ("E007@l40s", "deferred"),
        ("E121@l40s", "deferred"),
        ("E131@l40s", "deferred"),
    }


def test_resolve_platform_config_preserves_logical_id_and_source_config() -> None:
    entry = next(entry for entry in discover_experiment_configs() if entry.config.experiment_id == "E008")
    profile = load_platform_profile("platforms/l40s.json")
    condition = load_platform_conditions(
        "platforms/execution_matrix.json",
        "l40s",
        experiment_ids={"E008"},
        include_deferred=False,
    )[0]

    resolved = resolve_platform_config(entry, profile, condition)

    assert resolved.experiment_id == "E008"
    assert resolved.hardware.platform == "lovelace_l40s"
    assert resolved.hardware.gpu_model == "NVIDIA L40S 48GB"
    assert resolved.hardware.interconnect == "PCIe"
    assert resolved.runtime.output_dir == "results/raw/l40s"
    assert resolved.metadata["condition_id"] == "E008@l40s"
    assert resolved.metadata["source_config_path"] == str(entry.path)
    assert len(str(resolved.metadata["source_config_sha256"])) == 64


def test_materialize_platform_configs_writes_loadable_resolved_configs(tmp_path: Path) -> None:
    profile = load_platform_profile("platforms/l40s.json")
    conditions = load_platform_conditions(
        "platforms/execution_matrix.json",
        "l40s",
        wave="2",
        include_deferred=False,
    )

    paths = materialize_platform_configs(profile, conditions, tmp_path)

    assert [path.name for path in paths] == ["E123.json", "E124.json", "E127.json", "E128.json"]
    config = load_experiment_config(paths[0])
    assert config.metadata["condition_id"] == "E123@l40s"
    assert config.hardware.platform == "lovelace_l40s"


def test_materialize_platform_batch_filters_by_wave_and_experiment(tmp_path: Path) -> None:
    paths = _materialize_platform_batch(
        platform_profile="platforms/l40s.json",
        execution_matrix="platforms/execution_matrix.json",
        config_root="configs",
        output_dir=str(tmp_path),
        wave="1",
        experiment_ids={"E008"},
        include_deferred=False,
    )

    assert paths == [tmp_path / "E008.json"]
    config = load_experiment_config(paths[0])
    assert config.metadata["condition_id"] == "E008@l40s"
    assert config.metadata["required_trials"] == 3


def test_render_platform_slurm_uses_profile_partition_and_matrix_trials(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    slurm_dir = tmp_path / "slurm"

    config_paths, script_paths = _render_platform_slurm(
        platform_profile="platforms/l40s.json",
        execution_matrix="platforms/execution_matrix.json",
        config_root="configs",
        config_output_dir=str(config_dir),
        output_dir=str(slurm_dir),
        wave="1",
        experiment_ids={"E008"},
        profile_path=None,
        partition=None,
        account=None,
        time_limit=None,
        cpus_per_task=None,
        modules=[],
        env_values=[],
    )

    assert config_paths == [config_dir / "E008.json"]
    assert script_paths == [slurm_dir / "E008.sbatch"]
    script = script_paths[0].read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu_l40s" in script
    assert "#SBATCH --array=1-3" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "source slurm/modules_for_hpc_ai" in script
    assert f"--config {config_dir / 'E008.json'}" in script


def test_platform_condition_rejects_duplicate_matrix_rows(tmp_path: Path) -> None:
    matrix_path = tmp_path / "duplicate.json"
    matrix_path.write_text(
        json.dumps(
            {
                "waves": [
                    {
                        "platform": "l40s",
                        "wave": "0",
                        "name": "one",
                        "experiment_ids": ["E001"],
                    },
                    {
                        "platform": "l40s",
                        "wave": "1",
                        "name": "two",
                        "experiment_ids": ["E001"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate platform conditions"):
        load_platform_conditions(matrix_path, "l40s", include_deferred=False)


def test_platform_result_writes_atomically_under_platform_root(tmp_path: Path) -> None:
    entry = next(entry for entry in discover_experiment_configs() if entry.config.experiment_id == "E123")
    profile = load_platform_profile("platforms/l40s.json")
    condition = load_platform_conditions(
        "platforms/execution_matrix.json",
        "l40s",
        experiment_ids={"E123"},
        include_deferred=False,
    )[0]
    config = resolve_platform_config(entry, profile, condition)
    config_hash = hashlib.sha256(
        json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = ExperimentResult.from_config(
        config,
        status="completed",
        started_at="2026-08-27T10:00:00+00:00",
        completed_at="2026-08-27T10:01:00+00:00",
        runtime_seconds=60.0,
        repetition_number=1,
        metrics={
            "measured_seconds": 50.0,
            "gpu_hours": 50.0 / 3600,
            "total_tokens": 1000,
            "throughput_tokens_sec": 20.0,
            "throughput_unit": "training_tokens_per_second",
        },
        environment={
            "git_commit": "a" * 40,
            "git_dirty": False,
            "config_sha256": config_hash,
            "command": "python -m hpc_ai_perf.cli run",
            "observed_gpu_model": "NVIDIA L40S",
        },
    )

    raw_root = tmp_path / "results" / "raw"
    result_path = result.write_json(raw_root / "l40s", preserve_trial=True)

    assert result_path.parent == raw_root / "l40s" / "E123"
    assert result.condition_id == "E123@l40s"
    assert result.platform_id == "l40s"
    assert not list(result_path.parent.glob("*.tmp"))
    assert validate_result_records(raw_root) == []

    flattened = flatten_result_record(result.to_dict(), result_path)
    assert flattened["condition_id"] == "E123@l40s"
    assert flattened["platform_id"] == "l40s"


def test_platform_result_validation_rejects_wrong_platform_directory(tmp_path: Path) -> None:
    entry = next(entry for entry in discover_experiment_configs() if entry.config.experiment_id == "E123")
    profile = load_platform_profile("platforms/l40s.json")
    condition = load_platform_conditions(
        "platforms/execution_matrix.json",
        "l40s",
        experiment_ids={"E123"},
        include_deferred=False,
    )[0]
    config = resolve_platform_config(entry, profile, condition)
    result = ExperimentResult.from_config(
        config,
        status="failed",
        started_at="2026-08-27T10:00:00+00:00",
        completed_at="2026-08-27T10:00:01+00:00",
        runtime_seconds=1.0,
        error="test failure",
    )
    raw_root = tmp_path / "results" / "raw"
    result.write_json(raw_root / "h100", preserve_trial=True)

    issues = validate_result_records(raw_root)

    assert any("result path platform does not match platform_id" in issue.message for issue in issues)


def test_platform_tracker_counts_latest_valid_independent_trials(tmp_path: Path) -> None:
    entry = next(entry for entry in discover_experiment_configs() if entry.config.experiment_id == "E123")
    profile = load_platform_profile("platforms/l40s.json")
    conditions = load_platform_conditions(
        "platforms/execution_matrix.json",
        "l40s",
        wave="2",
        include_deferred=False,
    )
    condition = next(item for item in conditions if item.matrix_id == "E123")
    config = resolve_platform_config(entry, profile, condition)
    raw_root = tmp_path / "results" / "raw"

    failed = ExperimentResult.from_config(
        config,
        status="failed",
        started_at="2026-08-27T10:00:00+00:00",
        completed_at="2026-08-27T10:00:30+00:00",
        runtime_seconds=30.0,
        repetition_number=1,
        error="transient scheduler failure",
    )
    failed.write_json(raw_root / "l40s", preserve_trial=True)

    config_hash = hashlib.sha256(
        json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    completed = ExperimentResult.from_config(
        config,
        status="completed",
        started_at="2026-08-27T10:01:00+00:00",
        completed_at="2026-08-27T10:02:00+00:00",
        runtime_seconds=60.0,
        repetition_number=1,
        metrics={
            "measured_seconds": 50.0,
            "gpu_hours": 50.0 / 3600,
            "total_tokens": 1000,
            "throughput_unit": "training_tokens_per_second",
        },
        environment={
            "git_commit": "a" * 40,
            "git_dirty": False,
            "config_sha256": config_hash,
            "command": "python -m hpc_ai_perf.cli run",
            "observed_gpu_model": "NVIDIA L40S",
        },
    )
    completed.write_json(raw_root / "l40s", preserve_trial=True)

    rows = build_platform_progress_rows(conditions, raw_root)
    row = next(item for item in rows if item["condition_id"] == "E123@l40s")

    assert row["status"] == "Partial"
    assert row["valid_trials"] == 1
    assert row["failed_trials"] == 0
    assert row["required_trials"] == 3


def test_platform_tracker_writes_deferred_conditions(tmp_path: Path) -> None:
    csv_path, markdown_path = track_platform_progress(
        "platforms/l40s.json",
        result_dir=tmp_path / "raw",
        output_csv=tmp_path / "tracker.csv",
        output_markdown=tmp_path / "tracker.md",
        wave="deferred",
    )

    assert csv_path.exists()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# L40S Experiment Tracker" in markdown
    assert "E007@l40s" in markdown
    assert "Deferred" in markdown
