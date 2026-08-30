from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hpc_ai_perf.cli import _materialize_platform_batch, _render_platform_slurm, _route_bundle_runtime_outputs
from hpc_ai_perf.config import (
    config_sha256,
    discover_experiment_configs,
    load_experiment_config,
    load_platform_conditions,
    load_platform_profile,
    materialize_platform_configs,
    resolve_platform_config,
)
from hpc_ai_perf.results import (
    ExperimentResult,
    flatten_result_record,
    import_result_evidence,
    verify_result_evidence,
)
from hpc_ai_perf.slurm.bundle import _BUNDLE_ID, SubmissionBundle, write_submission_manifest
from hpc_ai_perf.slurm.lifecycle import (
    append_wave_attempt,
    current_wave_attempt,
    mark_wave_collected,
    summarize_bundle_results,
)
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


def test_submission_bundle_id_accepts_documented_utc_timestamp() -> None:
    assert _BUNDLE_ID.fullmatch("l40s-wave-1-f18d325-20260829T231916Z")


def test_bundle_routes_runtime_artifacts_outside_snapshot(tmp_path: Path) -> None:
    source = Path("configs/workload_characterisation/E024_distilbert_compute.json")
    resolved_path = tmp_path / "E024.json"
    resolved_path.write_bytes(source.read_bytes())
    cache_root = tmp_path / "bundle" / "cache" / "runtime"

    _route_bundle_runtime_outputs([resolved_path], cache_root)

    routed = load_experiment_config(resolved_path)
    assert routed.runtime.output_dir == str(cache_root)
    assert config_sha256(routed) == config_sha256(json.loads(resolved_path.read_text(encoding="utf-8")))


def test_bundle_manifest_separates_file_and_config_identities(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    config_dir = root / "configs"
    result_dir = root / "results"
    for directory in (config_dir, result_dir, root / "repo", root / "scripts", root / "logs", root / "cache"):
        directory.mkdir(parents=True, exist_ok=True)
    entry = next(entry for entry in discover_experiment_configs() if entry.config.experiment_id == "E123")
    config = resolve_platform_config(
        entry,
        load_platform_profile("platforms/l40s.json"),
        load_platform_conditions(
            "platforms/execution_matrix.json", "l40s", experiment_ids={"E123"}, include_deferred=False
        )[0],
    )
    config_path = config_dir / "E123.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    profile_path = root / "profile.json"
    profile_path.write_text("{}\n", encoding="utf-8")
    bundle = SubmissionBundle(
        bundle_id="l40s-wave-1-test",
        root=root,
        source_repository=tmp_path,
        repo_dir=root / "repo",
        config_dir=config_dir,
        script_dir=root / "scripts",
        result_dir=result_dir,
        log_dir=root / "logs",
        cache_dir=root / "cache",
        manifest_path=root / "SUBMISSION_MANIFEST.json",
        result_manifest_path=result_dir / "SUBMISSION_MANIFEST.json",
        script_checksums_path=root / "SCRIPT_SHA256SUMS",
        git_commit="a" * 40,
        created_at="2026-08-30T12:00:00+00:00",
    )

    manifest = write_submission_manifest(
        bundle, platform_id="l40s", wave="1", input_files={"platform_profile": profile_path}
    )
    identities = manifest["configs"]["E123"]

    assert manifest["schema_version"] == "1.1"
    assert identities["config_sha256"] == config_sha256(config)
    assert identities["file_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert identities["file_sha256"] != identities["config_sha256"]


def test_wave_state_and_result_summary_track_complete_bundle(tmp_path: Path) -> None:
    entry = next(entry for entry in discover_experiment_configs() if entry.config.experiment_id == "E123")
    config = resolve_platform_config(
        entry,
        load_platform_profile("platforms/l40s.json"),
        load_platform_conditions(
            "platforms/execution_matrix.json", "l40s", experiment_ids={"E123"}, include_deferred=False
        )[0],
    )
    bundle_dir = tmp_path / "bundle"
    result_dir = bundle_dir / "results"
    result_dir.mkdir(parents=True)
    manifest = {
        "schema_version": "1.1",
        "bundle_id": "l40s-wave-1-test",
        "platform_id": "l40s",
        "wave": "1",
        "repository": {"git_commit": "a" * 40, "snapshot_directory": "repo"},
        "configs": {
            "E123": {
                "path": "E123.json",
                "file_sha256": "f" * 64,
                "config_sha256": config_sha256(config),
                "required_trials": 1,
            }
        },
    }
    manifest_path = result_dir / "SUBMISSION_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    result = ExperimentResult.from_config(
        config,
        status="completed",
        started_at="2026-08-30T12:00:00+00:00",
        completed_at="2026-08-30T12:01:00+00:00",
        runtime_seconds=60.0,
        repetition_number=1,
        metrics={"measured_seconds": 50.0, "gpu_hours": 50.0 / 3600, "total_tokens": 1000},
        environment={
            "git_commit": "a" * 40,
            "git_dirty": False,
            "config_sha256": config_sha256(config),
            "command": "python -m hpc_ai_perf.cli run",
            "observed_gpu_model": "NVIDIA L40S",
            "submission_bundle_id": "l40s-wave-1-test",
            "submission_bundle_manifest_sha256": manifest_hash,
            "submission_bundle_platform": "l40s",
            "submission_bundle_wave": "1",
        },
    )
    result.write_json(result_dir, preserve_trial=True)

    status = summarize_bundle_results(bundle_dir)
    state_path = tmp_path / "wave_1.json"
    append_wave_attempt(
        state_path,
        platform_id="l40s",
        wave="1",
        bundle_dir=bundle_dir,
        bundle_id="l40s-wave-1-test",
        manifest_sha256=manifest_hash,
        status_path=bundle_dir / "SLURM_STATUS.csv",
        experiment_ids=["E123"],
    )
    mark_wave_collected(state_path, ["results/raw/l40s/E123/result.json"])

    assert status.complete
    assert status.completed_trials == 1
    assert current_wave_attempt(state_path)["bundle_id"] == "l40s-wave-1-test"


def test_execution_matrix_expands_ranges_and_keeps_platforms_separate() -> None:
    l40s = load_platform_conditions("platforms/execution_matrix.json", "l40s", wave="0")
    h100 = load_platform_conditions("platforms/execution_matrix.json", "h100", wave="0")

    assert [condition.condition_id for condition in l40s] == [f"E{index:03d}@l40s" for index in range(1, 6)]
    assert [condition.condition_id for condition in h100] == [f"E{index:03d}@h100" for index in range(1, 6)]


def test_execution_matrix_routes_every_experiment_once_per_platform() -> None:
    expected_ids = {f"E{index:03d}" for index in range(1, 201)}

    for platform_id, planned_count in [("l40s", 144), ("h100", 144)]:
        conditions = load_platform_conditions("platforms/execution_matrix.json", platform_id)

        assert len(conditions) == 200
        assert {condition.matrix_id for condition in conditions} == expected_ids
        assert sum(condition.status == "planned" for condition in conditions) == planned_count


def test_execution_matrix_defers_vllm_and_cuda_toolkit_rows_on_both_platforms() -> None:
    expected_ids = {
        "E006",
        "E007",
        "E130",
        *(f"E{index:03d}" for index in range(132, 149)),
        "E178",
        *(f"E{index:03d}" for index in range(180, 197)),
    }

    for platform_id in ("l40s", "h100"):
        conditions = load_platform_conditions(
            "platforms/execution_matrix.json",
            platform_id,
            experiment_ids=expected_ids,
        )

        assert {condition.matrix_id for condition in conditions} == expected_ids
        assert {condition.status for condition in conditions} == {"deferred"}
        assert all(condition.wave == "deferred" for condition in conditions)


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
        run_output_dir=None,
        log_dir=None,
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
    assert f"#SBATCH --output={slurm_dir / 'logs'}/%x-%A_%a.out" in script
    assert "--output-dir build/platform_results/l40s/wave_1" in script
    assert "source slurm/modules_for_hpc_ai" in script
    assert "export DS_IGNORE_CUDA_DETECTION=1" in script
    assert "export PYTORCH_NVML_BASED_CUDA_CHECK=1" in script
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


def test_matrix_identity_rejects_conflicting_platform_signals(tmp_path: Path) -> None:
    entry = next(entry for entry in discover_experiment_configs() if entry.config.experiment_id == "E123")
    profile = load_platform_profile("platforms/l40s.json")
    condition = load_platform_conditions(
        "platforms/execution_matrix.json",
        "l40s",
        experiment_ids={"E123"},
        include_deferred=False,
    )[0]
    config = resolve_platform_config(entry, profile, condition)
    record = ExperimentResult.from_config(
        config,
        status="failed",
        started_at="2026-08-27T10:00:00+00:00",
        completed_at="2026-08-27T10:00:01+00:00",
        runtime_seconds=1.0,
        error="identity test",
    ).to_dict()
    record["environment"]["observed_gpu_model"] = "NVIDIA H100 SXM 80GB"
    result_path = tmp_path / "raw" / "l40s" / "E123" / "conflict.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(record), encoding="utf-8")

    issues = validate_result_records(tmp_path / "raw")

    assert any("exactly one L40S/H100 platform" in issue.message for issue in issues)
    with pytest.raises(ValueError, match="exactly one L40S/H100 platform"):
        flatten_result_record(record, result_path)


def test_legacy_matrix_identity_is_derived_conservatively(tmp_path: Path) -> None:
    entry = next(entry for entry in discover_experiment_configs() if entry.config.experiment_id == "E123")
    record = ExperimentResult.from_config(
        entry.config,
        status="failed",
        started_at="2026-08-27T10:00:00+00:00",
        completed_at="2026-08-27T10:00:01+00:00",
        runtime_seconds=1.0,
        error="legacy evidence",
    ).to_dict()
    record["schema_version"] = "1.2.0"
    for field in ["condition_id", "platform_id", "comparability"]:
        record.pop(field)
    result_path = tmp_path / "raw" / "l40s" / "E123" / "legacy.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(record), encoding="utf-8")

    flattened = flatten_result_record(record, result_path)

    assert flattened["condition_id"] == "E123@l40s"
    assert flattened["platform_id"] == "l40s"
    assert flattened["comparability"] == "platform_specific"
    assert validate_result_records(tmp_path / "raw") == []


def test_platform_contract_requires_explicit_comparability(tmp_path: Path) -> None:
    entry = next(entry for entry in discover_experiment_configs() if entry.config.experiment_id == "E123")
    record = ExperimentResult.from_config(
        entry.config,
        status="failed",
        started_at="2026-08-27T10:00:00+00:00",
        completed_at="2026-08-27T10:00:01+00:00",
        runtime_seconds=1.0,
        error="missing contract field",
    ).to_dict()
    record.pop("comparability")
    result_path = tmp_path / "E123.json"
    result_path.write_text(json.dumps(record), encoding="utf-8")

    issues = validate_result_records(tmp_path)

    assert any("must include comparability" in issue.message for issue in issues)
    with pytest.raises(ValueError, match="must include comparability"):
        flatten_result_record(record, result_path)


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


def test_import_result_evidence_is_platform_qualified_and_idempotent(tmp_path: Path) -> None:
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
        repetition_number=1,
        error="retained OOM evidence",
    )
    copied_dir = tmp_path / "copied"
    copied_dir.mkdir()
    source = copied_dir / "E123-trial.json"
    source.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    result_root = tmp_path / "results" / "raw"

    first = import_result_evidence(copied_dir, result_root)
    second = import_result_evidence(copied_dir, result_root)
    target = result_root / "l40s" / "E123" / source.name

    assert first.imported == (target,)
    assert second.unchanged == (target,)
    assert (result_root / "l40s" / "SHA256SUMS").exists()
    assert verify_result_evidence(result_root) == []

    source.write_text(source.read_text(encoding="utf-8").replace("retained OOM", "different OOM"), encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable evidence"):
        import_result_evidence(copied_dir, result_root)


def test_import_result_evidence_rejects_sensitive_fields(tmp_path: Path) -> None:
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
        error="test",
    ).to_dict()
    result["environment"]["api_key"] = "must-not-be-committed"
    source = tmp_path / "E123-secret.json"
    source.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="Sensitive field `environment.api_key`"):
        import_result_evidence(source, tmp_path / "raw")


def test_import_result_evidence_accepts_verified_submission_bundle(tmp_path: Path) -> None:
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
    staged = tmp_path / "bundle" / "results"
    staged.mkdir(parents=True)
    manifest = {
        "schema_version": "1.1",
        "bundle_id": "l40s-wave-2-test",
        "platform_id": "l40s",
        "wave": "2",
        "repository": {"git_commit": "a" * 40, "snapshot_directory": "repo"},
        "configs": {
            "E123": {
                "path": "E123.json",
                "file_sha256": "f" * 64,
                "config_sha256": config_hash,
                "required_trials": 3,
            }
        },
    }
    manifest_path = staged / "SUBMISSION_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
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
        },
        environment={
            "git_commit": "a" * 40,
            "git_dirty": False,
            "config_sha256": config_hash,
            "command": "python -m hpc_ai_perf.cli run",
            "observed_gpu_model": "NVIDIA L40S",
            "submission_bundle_id": "l40s-wave-2-test",
            "submission_bundle_manifest_sha256": manifest_hash,
            "submission_bundle_platform": "l40s",
            "submission_bundle_wave": "2",
        },
    )
    result_path = result.write_json(staged, preserve_trial=True)

    summary = import_result_evidence(staged, tmp_path / "raw")

    assert summary.imported == (tmp_path / "raw" / "l40s" / "E123" / result_path.name,)
    assert verify_result_evidence(tmp_path / "raw") == []


def test_verify_result_evidence_ignores_empty_platform_directory(tmp_path: Path) -> None:
    (tmp_path / "raw" / "h100").mkdir(parents=True)

    assert verify_result_evidence(tmp_path / "raw") == []
