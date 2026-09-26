from __future__ import annotations

import hashlib
import json
import sys
import venv
from pathlib import Path

import pytest
from scripts.check_private_torch import is_import_below_environment

from hpc_ai_perf.cli import (
    _apply_current_wave_policy,
    _bundle_setup_commands,
    _completed_planned_wave_evidence_ids,
    _incomplete_planned_wave_experiment_ids,
    _materialize_platform_batch,
    _planned_wave_experiment_ids,
    _render_platform_slurm,
    _venv_python_executable,
)
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
from hpc_ai_perf.slurm.bundle import (
    _BUNDLE_ID,
    SubmissionBundle,
    _environment_contract_problems,
    capture_environment_contract,
    write_submission_manifest,
)
from hpc_ai_perf.slurm.lifecycle import (
    BundleResultStatus,
    ExperimentResultStatus,
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
    assert l40s.max_gpus == 8
    assert h100.platform_id == "h100"
    assert h100.max_gpus == 4


def test_submission_bundle_id_accepts_documented_utc_timestamp() -> None:
    assert _BUNDLE_ID.fullmatch("l40s-wave-1-f18d325-20260829T231916Z")


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

    environment_contract = {"selected_environment": "isolated-vllm"}
    manifest = write_submission_manifest(
        bundle,
        platform_id="l40s",
        wave="1",
        input_files={"platform_profile": profile_path},
        environment_contract=environment_contract,
    )
    identities = manifest["configs"]["E123"]

    assert manifest["schema_version"] == "1.1"
    assert identities["config_sha256"] == config_sha256(config)
    assert identities["file_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert identities["file_sha256"] != identities["config_sha256"]
    assert manifest["environment_contract"] == environment_contract


def test_capture_environment_contract_can_probe_a_selected_virtualenv_python() -> None:
    contract = capture_environment_contract(python_executable=sys.executable)

    assert contract["python_executable"] == str(Path(sys.executable))
    assert isinstance(contract["packages"], dict)
    assert "transformer-engine" in contract["packages"]
    assert "loaded_modules" not in contract


def test_capture_environment_contract_uses_virtualenv_launcher(tmp_path: Path) -> None:
    environment = tmp_path / ".venv"
    venv.create(environment, with_pip=False)
    executable = environment / "bin" / "python"

    selected_executable = _venv_python_executable(environment / "bin" / "activate")
    contract = capture_environment_contract(python_executable=selected_executable)

    assert selected_executable == executable
    assert contract["python_executable"] == str(executable)


def test_bundle_setup_isolates_external_virtualenv_from_module_pythonpath(tmp_path: Path) -> None:
    repository = tmp_path / "project"
    bundle_root = tmp_path / "bundle"
    bundle = SubmissionBundle(
        bundle_id="l40s-wave-vllm-pilot-test",
        root=bundle_root,
        source_repository=repository,
        repo_dir=bundle_root / "repo",
        config_dir=bundle_root / "configs",
        script_dir=bundle_root / "scripts",
        result_dir=bundle_root / "results",
        log_dir=bundle_root / "logs",
        cache_dir=bundle_root / "cache",
        manifest_path=bundle_root / "SUBMISSION_MANIFEST.json",
        result_manifest_path=bundle_root / "results" / "SUBMISSION_MANIFEST.json",
        script_checksums_path=bundle_root / "SCRIPT_SHA256SUMS",
        git_commit="a" * 40,
        created_at="2026-09-01T20:00:00+00:00",
    )
    private_activate = tmp_path / "private-vllm" / "bin" / "activate"

    commands = _bundle_setup_commands(
        ["source slurm/modules_for_hpc_ai", "source .venv/bin/activate"],
        bundle=bundle,
        venv_activate=private_activate,
    )

    module_index = commands.index("source slurm/modules_for_hpc_ai")
    unset_index = commands.index("unset PYTHONPATH PYTHONHOME")
    activate_index = commands.index(f"source {private_activate}")
    assert module_index < unset_index < activate_index
    assert commands[-1] == f"export PYTHONPATH={bundle.repo_dir / 'src'}"


def test_bundle_setup_preserves_module_pythonpath_for_project_virtualenv(tmp_path: Path) -> None:
    repository = tmp_path / "project"
    bundle_root = tmp_path / "bundle"
    bundle = SubmissionBundle(
        bundle_id="l40s-wave-1-test",
        root=bundle_root,
        source_repository=repository,
        repo_dir=bundle_root / "repo",
        config_dir=bundle_root / "configs",
        script_dir=bundle_root / "scripts",
        result_dir=bundle_root / "results",
        log_dir=bundle_root / "logs",
        cache_dir=bundle_root / "cache",
        manifest_path=bundle_root / "SUBMISSION_MANIFEST.json",
        result_manifest_path=bundle_root / "results" / "SUBMISSION_MANIFEST.json",
        script_checksums_path=bundle_root / "SCRIPT_SHA256SUMS",
        git_commit="a" * 40,
        created_at="2026-09-01T20:00:00+00:00",
    )
    project_activate = repository / ".venv" / "bin" / "activate"

    commands = _bundle_setup_commands(
        ["source slurm/modules_for_hpc_ai", "source .venv/bin/activate"],
        bundle=bundle,
        venv_activate=project_activate,
    )

    assert "unset PYTHONPATH PYTHONHOME" not in commands
    assert commands[-1] == f"export PYTHONPATH={bundle.repo_dir / 'src'}${{PYTHONPATH:+:$PYTHONPATH}}"


def test_environment_contract_ignores_inherited_module_state() -> None:
    expected = {
        "python_version": "3.11.11",
        "packages": {"torch": "2.8.0"},
        "torch_cuda": "12.4",
        "loaded_modules": "submit-shell-only-module",
    }
    observed = {**expected, "loaded_modules": "slurm-job-module-state"}

    assert _environment_contract_problems(expected, observed) == []


def test_environment_contract_reports_runtime_identity_changes() -> None:
    expected = {
        "python_version": "3.11.11",
        "packages": {"torch": "2.8.0"},
        "torch_cuda": "12.4",
    }
    observed = {**expected, "packages": {"torch": "2.9.0"}, "torch_cuda": "12.5"}

    assert _environment_contract_problems(expected, observed) == [
        "runtime environment contract differs for packages",
        "runtime environment contract differs for torch_cuda",
    ]


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

    assert [condition.condition_id for condition in l40s] == ["E001@l40s", "E002@l40s", "E004@l40s", "E005@l40s"]
    assert [condition.condition_id for condition in h100] == [f"E{index:03d}@h100" for index in range(1, 6)]


def test_execution_matrix_routes_every_experiment_once_per_platform() -> None:
    source_ids = {f"E{index:03d}" for index in range(1, 201)}
    supplemental_ids = {f"E{index:03d}" for index in range(201, 274)}
    l40s_only_supplemental_ids = (
        {f"E{index:03d}" for index in range(201, 210)}
        | {f"E{index:03d}" for index in range(215, 233)}
        | {f"E{index:03d}" for index in range(235, 239)}
    )
    shared_telemetry_ids = {f"E{index:03d}" for index in range(210, 215)}
    h100_only_supplemental_ids = {"E233", "E234"} | {f"E{index:03d}" for index in range(250, 274)}
    shared_lowbit_ids = {f"E{index:03d}" for index in range(239, 247)}
    shared_long_context_ids = {f"E{index:03d}" for index in range(247, 250)}

    l40s_conditions = load_platform_conditions("platforms/execution_matrix.json", "l40s")
    h100_conditions = load_platform_conditions("platforms/execution_matrix.json", "h100")

    assert len([condition for condition in l40s_conditions if condition.matrix_id in source_ids]) == 200
    assert {condition.matrix_id for condition in l40s_conditions if condition.matrix_id in source_ids} == source_ids
    assert {
        condition.matrix_id for condition in l40s_conditions if condition.matrix_id not in source_ids
    } == l40s_only_supplemental_ids | shared_telemetry_ids | shared_lowbit_ids | shared_long_context_ids
    assert len(h100_conditions) == 242
    assert {
        condition.matrix_id for condition in h100_conditions
    } == source_ids | shared_telemetry_ids | h100_only_supplemental_ids | shared_lowbit_ids | shared_long_context_ids
    l40s_planned_source_count = sum(
        condition.status == "planned" for condition in l40s_conditions if condition.matrix_id in source_ids
    )

    assert l40s_planned_source_count == 180
    assert (
        sum(condition.status == "planned" for condition in l40s_conditions if condition.matrix_id in supplemental_ids)
        == 43
    )
    assert (
        sum(
            condition.status == "planned"
            for condition in l40s_conditions
            if condition.matrix_id in l40s_only_supplemental_ids
        )
        == 27
    )
    assert sum(condition.status == "planned" for condition in h100_conditions) == len(h100_conditions) - 18


def test_telemetry_revalidation_wave_is_shared_but_platform_separated() -> None:
    l40s = load_platform_conditions(
        "platforms/execution_matrix.json", "l40s", wave="telemetry-revalidation", include_deferred=False
    )
    h100 = load_platform_conditions(
        "platforms/execution_matrix.json", "h100", wave="telemetry-revalidation", include_deferred=False
    )

    assert [condition.matrix_id for condition in l40s] == ["E210", "E211", "E212", "E213"]
    assert [condition.condition_id for condition in l40s] == [f"E{index:03d}@l40s" for index in range(210, 214)]
    assert [condition.condition_id for condition in h100] == [f"E{index:03d}@h100" for index in range(210, 214)]
    assert all(condition.comparability == "platform_specific" for condition in [*l40s, *h100])


def test_energy_revalidation_wave_uses_new_immutable_condition() -> None:
    l40s = load_platform_conditions(
        "platforms/execution_matrix.json", "l40s", wave="energy-revalidation", include_deferred=False
    )
    h100 = load_platform_conditions(
        "platforms/execution_matrix.json", "h100", wave="energy-revalidation", include_deferred=False
    )

    assert [condition.condition_id for condition in l40s] == ["E214@l40s"]
    assert [condition.condition_id for condition in h100] == ["E214@h100"]
    assert all(condition.required_trials == 3 for condition in [*l40s, *h100])
    assert all(condition.comparability == "platform_specific" for condition in [*l40s, *h100])


def test_hardware_energy_revalidation_uses_new_l40s_only_conditions() -> None:
    conditions = load_platform_conditions(
        "platforms/execution_matrix.json",
        "l40s",
        wave="hardware-energy-revalidation",
        include_deferred=False,
    )

    assert [condition.condition_id for condition in conditions] == [
        "E235@l40s",
        "E236@l40s",
        "E237@l40s",
        "E238@l40s",
    ]
    assert {condition.required_trials for condition in conditions} == {3}
    assert {condition.comparability for condition in conditions} == {"platform_specific"}
    assert not load_platform_conditions(
        "platforms/execution_matrix.json",
        "h100",
        wave="hardware-energy-revalidation",
        include_deferred=False,
    )


def test_runnable_l40s_conditions_fit_the_verified_eight_gpu_node() -> None:
    profile = load_platform_profile("platforms/l40s.json")
    entries = {entry.config.experiment_id: entry for entry in discover_experiment_configs()}
    conditions = load_platform_conditions("platforms/execution_matrix.json", "l40s")

    planned = [condition for condition in conditions if condition.status == "planned"]

    assert planned
    assert all(entries[condition.matrix_id].config.hardware.gpus <= profile.max_gpus for condition in planned)
    wave_two = [condition.matrix_id for condition in planned if condition.wave == "2"]
    assert wave_two == ["E123", "E124", "E127", "E128"]


def test_l40s_four_gpu_conditions_use_dedicated_release_waves() -> None:
    entries = {entry.config.experiment_id: entry for entry in discover_experiment_configs()}
    expected = {
        "4gpu-pilot": ({"E003"}, 1, "platform_specific"),
        "4gpu-core": (
            {"E034", "E037", "E040", "E043", "E046", "E048", "E050", "E056", "E058", "E172"},
            3,
            "exact",
        ),
        "4gpu-communication": ({"E115", "E116", "E119", "E120"}, 3, "platform_specific"),
        "vllm-tp4": ({"E148", "E196"}, 3, "exact"),
    }

    for wave, (expected_ids, required_trials, comparability) in expected.items():
        conditions = load_platform_conditions(
            "platforms/execution_matrix.json",
            "l40s",
            wave=wave,
            include_deferred=False,
        )
        assert {condition.matrix_id for condition in conditions} == expected_ids
        assert {condition.required_trials for condition in conditions} == {required_trials}
        assert {condition.comparability for condition in conditions} == {comparability}
        assert all(entries[condition.matrix_id].config.hardware.gpus == 4 for condition in conditions)


def test_h100_resource_homogeneous_waves_separate_two_and_four_gpu_work() -> None:
    entries = {entry.config.experiment_id: entry for entry in discover_experiment_configs()}
    expected = {
        "4b-2gpu": ({"E113", "E114", "E117", "E118"}, 2, "platform_specific"),
        "4b-4gpu": ({"E115", "E116", "E119", "E120"}, 4, "platform_specific"),
        "optional-1to2gpu": ({"E016", "E017", "E044", "E045"}, None, "exact"),
        "optional-4gpu": ({"E046"}, 4, "exact"),
    }

    for wave, (expected_ids, expected_gpus, comparability) in expected.items():
        conditions = load_platform_conditions(
            "platforms/execution_matrix.json",
            "h100",
            wave=wave,
            include_deferred=False,
        )
        assert {condition.matrix_id for condition in conditions} == expected_ids
        assert {condition.required_trials for condition in conditions} == {3}
        assert {condition.comparability for condition in conditions} == {comparability}
        gpu_counts = {entries[condition.matrix_id].config.hardware.gpus for condition in conditions}
        if expected_gpus is not None:
            assert gpu_counts == {expected_gpus}
        else:
            assert gpu_counts <= {1, 2}


def test_l40s_eight_gpu_conditions_use_architecture_aware_release_waves() -> None:
    entries = {entry.config.experiment_id: entry for entry in discover_experiment_configs()}
    expected = {
        "8gpu-pilot": ({"E215"}, 1, "platform_specific"),
        "8gpu-scaling": ({"E216", "E217", "E218", "E219", "E220"}, 3, "exact"),
        "8gpu-strategy": ({"E221", "E222", "E223", "E224"}, 3, "exact"),
        "8gpu-communication": ({"E225", "E226", "E227", "E228"}, 3, "platform_specific"),
        "vllm-tp8": ({"E229"}, 3, "exact"),
        "vllm-tp8-canary": ({"E230"}, 1, "platform_specific"),
    }

    for wave, (expected_ids, required_trials, comparability) in expected.items():
        conditions = load_platform_conditions(
            "platforms/execution_matrix.json",
            "l40s",
            wave=wave,
            include_deferred=False,
        )
        assert {condition.matrix_id for condition in conditions} == expected_ids
        assert {condition.required_trials for condition in conditions} == {required_trials}
        assert {condition.comparability for condition in conditions} == {comparability}
        assert all(entries[condition.matrix_id].config.hardware.gpus == 8 for condition in conditions)

    assert entries["E215"].config.workload.parameters["capture_gpu_topology"] is True
    assert entries["E229"].config.workload.parameters["tensor_parallel_size"] == 8
    assert entries["E230"].config.workload.parameters["tensor_parallel_size"] == 8


def test_l40s_fp8_training_wave_pairs_transformer_engine_controls_and_treatments() -> None:
    conditions = load_platform_conditions(
        "platforms/execution_matrix.json",
        "l40s",
        wave="training-fp8",
        include_deferred=False,
    )

    assert {condition.matrix_id for condition in conditions} == {"E086", "E093", "E231", "E232"}
    assert {condition.required_trials for condition in conditions} == {3}
    assert {condition.comparability for condition in conditions} == {"exact"}

    h100 = load_platform_conditions(
        "platforms/execution_matrix.json",
        "h100",
        experiment_ids={"E086", "E093"},
    )
    assert {condition.status for condition in h100} == {"planned"}
    assert {condition.wave for condition in h100} == {"training-fp8"}


def test_h100_fp8_training_wave_uses_h100_backend_matched_controls() -> None:
    conditions = load_platform_conditions(
        "platforms/execution_matrix.json",
        "h100",
        wave="training-fp8",
        include_deferred=False,
    )
    assert {condition.matrix_id for condition in conditions} == {"E086", "E093", "E233", "E234"}
    assert {condition.required_trials for condition in conditions} == {3}
    assert {condition.comparability for condition in conditions} == {"exact"}

    profile = load_platform_profile("platforms/h100.json")
    entries = {entry.config.experiment_id: entry for entry in discover_experiment_configs()}
    resolved = {
        condition.matrix_id: resolve_platform_config(entries[condition.matrix_id], profile, condition)
        for condition in conditions
    }

    assert resolved["E086"].workload.parameters["comparison_control_id"] == "E233"
    assert resolved["E093"].workload.parameters["comparison_control_id"] == "E234"
    assert resolved["E086"].metadata["platform_workload_parameter_overrides"] == {"comparison_control_id": "E233"}
    assert resolved["E233"].workload.parameters["comparison_treatment_id"] == "E086"
    assert resolved["E234"].workload.parameters["comparison_treatment_id"] == "E093"
    assert resolved["E233"].hardware.gpu_model == "NVIDIA H100 SXM 80GB"


def test_l40s_wave_three_collection_scope_excludes_gpt2_strong_scaling_oom_deferrals() -> None:
    wave_three = _planned_wave_experiment_ids("l40s", "3")

    assert len(wave_three) == 18
    assert "E032" not in wave_three
    assert "E033" not in wave_three
    assert {"E036", "E049", "E062"} <= wave_three


def test_h100_wave_three_collection_scope_excludes_measured_gpt2_oom_deferral() -> None:
    wave_three = _planned_wave_experiment_ids("h100", "3")

    assert len(wave_three) == 27
    assert "E032" not in wave_three
    assert {"E033", "E034", "E062"} <= wave_three


def test_l40s_wave_four_collection_scope_excludes_exact_micro_batch_oom_deferrals() -> None:
    wave_four = _planned_wave_experiment_ids("l40s", "4")

    assert len(wave_four) == 44
    assert "E067" not in wave_four
    assert "E068" not in wave_four
    assert "E096" not in wave_four
    assert {"E064", "E074", "E112"} <= wave_four


def test_h100_wave_four_collection_scope_excludes_measured_micro_batch_oom_deferral() -> None:
    wave_four = _planned_wave_experiment_ids("h100", "4")

    assert len(wave_four) == 45
    assert "E068" not in wave_four
    assert "E096" not in wave_four
    assert {"E063", "E067", "E112"} <= wave_four


def test_l40s_capacity_adjusted_wave_materializes_boundary_probes(tmp_path: Path) -> None:
    profile = load_platform_profile("platforms/l40s.json")
    conditions = load_platform_conditions(
        "platforms/execution_matrix.json",
        "l40s",
        wave="4c",
        include_deferred=False,
    )

    paths = materialize_platform_configs(profile, conditions, tmp_path)

    assert [path.name for path in paths] == ["E201.json", "E202.json", "E205.json"]
    config = load_experiment_config(paths[-1])
    assert config.experiment_id == "E205"
    assert config.workload.parameters["micro_batch_size"] == 25
    assert config.metadata["condition_id"] == "E205@l40s"
    assert config.metadata["comparability"] == "capacity_adjusted"
    assert config.runtime.output_dir == "results/raw/l40s"


def test_h100_capacity_adjusted_wave_preserves_the_global_workload(tmp_path: Path) -> None:
    profile = load_platform_profile("platforms/h100.json")
    scaling_conditions = load_platform_conditions(
        "platforms/execution_matrix.json",
        "h100",
        wave="h100-capacity",
        include_deferred=False,
    )
    boundary_conditions = load_platform_conditions(
        "platforms/execution_matrix.json",
        "h100",
        wave="h100-capacity-boundary",
        include_deferred=False,
    )
    allocator_conditions = load_platform_conditions(
        "platforms/execution_matrix.json",
        "h100",
        wave="h100-capacity-allocator-canary",
        include_deferred=False,
    )

    scaling_path = materialize_platform_configs(profile, scaling_conditions, tmp_path / "scaling")[0]
    boundary_path = materialize_platform_configs(profile, boundary_conditions, tmp_path / "boundary")[0]
    allocator_path = materialize_platform_configs(profile, allocator_conditions, tmp_path / "allocator")[0]

    assert scaling_path.name == "E256.json"
    assert boundary_path.name == "E258.json"
    assert allocator_path.name == "E259.json"
    scaling = load_experiment_config(scaling_path)
    memory = load_experiment_config(boundary_path)
    allocator = load_experiment_config(allocator_path)
    assert scaling.metadata["condition_id"] == "E256@h100"
    assert scaling.metadata["comparability"] == "capacity_adjusted"
    assert scaling.workload.parameters["per_gpu_batch_size"] == 32
    assert scaling.workload.parameters["gradient_accumulation_steps"] == 2
    assert scaling.workload.parameters["global_batch_size"] == 64
    assert memory.metadata["condition_id"] == "E258@h100"
    assert memory.metadata["comparability"] == "capacity_adjusted"
    assert memory.workload.parameters["micro_batch_size"] == 40
    assert allocator.metadata["condition_id"] == "E259@h100"
    assert allocator.runtime.environment == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    assert all(config.runtime.output_dir == "results/raw/h100" for config in [scaling, memory, allocator])


def test_l40s_checkpointed_scaling_wave_preserves_original_h100_protocol(tmp_path: Path) -> None:
    l40s_profile = load_platform_profile("platforms/l40s.json")
    l40s_conditions = load_platform_conditions(
        "platforms/execution_matrix.json",
        "l40s",
        wave="optional-capacity",
        include_deferred=False,
    )

    adjusted_paths = materialize_platform_configs(l40s_profile, l40s_conditions, tmp_path / "l40s")

    assert [path.name for path in adjusted_paths] == ["E208.json", "E209.json"]
    adjusted_configs = [load_experiment_config(path) for path in adjusted_paths]
    assert [config.hardware.gpus for config in adjusted_configs] == [1, 2]
    assert [config.workload.parameters["per_gpu_batch_size"] for config in adjusted_configs] == [16, 8]
    assert all(config.workload.parameters["activation_checkpointing"] is True for config in adjusted_configs)
    assert all(config.workload.parameters["global_batch_size"] == 16 for config in adjusted_configs)
    assert all(config.metadata["comparability"] == "capacity_adjusted" for config in adjusted_configs)

    l40s_e044 = next(
        condition
        for condition in load_platform_conditions("platforms/execution_matrix.json", "l40s")
        if condition.matrix_id == "E044"
    )
    assert l40s_e044.status == "deferred"

    h100_profile = load_platform_profile("platforms/h100.json")
    h100_e044 = next(
        condition
        for condition in load_platform_conditions(
            "platforms/execution_matrix.json",
            "h100",
            wave="optional-1to2gpu",
            include_deferred=False,
        )
        if condition.matrix_id == "E044"
    )
    h100_path = materialize_platform_configs(h100_profile, [h100_e044], tmp_path / "h100")[0]
    h100_config = load_experiment_config(h100_path)

    assert h100_config.metadata["comparability"] == "exact"
    assert h100_config.hardware.gpu_model == "NVIDIA H100 SXM 80GB"
    assert h100_config.workload.parameters.get("activation_checkpointing", False) is False
    assert h100_config.workload.parameters["per_gpu_batch_size"] == 16


def test_wave_status_stops_retrying_conditions_deferred_after_submission() -> None:
    historical = BundleResultStatus(
        Path("bundle"),
        (
            ExperimentResultStatus("E016", 3, (1, 2, 3), (), 0),
            ExperimentResultStatus("E017", 3, (1, 2, 3), (), 0),
            ExperimentResultStatus("E044", 3, (), (1, 2, 3), 0),
            ExperimentResultStatus("E045", 3, (1, 2, 3), (), 0),
        ),
    )

    current = _apply_current_wave_policy(historical, "l40s", "optional")

    assert current.complete
    assert current.completed_trials == 9
    assert current.expected_trials == 9
    assert [experiment.experiment_id for experiment in current.experiments] == ["E016", "E017", "E045"]


def test_h100_e172_returns_to_wave_five_when_full_node_is_available() -> None:
    wave_five = load_platform_conditions(
        "platforms/execution_matrix.json",
        "h100",
        wave="5",
        include_deferred=False,
    )
    e172 = next(condition for condition in wave_five if condition.matrix_id == "E172")
    assert e172.status == "planned"
    assert e172.required_trials == 3


def test_h100_qwen7b_sharding_study_has_a_separate_canary_and_controlled_pairs() -> None:
    canaries = load_platform_conditions(
        "platforms/execution_matrix.json",
        "h100",
        wave="qwen7b-training-canary",
        include_deferred=False,
    )
    canonical = load_platform_conditions(
        "platforms/execution_matrix.json",
        "h100",
        wave="qwen7b-training",
        include_deferred=False,
    )
    entries = {entry.config.experiment_id: entry.config for entry in discover_experiment_configs()}

    assert {condition.matrix_id for condition in canaries} == {"E250", "E251"}
    assert {condition.required_trials for condition in canaries} == {1}
    assert {condition.matrix_id for condition in canonical} == {"E252", "E253", "E254", "E255"}
    assert {condition.required_trials for condition in canonical} == {3}
    assert {condition.comparability for condition in [*canaries, *canonical]} == {"platform_specific"}

    for experiment_id in {"E250", "E251", "E252", "E253", "E254", "E255"}:
        config = entries[experiment_id]
        assert config.workload.name == "qwen2_5_7b_training"
        assert config.runtime.precision == "bf16"
        assert config.workload.parameters["model_load_dtype"] == "bf16"
        assert config.workload.parameters["effective_batch_size"] == 16
        assert config.workload.parameters["activation_checkpointing"] is True

    for fsdp_id, zero_id, gpu_count, accumulation in (("E252", "E253", 2, 8), ("E254", "E255", 4, 4)):
        fsdp, zero = entries[fsdp_id], entries[zero_id]
        assert fsdp.hardware.gpus == zero.hardware.gpus == gpu_count
        assert fsdp.workload.parameters["gradient_accumulation_steps"] == accumulation
        assert zero.workload.parameters["gradient_accumulation_steps"] == accumulation
        assert fsdp.distributed.strategy == "fsdp"
        assert zero.distributed.strategy == "deepspeed_zero3"


def test_retry_scope_includes_new_policy_conditions_missing_from_prior_attempt(monkeypatch, tmp_path: Path) -> None:
    state_path = tmp_path / "wave_4c.json"
    state_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "hpc_ai_perf.cli._planned_wave_experiment_ids",
        lambda platform_id, wave: {"E201", "E202", "E205"},
    )
    monkeypatch.setattr(
        "hpc_ai_perf.cli._completed_planned_wave_evidence_ids",
        lambda platform_id, wave, result_dir: set(),
    )
    monkeypatch.setattr(
        "hpc_ai_perf.cli.load_wave_state",
        lambda path: {"attempts": [{"bundle_dir": str(tmp_path / "old_bundle")}]},
    )
    monkeypatch.setattr(
        "hpc_ai_perf.cli.summarize_bundle_results",
        lambda bundle_dir: BundleResultStatus(
            Path(bundle_dir),
            (
                ExperimentResultStatus("E201", 3, (1, 2, 3), (), 0),
                ExperimentResultStatus("E202", 3, (1, 2, 3), (), 0),
                ExperimentResultStatus("E203", 3, (), (1, 2, 3), 0),
                ExperimentResultStatus("E204", 3, (), (1, 2, 3), 0),
            ),
        ),
    )

    retry_ids = _incomplete_planned_wave_experiment_ids("l40s", "4c", state_path)

    assert retry_ids == ["E205"]


def test_retry_scope_bootstraps_from_accepted_evidence_without_state(monkeypatch, tmp_path: Path) -> None:
    state_path = tmp_path / "missing-wave-state.json"
    monkeypatch.setattr(
        "hpc_ai_perf.cli._planned_wave_experiment_ids",
        lambda platform_id, wave: {"E001", "E002", "E004", "E005"},
    )
    monkeypatch.setattr(
        "hpc_ai_perf.cli._completed_planned_wave_evidence_ids",
        lambda platform_id, wave, result_dir: {"E001"},
    )

    retry_ids = _incomplete_planned_wave_experiment_ids("l40s", "0", state_path)

    assert retry_ids == ["E002", "E004", "E005"]


def test_completed_wave_evidence_uses_validated_platform_tracker_rows(monkeypatch) -> None:
    monkeypatch.setattr(
        "hpc_ai_perf.cli.load_platform_conditions",
        lambda *args, **kwargs: [object()],
    )
    monkeypatch.setattr(
        "hpc_ai_perf.cli.build_platform_progress_rows",
        lambda conditions, result_dir: [
            {"matrix_id": "E001", "status": "Complete"},
            {"matrix_id": "E002", "status": "Ready"},
            {"matrix_id": "E004", "status": "Failed"},
        ],
    )

    completed = _completed_planned_wave_evidence_ids("l40s", "0")

    assert completed == {"E001"}


def test_execution_matrix_defers_only_unaccepted_or_unsupported_backend_and_platform_rows() -> None:
    shared_deferred_ids = {
        "E007",
    }

    for platform_id in ("l40s", "h100"):
        expected_ids = set(shared_deferred_ids)
        if platform_id == "l40s":
            expected_ids.update(
                {
                    "E032",
                    "E033",
                    "E067",
                    "E068",
                    "E203",
                    "E204",
                    "E206",
                    "E207",
                }
            )
        else:
            expected_ids.update({"E032", "E068", "E257"})
        conditions = load_platform_conditions(
            "platforms/execution_matrix.json",
            platform_id,
            experiment_ids=expected_ids,
        )

        assert {condition.matrix_id for condition in conditions} == expected_ids
        assert {condition.status for condition in conditions} == {"deferred"}
        assert all(condition.wave == "deferred" for condition in conditions)


def test_execution_matrix_schedules_accepted_vllm_scope_in_platform_separated_waves() -> None:
    later_waves = {
        "vllm-framework": {"E130", "E178"},
        "vllm-fp8": {"E151", "E199"},
        "vllm-batching": {
            *(f"E{index:03d}" for index in range(132, 138)),
            *(f"E{index:03d}" for index in range(180, 186)),
        },
        "vllm-concurrency": {
            *(f"E{index:03d}" for index in range(138, 143)),
            *(f"E{index:03d}" for index in range(186, 191)),
        },
        "vllm-prompt": {
            *(f"E{index:03d}" for index in range(143, 147)),
            *(f"E{index:03d}" for index in range(191, 195)),
        },
        "vllm-tp2": {"E147", "E195"},
        "vllm-tp4": {"E148", "E196"},
    }
    for platform_id in ("l40s", "h100"):
        pilot = load_platform_conditions("platforms/execution_matrix.json", platform_id, wave="vllm-pilot")
        assert [(condition.matrix_id, condition.required_trials) for condition in pilot] == [("E006", 1)]
        assert {condition.condition_id for condition in pilot} == {f"E006@{platform_id}"}

        for wave, expected_ids in later_waves.items():
            wave_conditions = load_platform_conditions("platforms/execution_matrix.json", platform_id, wave=wave)
            assert {condition.matrix_id for condition in wave_conditions} == expected_ids
            assert {condition.required_trials for condition in wave_conditions} == {3}
            assert {condition.comparability for condition in wave_conditions} == {"exact"}
            assert {condition.condition_id for condition in wave_conditions} == {
                f"{experiment_id}@{platform_id}" for experiment_id in expected_ids
            }

    all_vllm_ids = {"E006"}
    for expected_ids in later_waves.values():
        all_vllm_ids.update(expected_ids)
    assert len(all_vllm_ids) == 39
    assert {"E148", "E196"} <= all_vllm_ids


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
    assert "DS_IGNORE_CUDA_DETECTION" not in script
    assert "export PYTORCH_NVML_BASED_CUDA_CHECK=1" in script
    assert f"--config {config_dir / 'E008.json'}" in script


def test_render_l40s_four_gpu_pilot_requests_four_gpus_and_thirty_two_cpus(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    slurm_dir = tmp_path / "slurm"

    _, script_paths = _render_platform_slurm(
        platform_profile="platforms/l40s.json",
        execution_matrix="platforms/execution_matrix.json",
        config_root="configs",
        config_output_dir=str(config_dir),
        output_dir=str(slurm_dir),
        run_output_dir=None,
        log_dir=None,
        wave="4gpu-pilot",
        experiment_ids={"E003"},
        profile_path=None,
        partition=None,
        account=None,
        time_limit=None,
        cpus_per_task=None,
        modules=[],
        env_values=[],
    )

    script = script_paths[0].read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu_l40s" in script
    assert "#SBATCH --gres=gpu:4" in script
    assert "#SBATCH --cpus-per-task=32" in script
    assert "#SBATCH --array" not in script


def test_render_l40s_eight_gpu_pilot_requests_full_node_capacity(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    slurm_dir = tmp_path / "slurm"

    _, script_paths = _render_platform_slurm(
        platform_profile="platforms/l40s.json",
        execution_matrix="platforms/execution_matrix.json",
        config_root="configs",
        config_output_dir=str(config_dir),
        output_dir=str(slurm_dir),
        run_output_dir=None,
        log_dir=None,
        wave="8gpu-pilot",
        experiment_ids={"E215"},
        profile_path=None,
        partition=None,
        account=None,
        time_limit=None,
        cpus_per_task=None,
        modules=[],
        env_values=[],
    )

    script = script_paths[0].read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu_l40s" in script
    assert "#SBATCH --gres=gpu:8" in script
    assert "#SBATCH --cpus-per-task=64" in script
    assert "#SBATCH --array" not in script


def test_lovelace_module_helper_uses_verified_cuda_build_stack() -> None:
    helper = Path("slurm/modules_for_hpc_ai").read_text(encoding="utf-8")

    assert "module load spack-modules/217" in helper
    assert "module load py-torch/2.8.0-rmohf3g" in helper
    assert "unset DS_IGNORE_CUDA_DETECTION" in helper
    assert 'export CUDA_HOME="$(cd "$(dirname "$_hpc_ai_nvcc")/.." && pwd)"' in helper


def test_private_torch_check_rejects_site_module_import(tmp_path: Path) -> None:
    environment = tmp_path / "vllm-0.10.2-gloo"
    private_torch = environment / "lib/python3.11/site-packages/torch/__init__.py"
    site_torch = tmp_path / "spack/py-torch/lib/python3.11/site-packages/torch/__init__.py"

    assert is_import_below_environment(private_torch, environment)
    assert not is_import_below_environment(site_torch, environment)


def test_private_gloo_torch_and_vllm_builds_preserve_the_shared_venv() -> None:
    torch_builder = Path("slurm/setup_pytorch_gloo_environment.sh").read_text(encoding="utf-8")
    torch_build_job = Path("slurm/smoke/pytorch_gloo_build_l40s.sbatch").read_text(encoding="utf-8")
    builder = Path("slurm/setup_vllm_environment.sh").read_text(encoding="utf-8")
    build_job = Path("slurm/smoke/vllm_build_l40s.sbatch").read_text(encoding="utf-8")
    acceptance = Path("slurm/smoke/vllm_acceptance.sbatch").read_text(encoding="utf-8")
    checker = Path("scripts/check_private_torch.py").read_text(encoding="utf-8")
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert 'TORCH_VERSION="${HPC_AI_TORCH_VERSION:-2.8.0}"' in torch_builder
    assert "unset PYTHONPATH PYTHONHOME" in torch_builder
    assert '"$BOOTSTRAP_PYTHON" -m venv "$TORCH_ENV"' in torch_builder
    assert '-m venv --system-site-packages "$TORCH_ENV"' not in torch_builder
    assert "export USE_DISTRIBUTED=1" in torch_builder
    assert "export USE_GLOO=1" in torch_builder
    assert "export USE_NCCL=1" in torch_builder
    assert "export USE_OPENMP=0" in torch_builder
    assert "export USE_FBGEMM=0" in torch_builder
    assert "export USE_MKLDNN=0" in torch_builder
    assert '"cmake>=3.27,<4"' in torch_builder
    assert "build-openmp-incompatible-" in torch_builder
    assert "Preserved incompatible OpenMP build cache" in torch_builder
    assert "export TORCH_CUDA_ARCH_LIST" in torch_builder
    assert 'setup.py bdist_wheel --dist-dir "$TORCH_WHEEL_DIR"' in torch_builder
    assert '"BUILD_MANIFEST.json"' in torch_builder
    assert "check_private_torch.py" in torch_builder
    assert "--require-gloo" in torch_builder
    assert "--require-nccl" in torch_builder
    assert "--require-cuda" in torch_builder
    assert "torch.__file__" in checker
    assert "dist.is_gloo_available()" in checker
    assert "dist.is_nccl_available()" in checker
    assert "process_group_smoke" in checker
    assert "--smoke-process-groups" in torch_builder
    assert "#SBATCH --partition=gpu_l40s" in torch_build_job
    assert "setup_pytorch_gloo_environment.sh" in torch_build_job

    assert 'VLLM_VERSION="${HPC_AI_VLLM_VERSION:-0.10.2}"' in builder
    assert 'VLLM_CLANG_BRACKET_DEPTH="${HPC_AI_VLLM_CLANG_BRACKET_DEPTH:-1024}"' in builder
    assert 'VLLM_TEXT_ONLY="${HPC_AI_VLLM_TEXT_ONLY:-1}"' in builder
    assert 'VLLM_TRANSFORMERS_VERSION="${HPC_AI_VLLM_TRANSFORMERS_VERSION:-4.57.6}"' in builder
    assert 'VLLM_TOKENIZERS_VERSION="${HPC_AI_VLLM_TOKENIZERS_VERSION:-0.22.1}"' in builder
    assert '-m venv --system-site-packages "$VLLM_ENV"' not in builder
    assert '"$BOOTSTRAP_PYTHON" -m venv "$VLLM_ENV"' in builder
    assert "unset PYTHONPATH PYTHONHOME" in builder
    assert 'TORCH_WHEEL_DIR="${HPC_AI_TORCH_WHEEL_DIR' in builder
    assert "Private Torch build manifest verified" in builder
    assert 'uv pip install --python "$VLLM_ENV/bin/python" "$TORCH_WHEEL"' in builder
    assert "check_private_torch.py" in builder
    assert "--smoke-process-groups" in builder
    assert '"setuptools>=77,<81"' in builder
    assert '"setuptools_scm[toml]>=8,<9"' in builder
    assert 'export PYTHONPATH="$VLLM_SITE_PACKAGES"' in builder
    assert "Using private setuptools" in builder
    assert "uv --verbose pip install" in builder
    assert 'uv pip install --python "$VLLM_ENV/bin/python" \\' in builder
    assert '"transformers==${VLLM_TRANSFORMERS_VERSION}"' in builder
    assert '"tokenizers==${VLLM_TOKENIZERS_VERSION}"' in builder
    assert "Version(installed_vllm_version).base_version" in builder
    assert "pip uninstall -y torchaudio torchvision" in builder
    assert "from transformers import ProcessorMixin" in builder
    assert "from vllm import LLM, SamplingParams" in builder
    assert 'installed_transformers_version != "$VLLM_TRANSFORMERS_VERSION"' in builder
    assert 'installed_tokenizers_version != "$VLLM_TOKENIZERS_VERSION"' in builder
    assert "TORCH_CUDA_ARCH_LIST" in builder
    assert "-Xcompiler=-fbracket-depth=${VLLM_CLANG_BRACKET_DEPTH}" in builder
    assert "#SBATCH --partition=gpu_l40s" in build_job
    assert "setup_vllm_environment.sh" in build_job
    assert "unset PYTHONPATH PYTHONHOME" in acceptance
    assert "check_private_torch.py" in acceptance
    assert "--smoke-process-groups" in acceptance
    assert "--require-vllm" in acceptance
    assert "E006_vllm_1gpu.json" in acceptance
    assert "build/vllm_acceptance" in acceptance
    assert 'os.environ["SLURM_JOB_ID"]' in acceptance
    assert 'for key in ("vllm_import", "engine_start", "request_served")' in acceptance
    assert "metrics.get(key) is not True" in acceptance
    assert "E006 acceptance failed" in acceptance
    assert "setup-pytorch-gloo-environment:" in makefile
    assert "submit-pytorch-gloo-build:" in makefile
    assert "submit-vllm-build:" in makefile
    assert "submit-vllm-acceptance:" in makefile
    assert "submit-vllm-pilot:" in makefile
    assert "submit-vllm-wave:" in makefile
    assert "retry-vllm-wave:" in makefile
    assert 'VENV_ACTIVATE="$(VLLM_ENV)/bin/activate"' in makefile
    assert "HPC_AI_VLLM_SOURCE" in makefile
    assert "--venv-activate $(VENV_ACTIVATE)" in makefile


def test_render_platform_slurm_reuses_frozen_bundle_configs_without_rewriting(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_paths = _materialize_platform_batch(
        platform_profile="platforms/l40s.json",
        execution_matrix="platforms/execution_matrix.json",
        config_root="configs",
        output_dir=str(config_dir),
        wave="1",
        experiment_ids={"E008"},
        include_deferred=False,
    )
    frozen_bytes = config_paths[0].read_bytes()

    _render_platform_slurm(
        platform_profile="platforms/l40s.json",
        execution_matrix="platforms/execution_matrix.json",
        config_root="configs",
        config_output_dir=str(config_dir),
        output_dir=str(tmp_path / "slurm"),
        run_output_dir=str(tmp_path / "bundle" / "results"),
        log_dir=str(tmp_path / "bundle" / "logs"),
        wave="1",
        experiment_ids={"E008"},
        profile_path=None,
        partition=None,
        account=None,
        time_limit=None,
        cpus_per_task=None,
        modules=[],
        env_values=[],
        reuse_resolved_configs=True,
    )

    assert config_paths[0].read_bytes() == frozen_bytes


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
    assert flattened["objective"] == entry.config.metadata["objective"]
    assert flattened["key_metrics"] == entry.config.metadata["key_metrics"]


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
