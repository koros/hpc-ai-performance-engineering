import hashlib
import json
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from hpc_ai_perf import cli
from hpc_ai_perf.config import config_sha256, discover_experiment_configs
from hpc_ai_perf.config.platform import load_platform_conditions, load_platform_profile, resolve_platform_config
from hpc_ai_perf.results import ExperimentResult, verify_result_evidence
from hpc_ai_perf.slurm.lifecycle import append_wave_attempt, load_wave_state, summarize_bundle_results
from hpc_ai_perf.tracking import build_platform_progress_rows


def write_bundle(root, configs, trials):
    results = root / "results"
    results.mkdir(parents=True)
    manifest = {
        "schema_version": "1.1",
        "bundle_id": root.name,
        "platform_id": "h100",
        "wave": "qwen7b-update-controls-4gpu",
        "repository": {"git_commit": "a" * 40, "snapshot_directory": "repo"},
        "configs": {
            eid: {"config_sha256": config_sha256(config), "required_trials": 3} for eid, config in configs.items()
        },
    }
    manifest_path = results / "SUBMISSION_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    paths = {}
    for eid, repetition, status in trials:
        config = configs[eid]
        record = ExperimentResult.from_config(
            config,
            status=status,
            started_at="2026-09-26T19:45:00+00:00",
            completed_at="2026-09-26T19:46:00+00:00",
            runtime_seconds=60,
            repetition_number=repetition,
            error="Interrupted by reservation end" if status == "failed" else None,
            metrics={"measured_seconds": 50, "gpu_hours": 50 * 4 / 3600, "total_tokens": 1000},
            environment={
                "git_commit": "a" * 40,
                "git_dirty": False,
                "config_sha256": config_sha256(config),
                "command": "python -m hpc_ai_perf.cli run",
                "observed_gpu_model": "NVIDIA H100",
                "submission_bundle_id": root.name,
                "submission_bundle_manifest_sha256": manifest_hash,
                "submission_bundle_platform": "h100",
                "submission_bundle_wave": "qwen7b-update-controls-4gpu",
            },
        )
        paths[eid, repetition] = record.write_json(results, preserve_trial=True)
    return manifest_hash, paths


@pytest.fixture
def partial_wave(tmp_path, monkeypatch):
    repository = Path.cwd()
    entries = {e.config.experiment_id: e for e in discover_experiment_configs("configs")}
    conditions = load_platform_conditions("platforms/execution_matrix.json", "h100", wave="qwen7b-update-controls-4gpu")
    profile = load_platform_profile("platforms/h100.json")
    configs = {c.matrix_id: resolve_platform_config(entries[c.matrix_id], profile, c) for c in conditions}
    bundle = tmp_path / "first-bundle"
    manifest_hash, paths = write_bundle(
        bundle, configs, [("E268", 1, "completed"), ("E268", 2, "failed"), ("E269", 1, "completed")]
    )
    invalid = json.loads(paths["E269", 1].read_text())
    invalid["environment"]["submission_bundle_manifest_sha256"] = "f" * 64
    paths["E269", 1].write_text(json.dumps(invalid))
    state_path = tmp_path / "wave.json"
    append_wave_attempt(
        state_path,
        platform_id="h100",
        wave="qwen7b-update-controls-4gpu",
        bundle_dir=bundle,
        bundle_id=bundle.name,
        manifest_sha256=manifest_hash,
        status_path=bundle / "SLURM_STATUS.csv",
        experiment_ids=sorted(configs),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_platform_conditions", lambda *args, **kwargs: conditions)
    monkeypatch.setattr(cli, "track_platform_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli,
        "build_platform_progress_rows",
        lambda cs, result_dir: build_platform_progress_rows(cs, result_dir, config_root=repository / "configs"),
    )
    args = Namespace(
        platform_profile=str(repository / "platforms/h100.json"),
        wave="qwen7b-update-controls-4gpu",
        state_file=str(state_path),
        result_root="results/raw",
        allow_incomplete=False,
    )
    return args, state_path, bundle, configs, paths


def test_default_collection_still_requires_complete_wave(partial_wave):
    args, _, _, _, _ = partial_wave
    with pytest.raises(ValueError, match="Wave evidence is incomplete"):
        cli._collect_platform_wave(args)
    assert not Path(args.result_root).exists()


def test_partial_collection_preserves_valid_trial_and_is_idempotent(partial_wave):
    args, state_path, bundle, _, paths = partial_wave
    args.allow_incomplete = True
    original = {path: path.read_bytes() for path in paths.values()}
    imported = cli._collect_platform_wave(args)
    assert len(imported) == 1
    assert Path(imported[0]).read_bytes() == paths["E268", 1].read_bytes()
    assert not (Path(args.result_root) / "h100/E269").exists()
    assert verify_result_evidence(args.result_root) == []
    state = load_wave_state(state_path)
    assert state["collection_status"] == "partial"
    assert state["incomplete_experiment_ids"] == ["E268", "E269"]
    assert state["collected_at"] is None
    assert state["last_collection_at"]
    assert cli._collect_platform_wave(args) == imported
    assert load_wave_state(state_path)["imported_paths"] == imported
    assert {path: path.read_bytes() for path in paths.values()} == original
    assert summarize_bundle_results(bundle).experiments[0].failed_trials == (2,)


def test_partial_collection_combines_recovery_trials_without_marking_missing_as_complete(partial_wave, tmp_path):
    args, state_path, _, configs, _ = partial_wave
    args.allow_incomplete = True
    cli._collect_platform_wave(args)
    recovery = tmp_path / "recovery-bundle"
    manifest_hash, _ = write_bundle(
        recovery,
        configs,
        [("E268", 2, "completed"), ("E268", 3, "completed")]
        + [("E269", repetition, "completed") for repetition in (1, 2, 3)],
    )
    append_wave_attempt(
        state_path,
        platform_id="h100",
        wave=args.wave,
        bundle_dir=recovery,
        bundle_id=recovery.name,
        manifest_sha256=manifest_hash,
        status_path=recovery / "SLURM_STATUS.csv",
        experiment_ids=sorted(configs),
    )
    imported = cli._collect_platform_wave(args)
    assert len(imported) == 6
    assert verify_result_evidence(args.result_root) == []
    state = load_wave_state(state_path)
    assert state["collection_status"] == "complete"
    assert state["incomplete_experiment_ids"] == []
    assert state["collected_at"]


def test_partial_collection_rejects_ambiguous_duplicate_trial(partial_wave):
    args, state_path, _, _, paths = partial_wave
    args.allow_incomplete = True
    paths["E268", 1].with_name("duplicate.json").write_bytes(paths["E268", 1].read_bytes())
    assert cli._collect_platform_wave(args) == []
    assert not (Path(args.result_root) / "h100").exists()
    assert load_wave_state(state_path)["collection_status"] == "partial"


def test_make_partial_collection_flag_reaches_cli_without_submission():
    result = subprocess.run(
        ["make", "-n", "collect-platform-wave", "PLATFORM=h100", "PLATFORM_ALLOW_INCOMPLETE=1"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--allow-incomplete" in result.stdout
    assert "sbatch" not in result.stdout
