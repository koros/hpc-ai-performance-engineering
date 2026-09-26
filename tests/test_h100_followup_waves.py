import json
from pathlib import Path

import pytest
from scripts.verify_h100_energy_anchors import verify_results

from hpc_ai_perf.cli import _build_parser, _render_platform_slurm, _submit_platform_wave
from hpc_ai_perf.config import discover_experiment_configs, load_experiment_config
from hpc_ai_perf.config.platform import (
    load_platform_conditions,
    load_platform_profile,
    materialize_platform_configs,
    resolve_platform_config,
)
from hpc_ai_perf.validation.capabilities import capability_issues

WAVES = {
    "qwen7b-update-canary": ({"E261", "E262", "E263"}, 2, 1),
    "qwen7b-update-controls-2gpu": ({"E264", "E265", "E266"}, 2, 3),
    "qwen7b-update-controls-4gpu": ({"E267", "E268", "E269"}, 4, 3),
    "h100-capacity-allocator-validation": ({"E260"}, 1, 3),
    "lowbit-quality": ({"E270", "E271", "E272", "E273"}, 1, 3),
    "h100-inference-energy-anchors": ({"E274", "E275", "E276"}, 1, 3),
}


@pytest.mark.parametrize("wave", WAVES)
def test_followup_waves_materialize_and_render_expected_h100_allocations(tmp_path, wave):
    ids, gpus, trials = WAVES[wave]
    conditions = load_platform_conditions("platforms/execution_matrix.json", "h100", wave=wave)
    assert {c.matrix_id for c in conditions} == ids
    assert {c.required_trials for c in conditions} == {trials}
    paths = materialize_platform_configs(load_platform_profile("platforms/h100.json"), conditions, tmp_path / "configs")
    for path in paths:
        config = load_experiment_config(path)
        assert config.hardware.gpus == gpus
        assert capability_issues(config) == []
    _render_platform_slurm(
        platform_profile="platforms/h100.json",
        execution_matrix="platforms/execution_matrix.json",
        config_root="configs",
        config_output_dir=str(tmp_path / "configs"),
        output_dir=str(tmp_path / "scripts"),
        run_output_dir=str(tmp_path / "results"),
        log_dir=str(tmp_path / "logs"),
        wave=wave,
        experiment_ids=None,
        profile_path=None,
        partition=None,
        account=None,
        time_limit=None,
        cpus_per_task=None,
        modules=[],
        env_values=[],
    )
    for eid in ids:
        script = (tmp_path / "scripts" / f"{eid}.sbatch").read_text()
        assert f"#SBATCH --gres=gpu:{gpus}" in script
        assert "#SBATCH --partition=gpu_h100" in script
        assert f"--nproc-per-node={gpus}" in script if gpus > 1 else "--nproc-per-node" not in script
        if eid == "E260":
            assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in script


def test_energy_anchors_have_duration_controls_and_preserve_originals():
    configs = [load_experiment_config(p) for p in Path("configs/hardware_energy_revalidation").glob("E27*.json")]
    assert len(configs) == 3
    for config in configs:
        assert config.workload.parameters["min_measured_seconds"] == 30
        assert config.runtime.max_steps == 100000
        assert config.profiling.parameters["sample_interval_seconds"] == 0.1
    legacy = load_experiment_config("configs/hardware_energy_revalidation/E238_l40s_distilbert_inference_energy.json")
    assert "min_measured_seconds" not in legacy.workload.parameters


def test_energy_verifier_requires_current_contract_and_interior_samples(tmp_path):
    entries = {e.config.experiment_id: e for e in discover_experiment_configs("configs/hardware_energy_revalidation")}
    for condition in load_platform_conditions(
        "platforms/execution_matrix.json", "h100", wave="h100-inference-energy-anchors"
    ):
        cfg = resolve_platform_config(
            entries[condition.matrix_id], load_platform_profile("platforms/h100.json"), condition
        )
        root = tmp_path / condition.matrix_id
        root.mkdir()
        for repetition in (1, 2, 3):
            root.joinpath(f"{repetition}.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "error": None,
                        "repetition_number": repetition,
                        "config": cfg.to_dict(),
                        "environment": {"observed_gpu_model": "NVIDIA H100"},
                        "metrics": {
                            "measured_seconds": 30,
                            "energy_scope": "measured_region",
                            "energy_measurement_status": "sufficient",
                            "energy_joules_measured_region": 1000,
                            "nvidia_smi.gpu_mapping_method": "cuda_uuid",
                            "nvidia_smi.measurement_interior_samples": 200,
                            "nvidia_smi.measurement_max_gap_seconds": 0.2,
                        },
                    }
                )
            )
    assert set(verify_results(tmp_path)) == {"E274", "E275", "E276"}
    path = tmp_path / "E274" / "1.json"
    record = json.loads(path.read_text())
    record["metrics"]["nvidia_smi.measurement_interior_samples"] = 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="duration and energy coverage"):
        verify_results(tmp_path)


@pytest.mark.parametrize("wave", ["qwen7b-update-controls-2gpu", "qwen7b-update-controls-4gpu", "lowbit-quality"])
def test_generic_submission_cannot_bypass_new_gates(tmp_path, monkeypatch, wave):
    monkeypatch.setenv("HPC_AI_QWEN_UPDATE_RESULT_ROOT", str(tmp_path))
    monkeypatch.delenv("HPC_AI_LOWBIT_ACCEPTANCE_DIR", raising=False)
    args = _build_parser().parse_args(
        ["submit-platform-wave", "--platform-profile", "platforms/h100.json", "--wave", wave]
    )
    with pytest.raises(ValueError):
        _submit_platform_wave(args)
