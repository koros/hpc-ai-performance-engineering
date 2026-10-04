"""The small crossover wave must preserve its batch and GPU controls."""

from hpc_ai_perf.config import discover_experiment_configs
from hpc_ai_perf.config.platform import load_platform_conditions, load_platform_profile, resolve_platform_config
from hpc_ai_perf.slurm.job import SlurmJob
from hpc_ai_perf.validation.capabilities import capability_issues


def test_crossover_grid_and_canaries_render_for_both_platforms():
    entries = {entry.config.experiment_id: entry for entry in discover_experiment_configs("configs/scaling_studies")}
    for platform in ("l40s", "h100"):
        profile = load_platform_profile(f"platforms/{platform}.json")
        for wave, expected, trials in (
            ("scaling-batch-canary", {f"E{n:03d}" for n in range(314, 318)}, 1),
            ("scaling-batch-crossover", {f"E{n:03d}" for n in range(305, 314)}, 3),
        ):
            conditions = load_platform_conditions("platforms/execution_matrix.json", platform, wave=wave)
            assert {condition.matrix_id for condition in conditions} == expected
            for condition in conditions:
                config = resolve_platform_config(entries[condition.matrix_id], profile, condition)
                p = config.workload.parameters
                assert condition.required_trials == trials
                assert config.hardware.gpus in {1, 2, 4}
                assert p["global_batch_size"] == config.hardware.gpus * p["per_gpu_batch_size"]
                assert config.runtime.precision == "bf16" and p["model_load_dtype"] == "fp32"
                assert config.workload.dataset == "synthetic_global_batches_v1"
                assert not capability_issues(config)
                script = SlurmJob.from_config(
                    config,
                    "/bundle/config.json",
                    partition=profile.partition,
                    repetitions=trials,
                ).render()
                assert f"#SBATCH --gres=gpu:{config.hardware.gpus}" in script
                if trials == 3:
                    assert "#SBATCH --array=1-3%1" in script

    # Nine strong-scaling points: global batch is fixed within each 1/2/4-GPU trio.
    assert {entries[f"E{n:03d}"].config.workload.parameters["global_batch_size"] for n in range(305, 308)} == {4}
    assert {entries[f"E{n:03d}"].config.workload.parameters["global_batch_size"] for n in range(308, 311)} == {8}
    assert {entries[f"E{n:03d}"].config.workload.parameters["global_batch_size"] for n in range(311, 314)} == {16}


def test_l40s_placement_diagnostics_remain_separate_from_performance_grid():
    entries = {entry.config.experiment_id: entry for entry in discover_experiment_configs("configs/scaling_studies")}
    profile = load_platform_profile("platforms/l40s.json")
    for wave, eid, repetitions in (
        ("scaling-placement-diagnostic", "E318", 3),
        ("scaling-trace-diagnostic", "E319", 1),
    ):
        conditions = load_platform_conditions("platforms/execution_matrix.json", "l40s", wave=wave)
        assert len(conditions) == 1 and conditions[0].matrix_id == eid
        config = resolve_platform_config(entries[eid], profile, conditions[0])
        assert config.hardware.gpus == 2
        assert config.workload.parameters["capture_gpu_topology"] is True
        assert "scaling_study_id" not in config.workload.parameters
        assert conditions[0].required_trials == repetitions
        assert not capability_issues(config)
