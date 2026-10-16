from pathlib import Path

from hpc_ai_perf.config import load_experiment_config
from hpc_ai_perf.planning import plan_serving_layout
from hpc_ai_perf.slurm.job import SlurmJob


def test_plan_serving_layout_resolves_tp2dp2_without_starting_vllm() -> None:
    config = load_experiment_config(Path("configs/inference/E711_tinyllama_tp2dp2_concurrency4.json"))

    plan = plan_serving_layout(config, visible_devices="4,6,7,9")

    assert plan["valid"] is True
    assert plan["serving_layout"] == "tp2_dp2"
    assert plan["request_routing"] == "round_robin"
    assert plan["gpu_groups"] == [["4", "6"], ["7", "9"]]
    assert plan["concurrent_requests"] == 4


def test_plan_serving_layout_reports_shape_errors_without_launching() -> None:
    config = load_experiment_config(Path("configs/inference/E711_tinyllama_tp2dp2_concurrency4.json"))

    plan = plan_serving_layout(config, visible_devices="0,1")

    assert plan["valid"] is False
    assert any("exposes 2 devices, expected 4" in issue for issue in plan["capability_issues"])


def test_serving_comparison_trials_cannot_compete_with_each_other() -> None:
    for experiment_number in range(702, 715):
        path = next(Path("configs/inference").glob(f"E{experiment_number}_*.json"))
        config = load_experiment_config(path)
        script = SlurmJob.from_config(config, path, repetitions=3).render()
        assert "#SBATCH --array=1-3%1" in script
        assert "# HPC_AI_SERIAL_WAVE=1" in script
