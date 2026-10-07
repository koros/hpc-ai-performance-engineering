from pathlib import Path

from hpc_ai_perf.config import load_experiment_config
from hpc_ai_perf.planning import plan_sweep


def test_plan_sweep_is_deterministic_and_does_not_replace_source_ids() -> None:
    base = load_experiment_config(Path("configs/inference/E702_tinyllama_tp2_long_prompt.json"))
    axes = {
        "hardware.gpus": [1, 2, 4],
        "distributed.gpus_per_node": [1, 2, 4],
        "workload.parameters.tensor_parallel_size": [1, 2, 4],
    }

    first = plan_sweep(base, plan_id="tinyllama-tp", axes=axes)
    second = plan_sweep(base, plan_id="tinyllama-tp", axes=axes)

    assert len(first) == 27
    assert [candidate.experiment_id for candidate in first] == [candidate.experiment_id for candidate in second]
    assert base.experiment_id == "E702"
    assert all(candidate.config.experiment_id != "E702" for candidate in first)
    assert all(candidate.config.metadata["sweep_source_experiment_id"] == "E702" for candidate in first)


def test_plan_sweep_marks_invalid_context_shapes_without_dropping_them() -> None:
    base = load_experiment_config(Path("configs/inference/E702_tinyllama_tp2_long_prompt.json"))
    candidates = plan_sweep(
        base,
        plan_id="tinyllama-context",
        axes={"workload.parameters.prompt_length": [1920, 4096]},
    )

    assert [candidate.valid for candidate in candidates] == [True, False]
    assert any("context window" in issue for issue in candidates[1].issues)
