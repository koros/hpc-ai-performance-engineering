from hpc_ai_perf.config import load_experiment_config
from hpc_ai_perf.config.platform import load_platform_conditions


def test_allocator_revalidation_preserves_canary_shape_and_separates_repetitions():
    original = load_experiment_config(
        "configs/platform_adjusted/E259_h100_gpt2_small_micro_batch_48_expandable_segments.json"
    )
    new = load_experiment_config("configs/platform_adjusted/E260_h100_gpt2_micro_batch_48_allocator_revalidation.json")
    assert new.workload == original.workload
    assert new.runtime == original.runtime
    (condition,) = load_platform_conditions(
        "platforms/execution_matrix.json", "h100", wave="h100-capacity-allocator-validation"
    )
    assert condition.required_trials == 3
    assert condition.comparability == "capacity_adjusted"
    (canary,) = load_platform_conditions(
        "platforms/execution_matrix.json", "h100", wave="h100-capacity-allocator-canary"
    )
    assert canary.required_trials == 1
    assert not load_platform_conditions("platforms/execution_matrix.json", "l40s", wave=condition.wave)
