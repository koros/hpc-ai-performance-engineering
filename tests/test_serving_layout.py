from dataclasses import replace
from pathlib import Path

import pytest

from hpc_ai_perf.config import load_experiment_config
from hpc_ai_perf.results.contract import UnsupportedWorkloadError
from hpc_ai_perf.serving_layout import declared_serving_layout, serving_layout_issues
from hpc_ai_perf.validation import capability_issues
from hpc_ai_perf.workloads.transformer import _run_vllm_inference


def test_explicit_tensor_parallel_layout_is_validated_without_changing_legacy_config() -> None:
    config = load_experiment_config(Path("configs/inference/E702_tinyllama_tp2_long_prompt.json"))
    parameters = dict(config.workload.parameters)
    parameters["serving_layout"] = "tp2"
    explicit = replace(config, workload=replace(config.workload, parameters=parameters))

    assert declared_serving_layout(parameters) == "tp2"
    assert serving_layout_issues(explicit) == []
    assert capability_issues(explicit) == []


def test_tp2_dp2_is_structurally_checked_and_marked_deferred() -> None:
    config = load_experiment_config(Path("configs/inference/E702_tinyllama_tp2_long_prompt.json"))
    parameters = dict(config.workload.parameters)
    parameters.update(
        {
            "serving_layout": "tp2_dp2",
            "data_parallel_size": 2,
            "replica_count": 2,
            "request_routing": "round_robin",
        }
    )
    four_gpu = replace(
        config,
        hardware=replace(config.hardware, gpus=4),
        distributed=replace(config.distributed, gpus_per_node=4),
        workload=replace(config.workload, parameters=parameters),
    )

    issues = capability_issues(four_gpu)

    assert any("multi-process replica router is deferred" in issue for issue in issues)
    assert not any("requires hardware.gpus=4" in issue for issue in issues)


def test_tp2_dp2_runner_fails_loudly_before_engine_startup() -> None:
    config = load_experiment_config(Path("configs/inference/E702_tinyllama_tp2_long_prompt.json"))
    parameters = dict(config.workload.parameters)
    parameters.update(
        {
            "serving_layout": "tp2_dp2",
            "data_parallel_size": 2,
            "replica_count": 2,
            "request_routing": "round_robin",
        }
    )
    four_gpu = replace(
        config,
        hardware=replace(config.hardware, gpus=4),
        distributed=replace(config.distributed, gpus_per_node=4),
        workload=replace(config.workload, parameters=parameters),
    )

    with pytest.raises(UnsupportedWorkloadError, match="multi-process replica router"):
        _run_vllm_inference(four_gpu)
