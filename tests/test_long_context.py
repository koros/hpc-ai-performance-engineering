from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hpc_ai_perf.config import load_experiment_config
from hpc_ai_perf.config.long_context import CONTEXT_IDS, runtime_fingerprint
from hpc_ai_perf.validation.capabilities import capability_issues
from hpc_ai_perf.workloads import default_registry
from hpc_ai_perf.workloads.long_context import engine_kwargs


@pytest.mark.parametrize("length,experiment", CONTEXT_IDS.items())
def test_long_context_configs_fix_model_cache_request_shape_and_real_input(length, experiment):
    path = Path(f"configs/long_context_serving/{experiment}_qwen25_7b_vllm_context_{length}.json")
    config = load_experiment_config(path)
    assert config.experiment_id == experiment
    assert config.workload.parameters["prompt_length"] == length
    assert config.workload.dataset == "Salesforce/wikitext"
    assert config.hardware.gpus == 1
    assert capability_issues(config) == []
    kwargs = engine_kwargs(config)
    assert kwargs["max_model_len"] == 9216
    assert kwargs["num_gpu_blocks_override"] == 1024
    assert kwargs["max_num_seqs"] == 1
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["enable_chunked_prefill"] is False


def test_long_context_contract_rejects_shape_and_cache_drift():
    config = load_experiment_config(Path("configs/long_context_serving/E249_qwen25_7b_vllm_context_8192.json"))
    wrong_context = replace(
        config,
        workload=replace(config.workload, parameters={**config.workload.parameters, "max_model_len": 8320}),
    )
    wrong_length = replace(
        config,
        workload=replace(config.workload, parameters={**config.workload.parameters, "prompt_length": 8000}),
    )
    assert capability_issues(wrong_context)
    assert capability_issues(wrong_length)
    assert len(runtime_fingerprint()) == 64


def test_long_context_adapter_is_registered_separately_from_lowbit():
    registry = default_registry()
    assert registry.get("qwen2_5_7b_long_context").name == "qwen2_5_7b_long_context"
    assert registry.get("qwen2_5_7b").name == "qwen2_5_7b"
