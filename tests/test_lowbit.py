from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from hpc_ai_perf.config import ExperimentConfig
from hpc_ai_perf.config.lowbit import (
    ARTIFACTS,
    DATASET,
    DATASET_REVISION,
    PROTOCOL,
    lowbit_issues,
    runtime_fingerprint,
)
from hpc_ai_perf.validation.capabilities import capability_issues
from hpc_ai_perf.workloads.lowbit import (
    engine_kwargs,
    quality_metrics,
    request_measurements,
    select_prompts,
    text_windows,
    validate_artifact,
    validate_backend,
)


def example_config(method="bf16", batch=1):
    model, revision, _ = ARTIFACTS[method]
    return ExperimentConfig.from_mapping(
        {
            "experiment_id": "test_lowbit",
            "experiment": "test_lowbit",
            "mode": "inference",
            "workload": {
                "name": "qwen2_5_7b",
                "model": model,
                "model_revision": revision,
                "dataset": DATASET,
                "parameters": {
                    "study_protocol": PROTOCOL,
                    "quantization_method": method,
                    "framework": "vllm",
                    "base_compute_dtype": "bf16",
                    "batch_size": batch,
                    "prompt_length": 512,
                    "max_new_tokens": 128,
                    "warmup_steps": 3,
                    "min_measured_seconds": 30,
                    "prompt_samples": 64,
                    "quality_samples": 32,
                    "quality_sequence_length": 256,
                    "quality_max_relative_ppl_increase": 0.1,
                    "dataset_revision": DATASET_REVISION,
                    "dataset_subset": "wikitext-2-raw-v1",
                    "performance_split": "validation",
                    "quality_split": "test",
                    "max_model_len": 1024,
                    "kv_cache_blocks": 1024,
                    "kv_cache_block_size": 16,
                    "gpu_memory_utilization": 0.8,
                    "comparison_control_id": "E239" if batch == 1 else "E240",
                },
            },
            "hardware": {"platform": "lovelace_h100", "gpus": 1, "gpu_model": "NVIDIA H100"},
            "distributed": {"strategy": "none", "nodes": 1, "gpus_per_node": 1},
            "runtime": {"precision": "bf16" if method == "bf16" else "int4", "max_steps": 300},
        }
    )


@pytest.mark.parametrize("method", list(ARTIFACTS))
def test_contract_and_engine_pin_artifact_tokenizer_compute_and_cache(method):
    config = example_config(method)
    assert capability_issues(config) == []
    kwargs = engine_kwargs(config)
    assert kwargs["model"] == ARTIFACTS[method][0]
    assert kwargs["revision"] == ARTIFACTS[method][1]
    assert kwargs["tokenizer"] == ARTIFACTS["bf16"][0]
    assert kwargs["dtype"] == "bfloat16"
    assert kwargs["quantization"] == ARTIFACTS[method][2]
    assert kwargs["num_gpu_blocks_override"] == 1024
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs.get("load_format") == ("bitsandbytes" if method == "bitsandbytes_nf4" else None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("study_protocol", "legacy"),
        ("batch_size", 0),
        ("comparison_control_id", "E178"),
        ("quality_split", "validation"),
        ("dataset_revision", "main"),
        ("kv_cache_blocks", 1),
        ("base_compute_dtype", "fp16"),
    ],
)
def test_contract_rejects_uncontrolled_comparisons(field, value):
    config = example_config()
    params = {**config.workload.parameters, field: value}
    assert lowbit_issues(replace(config, workload=replace(config.workload, parameters=params)))


def test_contract_rejects_moving_model_revision():
    config = example_config()
    assert lowbit_issues(replace(config, workload=replace(config.workload, model_revision="main")))
    assert len(runtime_fingerprint()) == 64


def test_artifact_and_observed_module_checks_reject_silent_fallback():
    valid = {
        "model_type": "qwen2",
        "num_hidden_layers": 28,
        "quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128},
    }
    validate_artifact("awq", valid)
    with pytest.raises(RuntimeError, match="quantization metadata"):
        validate_artifact("gptq", valid)
    with pytest.raises(RuntimeError, match="unquantized base"):
        validate_artifact("bf16", valid)
    with pytest.raises(RuntimeError, match="not observed"):
        validate_backend("awq", {"quantized_module_count": 0, "quantized_module_types": []})
    with pytest.raises(RuntimeError, match="NF4"):
        validate_backend(
            "bitsandbytes_nf4",
            {
                "quantized_module_count": 1,
                "quantized_module_types": ["BitsAndBytesLinearMethod"],
                "bnb_quant_types": ["fp4"],
            },
        )


def test_real_text_windows_are_deterministic_and_do_not_cross_documents():
    tokenizer = SimpleNamespace(encode=lambda text, **kw: list(map(int, text.split())))
    assert text_windows(["", "1 2 3 4 5", "6 7 8 9"], tokenizer, 2, 4) == [[1, 2], [3, 4], [6, 7], [8, 9]]
    with pytest.raises(RuntimeError, match="fewer"):
        text_windows(["1 2"], tokenizer, 2, 2)
    assert select_prompts([[1], [2], [3]], 1, 2) == [{"prompt_token_ids": [3]}, {"prompt_token_ids": [1]}]


def test_quality_uses_target_token_logprobs_not_generated_token_probability():
    output = SimpleNamespace(
        prompt_logprobs=[None, {2: SimpleNamespace(logprob=-1.0)}, {3: SimpleNamespace(logprob=-3.0)}]
    )
    llm = SimpleNamespace(generate=lambda *args, **kw: [output])
    result = quality_metrics(llm, SimpleNamespace, [[1, 2, 3]])
    assert result["quality_tokens"] == 2
    assert result["quality_loss"] == 2
    assert result["quality_perplexity"] == pytest.approx(7.389056)
    output.prompt_logprobs[2] = {3: SimpleNamespace(logprob=float("nan"))}
    with pytest.raises(RuntimeError, match="log probability"):
        quality_metrics(llm, SimpleNamespace, [[1, 2, 3]])


def test_missing_latency_or_short_generation_is_not_relabelled_successful():
    output = SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=[1, 2])],
        metrics=SimpleNamespace(arrival_time=1, first_token_time=2, finished_time=3),
    )
    assert request_measurements([output], 1, 2) == (2, [2], [1])
    with pytest.raises(RuntimeError, match="fixed number"):
        request_measurements([output], 1, 3)
    output.metrics.first_token_time = None
    with pytest.raises(RuntimeError, match="no batch-time fallback"):
        request_measurements([output], 1, 2)
