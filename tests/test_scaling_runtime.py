from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hpc_ai_perf.config import load_experiment_config
from hpc_ai_perf.config.scaling import INPUT_PROTOCOL, scaling_issues
from hpc_ai_perf.workloads import transformer as workload
from hpc_ai_perf.workloads.scaling import GlobalTrainingInputs


def config(gpus=1, accumulation=1):
    original = load_experiment_config("configs/scaling_studies/E044_tinyllama_strong_1gpu.json")
    return replace(
        original,
        hardware=replace(original.hardware, gpus=gpus),
        distributed=replace(original.distributed, gpus_per_node=gpus),
        runtime=replace(original.runtime, max_steps=2),
        profiling=replace(original.profiling, profilers=[]),
        workload=replace(
            original.workload,
            dataset=INPUT_PROTOCOL,
            parameters={
                "scaling_study_id": "test",
                "scaling_role": "performance",
                "scaling_type": "strong",
                "fixed_workload": True,
                "stack": "ddp",
                "input_protocol": INPUT_PROTOCOL,
                "per_gpu_batch_size": 8 // gpus // accumulation,
                "global_batch_size": 8,
                "gradient_accumulation_steps": accumulation,
                "sequence_length": 4,
                "warmup_steps": 1,
                "model_load_dtype": "fp32",
                "attention_backend": "sdpa",
                "training_update_diagnostics": True,
            },
        ),
    )


@pytest.mark.parametrize("gpus,accumulation", [(1, 1), (2, 1), (4, 1), (2, 2), (1, 4)])
def test_global_inputs_preserve_examples_across_gpu_and_accumulation_shapes(gpus, accumulation):
    torch = pytest.importorskip("torch")
    reference = GlobalTrainingInputs(torch, config(), SimpleNamespace(rank=0, world_size=1), "cpu", 128)
    shards = [
        GlobalTrainingInputs(torch, config(gpus, accumulation), SimpleNamespace(rank=r, world_size=gpus), "cpu", 128)
        for r in range(gpus)
    ]
    previous = None
    for _ in range(3):
        global_batch = reference.next_batch()["input_ids"]
        joined = torch.cat([torch.cat([s.next_batch()["input_ids"] for _ in range(accumulation)]) for s in shards])
        assert torch.equal(joined, global_batch)
        assert previous is None or not torch.equal(previous, global_batch)
        previous = global_batch
    assert {s.sha256 for s in shards} == {reference.sha256}
    with pytest.raises(RuntimeError, match="exhausted"):
        reference.next_batch()


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("fixed_workload", False, "fixed_workload"),
        ("stack", "fsdp", "stack"),
        ("global_batch_size", 7, "global batch"),
        ("num_workers", 4, "preloaded"),
        ("gradient_accumulation_steps", 0, "positive integer"),
        ("input_protocol", "unknown", "input_protocol"),
        ("attention_backend", "eager", "attention"),
    ],
)
def test_controlled_scaling_rejects_inconsistent_contracts(field, value, message):
    cfg = config()
    assert scaling_issues(cfg) == []
    cfg = replace(cfg, workload=replace(cfg.workload, parameters={**cfg.workload.parameters, field: value}))
    assert any(message in issue for issue in scaling_issues(cfg))


def test_controlled_training_consumes_new_inputs_and_performs_declared_updates(monkeypatch):
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 4)
            self.config = SimpleNamespace(vocab_size=32, use_cache=True, _attn_implementation="sdpa")

        def forward(self, input_ids, labels):
            return SimpleNamespace(loss=self.embedding(input_ids).square().mean())

    model = Model()
    original = model.embedding.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(workload, "_configure_training_strategy", lambda *a: (model, optimizer, None, False, {}))
    monkeypatch.setattr(workload, "_training_precision", lambda *a, **k: (nullcontext, None))
    metrics = workload._run_training_loop(
        torch,
        config(accumulation=2),
        workload.DistributedRuntime(0, 0, 1, False),
        model,
        task="causal_lm",
        model_name="test",
    ).metrics
    assert metrics["optimizer_step_calls"] == 3
    assert metrics["backward_calls"] == metrics["input_micro_batches_consumed"] == 6
    assert metrics["total_samples"] == 16  # warm-up excluded
    assert metrics["total_tokens"] == 64
    assert metrics["dataset_execution"] == INPUT_PROTOCOL
    assert len(metrics["global_input_sha256"]) == len(metrics["scaling_runtime_sha256"]) == 64
    assert metrics["training_use_cache"] is False
    assert metrics["data_loading_seconds"] == metrics["cpu_to_gpu_transfer_seconds"] == 0
    assert not torch.equal(original, model.embedding.weight)


def test_legacy_e044_retains_its_original_input_and_precision_contract():
    cfg = load_experiment_config("configs/scaling_studies/E044_tinyllama_strong_1gpu.json")
    assert cfg.workload.dataset == "synthetic_tokens_v1"
    assert "input_protocol" not in cfg.workload.parameters
    assert scaling_issues(cfg) == []
