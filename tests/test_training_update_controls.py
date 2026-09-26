from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hpc_ai_perf.config import load_experiment_config
from hpc_ai_perf.workloads import transformer as workload


def test_explicit_accumulation_boundaries_preserve_gradients_and_update_counts(monkeypatch):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    boundaries, scales, gradients = [], [], []

    class Engine:
        global_steps = 0
        global_samples = 0

        def parameters(self):
            return model.parameters()

        def train(self):
            model.train()

        def zero_grad(self):
            optimizer.zero_grad(set_to_none=True)

        def set_gradient_accumulation_boundary(self, boundary):
            boundaries.append(boundary)

        def __call__(self, **batch):
            return SimpleNamespace(loss=model(batch["x"]).square().mean())

        def backward(self, loss, *, scale_wrt_gas):
            scales.append(scale_wrt_gas)
            loss.backward()
            gradients.append(float(model.weight.grad))

        def step(self):
            optimizer.step()
            self.global_steps += 1
            self.global_samples += 3

    engine = Engine()
    config = load_experiment_config("configs/large_model_training/E253_qwen25_7b_zero3_2gpu.json")
    config = replace(
        config,
        runtime=replace(config.runtime, precision="fp32", max_steps=2),
        profiling=replace(config.profiling, profilers=[]),
        workload=replace(
            config.workload,
            parameters={
                "batch_size": 1,
                "sequence_length": 1,
                "gradient_accumulation_steps": 3,
                "warmup_steps": 1,
                "training_update_diagnostics": True,
                "deepspeed_accumulation_mode": "explicit_boundary_v1",
            },
        ),
    )
    monkeypatch.setattr(workload, "_configure_training_strategy", lambda *args: (engine, optimizer, engine, False, {}))
    monkeypatch.setattr(workload, "_training_precision", lambda *args, **kwargs: (nullcontext, None))
    monkeypatch.setattr(workload, "_training_data_iterator", lambda *args, **kwargs: None)
    monkeypatch.setattr(workload, "_model_config", lambda model: SimpleNamespace(vocab_size=2))
    monkeypatch.setattr(workload, "_make_device_batch", lambda *args: {"x": torch.ones(1, 1)})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = workload._run_training_loop(
        torch,
        config,
        workload.DistributedRuntime(0, 0, 1, False),
        engine,
        task="causal_lm",
        model_name="test",
    ).metrics
    assert boundaries == [False, False, True] * 3
    assert scales == [False] * 9  # no second loss division in DeepSpeed
    assert gradients[1] == pytest.approx(gradients[0] * 2)
    assert gradients[2] == pytest.approx(gradients[0] * 3)
    assert result["optimizer_step_calls"] == result["deepspeed_optimizer_updates"] == 3
    assert result["backward_calls"] == result["expected_backward_calls"] == 9
    assert result["deepspeed_samples_accounted"] == 9
    assert result["initial_training_loss"] >= 0
    assert result["optimizer_state_dtypes_observed"] == "torch.float32"


def test_optimizer_state_dtype_probe_reads_wrapped_states_and_handles_cycles():
    inner = SimpleNamespace(state={0: {"exp_avg": SimpleNamespace(dtype="torch.float32"), "step": 1}})
    outer = SimpleNamespace(state={}, optimizer=inner)
    inner.optimizer = outer
    assert workload._optimizer_state_dtypes(outer) == "torch.float32"


@pytest.mark.parametrize("minimum", [float("nan"), -1])
def test_classification_energy_duration_rejects_invalid_minimum(monkeypatch, minimum):
    torch = pytest.importorskip("torch")
    config = load_experiment_config("configs/hardware_energy_revalidation/E238_l40s_distilbert_inference_energy.json")
    config = replace(
        config,
        workload=replace(config.workload, parameters={**config.workload.parameters, "min_measured_seconds": minimum}),
    )
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=10), eval=lambda: None)
    model.to = lambda device: model
    monkeypatch.setattr(
        workload,
        "_load_sequence_classification_stack",
        lambda: (torch, SimpleNamespace(from_pretrained=lambda *args, **kwargs: model)),
    )
    monkeypatch.setattr(
        workload, "_prepare_runtime", lambda *args, **kwargs: workload.DistributedRuntime(0, 0, 1, False)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="min_measured_seconds"):
        workload._run_classification_inference(config)


def test_classification_energy_anchor_counts_actual_batches_and_fails_closed(monkeypatch):
    torch = pytest.importorskip("torch")
    config = load_experiment_config("configs/hardware_energy_revalidation/E238_l40s_distilbert_inference_energy.json")
    config = replace(
        config,
        runtime=replace(config.runtime, max_steps=100),
        workload=replace(
            config.workload, parameters={**config.workload.parameters, "warmup_steps": 1, "min_measured_seconds": 0.025}
        ),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(vocab_size=10),
        eval=lambda: None,
        parameters=lambda: iter([SimpleNamespace(dtype="torch.float32")]),
    )
    model.to = lambda device: model
    clock = [0.0]

    def round_call(*args):
        clock[0] += 0.01
        return [torch.ones(1, 2)], [0.01]

    monkeypatch.setattr(
        workload,
        "_load_sequence_classification_stack",
        lambda: (torch, SimpleNamespace(from_pretrained=lambda *args, **kwargs: model)),
    )
    monkeypatch.setattr(
        workload, "_prepare_runtime", lambda *args, **kwargs: workload.DistributedRuntime(0, 0, 1, False)
    )
    monkeypatch.setattr(workload, "_classification_round", round_call)
    monkeypatch.setattr(workload.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    metrics = workload._run_classification_inference(config).metrics
    assert metrics["batches"] == metrics["requests"] == 3
    assert metrics["total_tokens"] == 3 * 128
    config = replace(config, runtime=replace(config.runtime, max_steps=2))
    with pytest.raises(RuntimeError, match="Measured interval too short"):
        workload._run_classification_inference(config)
