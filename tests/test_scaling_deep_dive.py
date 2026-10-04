import importlib.util
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from hpc_ai_perf.analysis.scaling_deep_dive import (
    build_scaling_deep_dive_conditions,
    build_scaling_deep_dive_effects,
    placement_class,
    render_scaling_deep_dive_report,
)
from hpc_ai_perf.analysis.summaries import build_scaling_summary
from hpc_ai_perf.cli import _submit_platform_wave
from hpc_ai_perf.config import load_experiment_config
from hpc_ai_perf.config import scaling_deep_dive as design_module
from hpc_ai_perf.config.platform import load_platform_conditions
from hpc_ai_perf.config.scaling import DEEP_DIVE_STUDY_ID, scaling_issues
from hpc_ai_perf.slurm.job import SlurmJob
from hpc_ai_perf.validation.capabilities import capability_issues
from hpc_ai_perf.validation.scaling_deep_dive import _canary_issues, verify_deep_dive_canaries
from hpc_ai_perf.workloads import transformer as workload
from hpc_ai_perf.workloads.scaling import (
    GlobalTrainingInputs,
    RealTextInputs,
    scaling_runtime_sha256,
    wikitext_documents,
)

ENTRIES = design_module.design()
BY_ID = {experiment_id: (condition, studies) for experiment_id, condition, studies in ENTRIES}


def config_path(experiment_id):
    return next(Path(design_module.CONFIG_DIR).glob(f"{experiment_id}_*.json"))


def test_design_ids_are_contiguous_and_conditions_unique():
    ids = [int(experiment_id[1:]) for experiment_id, _, _ in ENTRIES]
    assert ids == list(range(design_module.FIRST_ID, design_module.FIRST_ID + len(ENTRIES)))
    assert len({condition for _, condition, _ in ENTRIES}) == len(ENTRIES)


def test_every_study_is_populated_and_reuses_shared_controls():
    owners = Counter(studies[0].key for _, _, studies in ENTRIES)
    assert set(owners) == {"C", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "RL"}
    # The GPT-2 8 x 1024 accumulation-1 runs serve S1, S2 (8,192-token split) and S8 (BF16 control).
    shared = [s for _, c, s in ENTRIES if c == design_module.Condition("gpt2_small", 2, 8, 1024)][0]
    assert [study.key for study in shared] == ["S1", "S2", "S8"]


@pytest.mark.parametrize("experiment_id", [e for e, _, _ in ENTRIES])
def test_conditions_respect_memory_rule_model_limits_and_platform_capacity(experiment_id):
    condition, _ = BY_ID[experiment_id]
    estimate = design_module.estimated_memory_gb(condition)
    for platform in condition.run_platforms:
        assert condition.gpus <= design_module.PLATFORM_MAX_GPUS[platform]
        assert estimate <= design_module.MEMORY_RULE * design_module.CAPACITY_GB[platform]
    assert condition.sequence_length <= design_module.WORKLOADS[condition.workload]["max_seq"]
    if condition.workload == "tinyllama":
        assert condition.micro_batch * condition.sequence_length <= 4096 or condition.strategy == "fsdp"


def test_generated_configs_and_waves_match_the_design():
    spec = importlib.util.spec_from_file_location("generator", "scripts/generate_scaling_deep_dive.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    for path, text in generator.expected_configs().items():
        assert path.read_text() == text, path.name
    assert len(list(Path(design_module.CONFIG_DIR).glob("E*.json"))) == len(ENTRIES)
    matrix = Path("platforms/execution_matrix.json").read_text()
    assert generator.expected_matrix(matrix) == matrix


@pytest.mark.parametrize("experiment_id", [e for e, _, _ in ENTRIES])
def test_generated_configs_are_research_ready(experiment_id):
    config = load_experiment_config(config_path(experiment_id))
    assert capability_issues(config) == []
    assert scaling_issues(config) == []
    p = config.workload.parameters
    assert p["scaling_study_id"] == DEEP_DIVE_STUDY_ID
    assert config.metadata["serial_submission"] is True
    assert p["global_batch_size"] == config.hardware.gpus * p["per_gpu_batch_size"] * p["gradient_accumulation_steps"]


def test_fp8_runs_name_their_matched_transformer_engine_bf16_control():
    for experiment_id, condition, _ in ENTRIES:
        if condition.precision != "te_fp8":
            continue
        control_id = load_experiment_config(config_path(experiment_id)).workload.parameters["comparison_control_id"]
        assert BY_ID[control_id][0] == replace(condition, precision="te_bf16")


def test_waves_route_each_condition_once_per_platform_with_declared_trials():
    for platform in ("h100", "l40s"):
        conditions = [
            c
            for c in load_platform_conditions("platforms/execution_matrix.json", platform)
            if c.wave.startswith(design_module.WAVE_PREFIX)
        ]
        ids = [c.matrix_id for c in conditions]
        assert len(ids) == len(set(ids))
        assert set(ids) == {e for e, c, _ in ENTRIES if platform in c.run_platforms}
        for c in conditions:
            condition, studies = BY_ID[c.matrix_id]
            assert c.required_trials == (1 if condition.role == "canary" else 3)
            assert c.wave == design_module.WAVE_PREFIX + studies[0].wave


def test_deep_dive_gate_rejects_missing_collected_evidence(tmp_path):
    with pytest.raises(ValueError, match="SHA256SUMS"):
        verify_deep_dive_canaries(tmp_path, "l40s")


def test_performance_submission_checks_deep_dive_gate_before_creating_bundle(monkeypatch):
    from hpc_ai_perf.validation import scaling_deep_dive

    def reject(*args, **kwargs):
        raise ValueError("canary gate refused")

    monkeypatch.setattr(scaling_deep_dive, "verify_deep_dive_canaries", reject)
    args = SimpleNamespace(
        wave="scaling-dd-accumulation",
        platform_profile="platforms/l40s.json",
        execution_matrix="platforms/execution_matrix.json",
    )
    with pytest.raises(ValueError, match="canary gate refused"):
        _submit_platform_wave(args)


def test_deep_dive_canary_checks_memory_updates_and_backend():
    config = load_experiment_config(config_path("E320"))
    p = config.workload.parameters
    updates = p["warmup_steps"] + config.runtime.max_steps
    record = {
        "status": "completed",
        "error": None,
        "platform_id": "l40s",
        "condition_id": "E320@l40s",
        "gpus": config.hardware.gpus,
        "git_dirty": False,
        "environment": {"git_dirty": False, "observed_gpu_model": "NVIDIA L40S"},
        "metrics": {
            "world_size": config.hardware.gpus,
            "scaling_runtime_sha256": scaling_runtime_sha256(),
            "dataset_execution": p["input_protocol"],
            "global_input_sha256": "a" * 64,
            "optimizer_step_calls": updates,
            "expected_optimizer_step_calls": updates,
            "backward_calls": updates * p["gradient_accumulation_steps"],
            "effective_global_batch_size": p["global_batch_size"],
            "strategy_observed": config.distributed.strategy,
            "attention_backend_observed": "sdpa",
            "training_use_cache": False,
            "throughput_tokens_sec": 100.0,
            "initial_training_loss": 2.0,
            "loss": 1.0,
            "memory_used_gb": 20.0,
        },
    }
    assert _canary_issues(record, config, "E320@l40s", "l40s") == []
    record["metrics"]["memory_used_gb"] = 40.0
    record["metrics"]["optimizer_step_calls"] = updates - 1
    record["metrics"]["attention_backend_observed"] = "eager"
    assert set(_canary_issues(record, config, "E320@l40s", "l40s")) == {
        "memory ceiling",
        "update accounting",
        "attention backend",
    }


def test_placement_runs_reserve_the_whole_l40s_node_but_start_only_their_ranks():
    placement = [(e, c) for e, c, _ in ENTRIES if c.allocation_gpus]
    assert placement and all(c.run_platforms == ("l40s",) and c.allocation_gpus == 8 for _, c in placement)
    path = config_path(placement[0][0])
    config = load_experiment_config(path)
    script = SlurmJob.from_config(config, path).render()
    assert "#SBATCH --gres=gpu:8" in script
    assert f"--nproc-per-node={config.hardware.gpus}" in script
    assert "export CUDA_DEVICE_ORDER=PCI_BUS_ID" in script
    with pytest.raises(ValueError, match="allocation_gpus"):
        SlurmJob.from_config(replace(config, metadata={**config.metadata, "allocation_gpus": 1}), path)


def base_config(**parameters):
    config = load_experiment_config(config_path("E386"))  # GPT-2 8 x 1024, one GPU
    return replace(config, workload=replace(config.workload, parameters={**config.workload.parameters, **parameters}))


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"ddp_comm_hook": "powersgd"}, "ddp_comm_hook"),
        ({"ddp_bucket_cap_mb": 0}, "bucket"),
        ({"fsdp_wrap_policy": "root"}, "only to the FSDP"),
        ({"model_load_dtype": "bf16"}, "FP32 parameters"),
        ({"text_batching": "pack"}, "real-text"),
        ({"ddp_static_graph": True, "gradient_accumulation_steps": 2, "global_batch_size": 16}, "static_graph"),
    ],
)
def test_deep_dive_contract_rejects_inconsistent_settings(changes, message):
    assert scaling_issues(base_config()) == []
    assert any(message in issue for issue in scaling_issues(base_config(**changes)))


def test_fsdp_and_transformer_engine_contracts_are_explicit():
    config = base_config(stack="fsdp")
    config = replace(config, distributed=replace(config.distributed, strategy="fsdp"))
    assert any("strategy_detail" in i for i in scaling_issues(config))
    tiny = load_experiment_config(config_path("E322"))
    te = replace(
        tiny,
        runtime=replace(tiny.runtime, precision="fp8"),
        workload=replace(
            tiny.workload,
            parameters={
                **tiny.workload.parameters,
                "precision_backend": "transformer_engine",
                "model_load_dtype": "bf16",
            },
        ),
    )
    assert any("GPT-2 Small and DistilBERT" in i for i in scaling_issues(te))


def runtime_config(gpus, accumulation, global_batch, **parameters):
    config = base_config(
        per_gpu_batch_size=global_batch // gpus // accumulation,
        global_batch_size=global_batch,
        gradient_accumulation_steps=accumulation,
        warmup_steps=1,
        sequence_length=8,
        **parameters,
    )
    return replace(
        config,
        hardware=replace(config.hardware, gpus=gpus),
        runtime=replace(config.runtime, max_steps=2),
    )


@pytest.mark.parametrize("gpus,accumulation", [(1, 1), (2, 1), (4, 1), (2, 2)])
def test_classification_inputs_keep_labels_with_their_sequences_across_shapes(gpus, accumulation):
    torch = pytest.importorskip("torch")
    make = lambda g, a, r: GlobalTrainingInputs(  # noqa: E731
        torch, runtime_config(g, a, 8), SimpleNamespace(rank=r, world_size=g), "cpu", 64, task="classification"
    )
    reference = make(1, 1, 0)
    shards = [make(gpus, accumulation, r) for r in range(gpus)]
    for _ in range(3):
        expected = reference.next_batch()
        batches = [s.next_batch() for s in shards for _ in range(accumulation)]
        assert torch.equal(torch.cat([b["input_ids"] for b in batches]), expected["input_ids"])
        assert torch.equal(torch.cat([b["labels"] for b in batches]), expected["labels"])
        assert batches[0]["attention_mask"].shape == batches[0]["input_ids"].shape
    assert {s.sha256 for s in shards} == {reference.sha256}


def test_causal_inputs_keep_their_original_stream_digest():
    torch = pytest.importorskip("torch")
    import hashlib

    config = runtime_config(1, 1, 8)
    inputs = GlobalTrainingInputs(torch, config, SimpleNamespace(rank=0, world_size=1), "cpu", 64)
    generator = torch.Generator(device="cpu").manual_seed(config.runtime.seed)
    original = torch.randint(0, 64, (3, 8, 8), generator=generator)
    assert inputs.sha256 == hashlib.sha256(original.numpy().tobytes()).hexdigest()


DOCUMENTS = [" ".join(["word"] * length) for length in [20, 300, 40, 900, 25, 60, 500, 18, 33, 700, 45, 80] * 6]


def real_text(gpus, accumulation, mode, rank=0, batch=2, sequence=256):
    torch = pytest.importorskip("torch")
    config = runtime_config(gpus, accumulation, gpus * accumulation * batch, text_batching=mode)
    config = replace(
        config,
        workload=replace(config.workload, parameters={**config.workload.parameters, "sequence_length": sequence}),
    )
    return RealTextInputs(
        torch,
        config,
        SimpleNamespace(rank=rank, world_size=gpus),
        "cpu",
        documents=DOCUMENTS,
        tokenize=lambda text: list(range(1, len(text.split()) + 1)),
        eos_id=0,
        measured_updates=2,
    )


@pytest.mark.parametrize("mode", ["pad_to_max", "dynamic_pad", "length_bucketed", "pack"])
def test_real_text_global_content_is_identical_across_gpu_counts(mode):
    reference = real_text(1, 1, mode, batch=4)
    for gpus, accumulation in [(2, 1), (4, 1), (2, 2)]:
        shards = [real_text(gpus, accumulation, mode, rank=r, batch=4 // gpus // accumulation) for r in range(gpus)]
        assert {s.sha256 for s in shards} == {reference.sha256}
        assert {s.useful_tokens for s in shards} == {reference.useful_tokens}
        lengths = sorted(int(m.sum()) for s in shards for b in s.batches_list for m in b["attention_mask"])
        assert lengths == sorted(int(m.sum()) for b in reference.batches_list for m in b["attention_mask"])


def test_real_text_padding_modes_trade_waste_for_balance():
    pad, dynamic, bucketed, pack = (
        real_text(4, 2, mode) for mode in ("pad_to_max", "dynamic_pad", "length_bucketed", "pack")
    )
    assert {tuple(b["input_ids"].shape) for b in pad.batches_list} == {(2, 256)}
    assert all(b["input_ids"].shape[1] % 8 == 0 for b in dynamic.batches_list)
    assert all((b["labels"] == -100).sum() == (b["attention_mask"] == 0).sum() for b in dynamic.batches_list)
    assert pack.processed_tokens == pack.useful_tokens and pad.processed_tokens > dynamic.processed_tokens
    assert bucketed.imbalance <= dynamic.imbalance and bucketed.processed_tokens <= dynamic.processed_tokens
    assert pad.truncated_documents > 0
    assert wikitext_documents([" = Heading = ", "", " A paragraph . "]) == ["A paragraph ."]


def test_ddp_options_pass_only_declared_settings():
    assert workload._ddp_options({}) == ({}, "none")
    kwargs, hook = workload._ddp_options(
        {"ddp_bucket_cap_mb": 100, "ddp_comm_hook": "bf16_compress", "ddp_static_graph": True}
    )
    assert kwargs == {"bucket_cap_mb": 100, "static_graph": True} and hook == "bf16_compress"


def test_fsdp_layer_wrap_finds_each_supported_block_type():
    block = type("Block", (), {})
    gpt2 = SimpleNamespace(transformer=SimpleNamespace(h=[block()]))
    llama = SimpleNamespace(model=SimpleNamespace(layers=[block()]))
    bert = SimpleNamespace(distilbert=SimpleNamespace(transformer=SimpleNamespace(layer=[block()])))
    assert {workload._transformer_layer_class(m) for m in (gpt2, llama, bert)} == {block}
    with pytest.raises(RuntimeError, match="transformer layer"):
        workload._transformer_layer_class(SimpleNamespace())


TOPOLOGY = "\n".join(
    [
        "\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tGPU4\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m",
        "GPU0\t X \tPIX\tSYS\tSYS\tSYS\tPIX\t0-31\t0\t\tN/A",
        "GPU1\tPIX\t X \tSYS\tSYS\tSYS\tPIX\t0-31\t0\t\tN/A",
        "GPU2\tSYS\tSYS\t X \tPIX\tSYS\tSYS\t0-31\t0\t\tN/A",
        "GPU3\tSYS\tSYS\tPIX\t X \tSYS\tSYS\t0-31\t0\t\tN/A",
        "GPU4\tSYS\tSYS\tSYS\tSYS\t X \tSYS\t32-63\t1\t\tN/A",
        "NIC0\tPIX\tPIX\tSYS\tSYS\tSYS\t X \t\t\t",
    ]
)
INVENTORY = "\n".join(f"GPU-{i}, 00000000:0{i}:00.0, 4, 4, 16, 16" for i in range(5))


@pytest.mark.parametrize(
    "ranks,expected",
    [([0], "single_gpu"), ([0, 1], "same_pcie_switch"), ([0, 2], "same_socket"), ([0, 4], "cross_socket")],
)
def test_placement_class_reads_the_trials_own_topology(ranks, expected):
    row = {
        "metric_training_rank_placement": json.dumps([{"gpu_uuid": f"GPU-{r}"} for r in ranks]),
        "metric_gpu_pcie_link_inventory": INVENTORY,
        "metric_gpu_topology_matrix": TOPOLOGY,
    }
    assert placement_class(row) == expected
    assert placement_class({"gpus": "1"}) == "single_gpu"
    assert placement_class({"gpus": "2"}) == "unknown"  # no placement evidence recorded
    if len(ranks) > 1:
        assert placement_class({**row, "metric_gpu_topology_matrix": ""}) == "unknown"


def trial(experiment_id, gpus, tps, step, platform="l40s", accumulation=1, **parameters):
    return {
        "status": "completed",
        "platform_id": platform,
        "experiment_id": experiment_id,
        "workload": "gpt2_small",
        "gpus": str(gpus),
        "strategy": "ddp",
        "precision": "bf16",
        "throughput_tokens_sec": str(tps),
        "memory_used_gb": "15",
        "metric_step_time_p50_ms": str(step),
        "metric_step_time_p95_ms": str(step * 1.1),
        "runtime_environment": '{"OMP_NUM_THREADS": "1"}',
        "parameter_scaling_study_id": DEEP_DIVE_STUDY_ID,
        "parameter_scaling_role": "performance",
        "parameter_per_gpu_batch_size": "8",
        "parameter_sequence_length": "1024",
        "parameter_gradient_accumulation_steps": str(accumulation),
        "parameter_global_batch_size": str(8 * gpus * accumulation),
        **{f"parameter_{k}": v for k, v in parameters.items()},
    }


def test_conditions_effects_and_report_from_trials():
    rows = [trial("E1", 1, v, 100) for v in (1000, 1010, 990)]
    rows += [trial("E2", 4, v, 125) for v in (3200, 3210, 3190)]
    rows += [trial("E3", 4, v, 106, accumulation=4) for v in (3800, 3790, 3810)]
    rows += [trial("E4", 1, v, 400, accumulation=4) for v in (1000, 1005, 995)]
    rows += [trial("E5", 4, v, 125, ddp_bucket_cap_mb="100") for v in (3200, 3220, 3180)]
    rows += [{**trial("E6", 4, 0, 0), "status": "failed", "error": "CUDA out of memory. Tried to allocate"}]
    conditions = build_scaling_deep_dive_conditions(rows)
    by_id = {c["experiment_id"]: c for c in conditions}
    assert by_id["E2"]["baseline_experiment_id"] == "E1"
    assert by_id["E2"]["scaling_efficiency"] == pytest.approx(0.8)
    assert by_id["E2"]["exposed_communication_ms"] == pytest.approx(25)
    assert by_id["E2"]["communication_fraction"] == pytest.approx(0.2)
    assert by_id["E3"]["baseline_experiment_id"] == "E4"  # same accumulation, one GPU
    assert by_id["E5"]["baseline_experiment_id"] == "E1"  # DDP settings do not apply at one GPU
    assert by_id["E6"]["oom_trials"] == 1 and by_id["E6"]["completed_trials"] == 0
    effects = build_scaling_deep_dive_effects(conditions)
    accumulation = next(e for e in effects if e["lever"] == "gradient_accumulation" and e["gpus"] == 4)
    assert accumulation["verdict"] == "helps" and accumulation["throughput_change_pct"] == pytest.approx(18.75)
    bucket = next(e for e in effects if e["lever"] == "ddp_communication")
    assert bucket["verdict"] == "no_material_effect"
    report = render_scaling_deep_dive_report(conditions, effects)
    assert "| gradient_accumulation | L40S | 1 |" in report and "CUDA OOM: 1" in report


def test_controlled_strong_curve_pools_accumulation_diagonal_but_legacy_does_not():
    def row(gpus, accumulation, study=DEEP_DIVE_STUDY_ID):
        r = trial(f"E-{gpus}", gpus, 1000 * gpus, 100, accumulation=accumulation)
        r.update(
            {
                "mode": "training",
                "phase": "Scaling deep dive",
                "scaling_type": "strong",
                "comparability": "exact",
                "model": "gpt2",
                "global_batch_size": "64",
                "per_gpu_batch_size": "8",
                "throughput_unit": "training_tokens_per_second",
                "parameter_scaling_study_id": study,
                "metric_total_tokens": "13107200",
                "metric_measured_seconds": str(13107200 / (1000 * gpus)),
            }
        )
        return r

    pooled = build_scaling_summary([row(1, 8), row(2, 4), row(4, 2), row(8, 1)])
    assert len({r["series_id"] for r in pooled if r["scaling_type"] == "strong"}) == 1
    legacy = build_scaling_summary([row(1, 8, ""), row(2, 4, "")])
    assert len({r["series_id"] for r in legacy if r["scaling_type"] == "strong"}) == 2
