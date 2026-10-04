"""Declarative design of the scaling deep dive (docs/scaling_deep_dive_plan.md).

Each study lists the conditions it needs. Identical conditions requested by
several studies are generated once, so one physical run serves every analysis
it fits (for example the 8 x 1024 GPT-2 accumulation-1 run is also a batch /
sequence split point). ``scripts/generate_scaling_deep_dive.py`` writes the
configs and execution-matrix waves from this module; tests check the same
definitions, including the memory rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from hpc_ai_perf.config.scaling import (
    DEEP_DIVE_STUDY_ID,
    INPUT_PROTOCOL,
    REAL_TEXT_DATASET,
    REAL_TEXT_PROTOCOL,
    REAL_TEXT_REVISION,
    REAL_TEXT_SUBSET,
)

FIRST_ID = 320
CONFIG_DIR = "configs/scaling_deep_dive"
PHASE = "Scaling deep dive"
WAVE_PREFIX = "scaling-dd-"
MEASURED_UPDATES = 200
CANARY_UPDATES = 20
LONG_RUN_UPDATES = 1000
TRIALS = 3
GPU_COUNTS = (1, 2, 4, 8)
MULTI_GPU = (2, 4, 8)
PLATFORM_MAX_GPUS = {"h100": 4, "l40s": 8}
CAPACITY_GB = {"h100": 79.19, "l40s": 44.52}
MEMORY_RULE = 0.85

WORKLOADS: dict[str, dict[str, Any]] = {
    "gpt2_small": {"model": "gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "max_seq": 1024},
    "gpt2_medium": {
        "model": "openai-community/gpt2-medium",
        "revision": "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
        "max_seq": 1024,
    },
    "gpt2_large": {
        "model": "openai-community/gpt2-large",
        "revision": "32b71b12589c2f8d625668d2335a01cac3249519",
        "max_seq": 1024,
    },
    "distilbert": {
        "model": "distilbert-base-uncased",
        "revision": "12040accade4e8a0f71eabdb258fecc2e7e948be",
        "max_seq": 512,
    },
    "tinyllama": {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "max_seq": 2048,
    },
}

# Peak memory per GPU (GiB) = training state + other fixed + activations per 1k tokens per
# micro-step. Training state is FP32 parameters, gradients and Adam moments (16 bytes per
# parameter); DDP adds an FP32 gradient-bucket copy (4 bytes per parameter). Activation slopes
# are fitted conservatively to completed runs (docs/scaling_deep_dive_plan.md, section 3).
MEMORY_MODEL = {  # (parameters in millions, other fixed GiB, GiB per 1k tokens)
    "gpt2_small": (124, 0.5, 1.65),
    "gpt2_medium": (355, 0.5, 4.4),
    "gpt2_large": (774, 0.5, 8.2),
    "distilbert": (66, 1.3, 0.2),
    "tinyllama": (1100, 0.0, 3.7),
}
FSDP_GATHER_GB = 1.0
FP32_ACTIVATION_FACTOR = 1.25

# Primary per-workload shape: what each lever is applied to.
PRIMARY_SHAPE = {"gpt2_small": (8, 1024), "distilbert": (32, 512), "tinyllama": (4, 1024)}
SPLITS = {
    "gpt2_small": [[(32, 256), (16, 512), (8, 1024)], [(8, 256), (4, 512), (2, 1024)]],
    "distilbert": [[(64, 128), (32, 256), (16, 512)], [(16, 128), (8, 256), (4, 512)]],
    "tinyllama": [[(16, 256), (8, 512), (4, 1024)], [(8, 256), (4, 512), (2, 1024)]],
}
L40S_PCI_ORDER = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


@dataclass(frozen=True)
class Condition:
    workload: str
    gpus: int
    micro_batch: int
    sequence_length: int
    accumulation: int = 1
    strategy: str = "ddp"
    strategy_detail: str = ""
    fsdp_wrap: str = ""
    ddp: tuple[tuple[str, Any], ...] = ()
    precision: str = "bf16"  # bf16 | fp32 | te_bf16 | te_fp8
    text_batching: str = ""  # empty means synthetic inputs
    environment: tuple[tuple[str, str], ...] = ()
    allocation_gpus: int = 0
    platforms: tuple[str, ...] = ()
    role: str = "performance"
    updates: int = MEASURED_UPDATES

    @property
    def global_batch(self) -> int:
        return self.gpus * self.micro_batch * self.accumulation

    @property
    def run_platforms(self) -> tuple[str, ...]:
        if self.platforms:
            return self.platforms
        return tuple(p for p in ("h100", "l40s") if self.gpus <= PLATFORM_MAX_GPUS[p])


@dataclass
class Study:
    key: str
    wave: str
    title: str
    objective: str
    research_questions: tuple[str, ...]
    comparability: str = "exact"
    conditions: list[Condition] = field(default_factory=list)


def estimated_memory_gb(condition: Condition) -> float:
    params_m, other, per_k = MEMORY_MODEL[condition.workload]
    state = params_m * 1e6 * 16 / 1024**3
    activations = per_k * condition.micro_batch * condition.sequence_length / 1024
    if condition.precision == "fp32":
        activations *= FP32_ACTIVATION_FACTOR
    if condition.strategy == "fsdp" and condition.gpus > 1:
        # State shards across ranks; one gathered block is materialised at a time.
        return state / condition.gpus + FSDP_GATHER_GB + other + activations
    return state * 1.25 + other + activations


def studies() -> list[Study]:
    s1 = Study(
        "S1",
        "accumulation",
        "Gradient accumulation and strong vs weak scaling",
        "Test whether syncing gradients less often recovers scaling efficiency",
        ("RQ1", "RQ4", "RQ5", "RQ7"),
    )
    for workload, (batch, seq) in PRIMARY_SHAPE.items():
        for accumulation in (1, 2, 4, 8):
            s1.conditions += [Condition(workload, g, batch, seq, accumulation) for g in GPU_COUNTS]

    s2 = Study(
        "S2",
        "split",
        "Tokens per GPU: batch versus sequence length",
        "Test whether efficiency follows tokens per GPU regardless of the batch x sequence split",
        ("RQ1", "RQ4", "RQ5", "RQ7"),
    )
    for workload, levels in SPLITS.items():
        shapes = [shape for level in levels for shape in level]
        if workload == "gpt2_small":
            shapes += [(4, 1024), (16, 1024)]
        for batch, seq in shapes:
            s2.conditions += [Condition(workload, g, batch, seq) for g in GPU_COUNTS]

    s3 = Study(
        "S3",
        "model-size",
        "Model size and gradient volume",
        "Measure how scaling efficiency falls as parameter count grows at matched work per GPU",
        ("RQ1", "RQ4", "RQ5", "RQ6", "RQ7"),
    )
    for workload in ("distilbert", "gpt2_small", "gpt2_medium", "gpt2_large", "tinyllama"):
        for accumulation in (1, 4):
            s3.conditions += [Condition(workload, g, 4, 512, accumulation) for g in GPU_COUNTS]

    s4 = Study(
        "S4",
        "ddp-comm",
        "DDP communication tuning",
        "Find which DDP bucket, compression and graph settings recover efficiency on PCIe and NVLink",
        ("RQ1", "RQ4", "RQ5", "RQ6", "RQ7"),
    )
    ddp_variants = [
        (("ddp_bucket_cap_mb", 5),),
        (("ddp_bucket_cap_mb", 100),),
        (("ddp_comm_hook", "bf16_compress"),),
        (("ddp_bucket_cap_mb", 100), ("ddp_comm_hook", "bf16_compress")),
        (("ddp_gradient_as_bucket_view", True), ("ddp_static_graph", True)),
    ]
    for workload, (batch, seq) in PRIMARY_SHAPE.items():
        for variant in ddp_variants:
            s4.conditions += [Condition(workload, g, batch, seq, ddp=variant) for g in MULTI_GPU]

    s5 = Study(
        "S5",
        "stacks",
        "Training stack scaling curves",
        "Compare DDP with FSDP sharding modes and wrapping as GPUs are added under identical inputs",
        ("RQ1", "RQ2", "RQ4", "RQ5", "RQ7"),
    )
    for workload, (batch, seq) in PRIMARY_SHAPE.items():
        for detail, wrap in (
            ("sharded_grad_op", "transformer_layer"),
            ("full_shard", "transformer_layer"),
            ("full_shard", "root"),
        ):
            s5.conditions += [
                Condition(workload, g, batch, seq, strategy="fsdp", strategy_detail=detail, fsdp_wrap=wrap)
                for g in MULTI_GPU
            ]
    # FSDP's practical advantage: a micro-batch DDP cannot fit on L40S.
    # Two GPUs leave too little sharding to fit the memory rule (about 39 GiB on 44.5 GiB).
    s5.conditions += [
        Condition("tinyllama", g, 8, 1024, strategy="fsdp", strategy_detail="full_shard", fsdp_wrap="transformer_layer")
        for g in (4, 8)
    ]

    s6 = Study(
        "S6",
        "topology",
        "GPU placement, peer-to-peer and NCCL algorithm",
        "Measure how placement on PCIe switches and sockets, P2P and NCCL algorithms change scaling",
        ("RQ1", "RQ4", "RQ6", "RQ7"),
        comparability="platform_specific",
    )
    l40s_sets = {
        2: ["0,1", "0,2", "0,4"],  # same PCIe switch, same socket, across sockets
        4: ["0,1,2,3", "0,1,4,5", "0,2,4,6"],  # one socket, two sockets paired, no shared switch
    }
    for workload in ("gpt2_small", "tinyllama"):
        batch, seq = PRIMARY_SHAPE[workload]
        for gpus, device_sets in l40s_sets.items():
            for devices in device_sets:
                env = (("CUDA_DEVICE_ORDER", "PCI_BUS_ID"), ("CUDA_VISIBLE_DEVICES", devices))
                s6.conditions.append(
                    Condition(workload, gpus, batch, seq, environment=env, allocation_gpus=8, platforms=("l40s",))
                )
        for devices in ("0,1", "0,4"):
            env = (("CUDA_DEVICE_ORDER", "PCI_BUS_ID"), ("CUDA_VISIBLE_DEVICES", devices), ("NCCL_P2P_DISABLE", "1"))
            s6.conditions.append(
                Condition(workload, 2, batch, seq, environment=env, allocation_gpus=8, platforms=("l40s",))
            )
        for env in ((("NCCL_ALGO", "Ring"),), (("NCCL_ALGO", "Tree"),), (("NCCL_P2P_DISABLE", "1"),)):
            s6.conditions.append(Condition(workload, 8, batch, seq, environment=env, platforms=("l40s",)))
        for gpus in (2, 4):
            s6.conditions.append(
                Condition(workload, gpus, batch, seq, environment=(("NCCL_P2P_DISABLE", "1"),), platforms=("h100",))
            )
        for algorithm in ("Ring", "Tree"):
            s6.conditions.append(
                Condition(workload, 4, batch, seq, environment=(("NCCL_ALGO", algorithm),), platforms=("h100",))
            )

    s7 = Study(
        "S7",
        "real-text",
        "Variable-length real text: padding, packing and stragglers",
        "Measure the efficiency cost of variable-length WikiText-103 inputs and how packing or bucketing recovers it",
        ("RQ1", "RQ4", "RQ5", "RQ7"),
    )
    for workload in ("gpt2_small", "tinyllama"):
        batch, seq = PRIMARY_SHAPE[workload]
        for mode in ("pad_to_max", "dynamic_pad", "length_bucketed", "pack"):
            s7.conditions += [Condition(workload, g, batch, seq, text_batching=mode) for g in GPU_COUNTS]

    s8 = Study(
        "S8",
        "precision",
        "Numeric precision and scaling",
        "Test whether faster per-GPU compute (BF16, FP8) leaves communication more exposed than FP32",
        ("RQ1", "RQ3", "RQ5", "RQ7"),
    )
    for workload, shape in (("gpt2_small", (8, 1024)), ("distilbert", (32, 512)), ("tinyllama", (2, 1024))):
        s8.conditions += [Condition(workload, g, *shape, precision="fp32") for g in GPU_COUNTS]
        s8.conditions += [Condition(workload, g, *shape) for g in GPU_COUNTS]
    for workload in ("gpt2_small", "distilbert"):
        for precision in ("te_bf16", "te_fp8"):
            s8.conditions += [Condition(workload, g, *PRIMARY_SHAPE[workload], precision=precision) for g in GPU_COUNTS]

    long_run = Study(
        "RL",
        "run-length",
        "Run-length check",
        "Confirm that 200 measured updates give the same throughput as 1,000",
        ("RQ1", "RQ7"),
    )
    for workload in ("gpt2_small", "tinyllama"):
        batch, seq = PRIMARY_SHAPE[workload]
        long_run.conditions += [Condition(workload, g, batch, seq, updates=LONG_RUN_UPDATES) for g in (4, 8)]

    return [s1, s2, s3, s4, s5, s6, s7, s8, long_run]


def canary_study(performance: list[Study]) -> Study:
    """One-trial, short canaries for every distinct memory/backend risk before the waves."""
    canary = Study(
        "C",
        "canary",
        "Deep-dive acceptance canaries",
        "Verify memory headroom, backends and input protocols at the least-sharded GPU count before the waves",
        ("RQ1", "RQ7"),
        comparability="platform_specific",
    )
    risks: dict[tuple, Condition] = {}
    for study in performance:
        for c in study.conditions:
            risk = (
                c.workload,
                c.micro_batch,
                c.sequence_length,
                c.strategy,
                c.strategy_detail,
                c.fsdp_wrap,
                c.ddp,
                c.precision,
                c.text_batching,
                c.allocation_gpus > 0,
            )
            if c.updates != MEASURED_UPDATES or c.gpus < 2:
                continue
            # Test each risk at the smallest multi-GPU count it runs at: the least sharding.
            if risk not in risks or c.gpus < risks[risk].gpus:
                risks[risk] = c
    for c in risks.values():
        canary.conditions.append(
            replace(
                c,
                accumulation=1,
                environment=c.environment if c.allocation_gpus else (),
                role="canary",
                updates=CANARY_UPDATES,
            )
        )
    # Largest preloaded input stream: DistilBERT accumulation 8 on all eight L40S GPUs.
    canary.conditions.append(Condition("distilbert", 8, 32, 512, 8, role="canary", updates=CANARY_UPDATES))
    return canary


def design() -> list[tuple[str, Condition, list[Study]]]:
    """Assign IDs in first-appearance order; return (id, condition, studies using it)."""
    performance = studies()
    ordered: dict[Condition, list[Study]] = {}
    for study in [canary_study(performance), *performance]:
        for condition in study.conditions:
            ordered.setdefault(condition, [])
            if study not in ordered[condition]:
                ordered[condition].append(study)
    return [(f"E{FIRST_ID + i}", c, s) for i, (c, s) in enumerate(ordered.items())]


def slug(condition: Condition) -> str:
    parts = [condition.workload, f"mb{condition.micro_batch}", f"seq{condition.sequence_length}"]
    if condition.accumulation != 1:
        parts.append(f"ga{condition.accumulation}")
    if condition.strategy == "fsdp":
        parts.append(f"fsdp_{condition.strategy_detail}_{condition.fsdp_wrap}")
    for key, value in condition.ddp:
        parts.append(f"{key.removeprefix('ddp_')}_{value}".lower())
    if condition.precision != "bf16":
        parts.append(condition.precision)
    if condition.text_batching:
        parts.append(f"wikitext_{condition.text_batching}")
    for key, value in condition.environment:
        if key != "CUDA_DEVICE_ORDER":
            parts.append(f"{key.lower()}_{value.replace(',', '')}".lower())
    if condition.updates != MEASURED_UPDATES:
        parts.append(f"{condition.updates}updates")
    if condition.role == "canary":
        parts.append("canary")
    parts.append(f"{condition.gpus}gpu")
    return "_".join(parts)


def build_config(
    experiment_id: str, condition: Condition, used_by: list[Study], ids: dict[Condition, str] | None = None
) -> dict[str, Any]:
    workload = WORKLOADS[condition.workload]
    p: dict[str, Any] = {
        "scaling_study_id": DEEP_DIVE_STUDY_ID,
        "scaling_role": condition.role,
        "scaling_type": "strong",
        "fixed_workload": True,
        "input_protocol": REAL_TEXT_PROTOCOL if condition.text_batching else INPUT_PROTOCOL,
        "stack": condition.strategy,
        "per_gpu_batch_size": condition.micro_batch,
        "global_batch_size": condition.global_batch,
        "gradient_accumulation_steps": condition.accumulation,
        "sequence_length": condition.sequence_length,
        "warmup_steps": 10,
        "learning_rate": 0.00005,
        "model_load_dtype": "bf16" if condition.precision.startswith("te_") else "fp32",
        "attention_backend": "sdpa",
        "activation_checkpointing": False,
        "training_update_diagnostics": True,
        "capture_gpu_topology": True,
        # Every run also belongs to the weak curve that fixes its micro-batch.
        "scaling_weak_per_gpu_batch_size": condition.micro_batch,
    }
    if condition.workload == "distilbert":
        p["num_labels"] = 2
    if condition.strategy == "fsdp":
        p["strategy_detail"] = condition.strategy_detail
        p["fsdp_wrap_policy"] = condition.fsdp_wrap
    p.update(dict(condition.ddp))
    if condition.precision.startswith("te_"):
        p.update({"precision_backend": "transformer_engine", "base_compute_dtype": "bf16"})
        if condition.precision == "te_fp8":
            p.update({"quantization_method": "fp8_delayed_scaling_hybrid", "fp8_amax_history_len": 16})
            control = replace(condition, precision="te_bf16")
            p["comparison_control_id"] = (ids or {}).get(control, "")
    if condition.text_batching:
        p.update(
            {
                "dataset_revision": REAL_TEXT_REVISION,
                "dataset_subset": REAL_TEXT_SUBSET,
                "text_batching": condition.text_batching,
            }
        )
    precision = {"bf16": "bf16", "fp32": "fp32", "te_bf16": "bf16", "te_fp8": "fp8"}[condition.precision]
    l40s_only = condition.run_platforms == ("l40s",)
    hardware = (
        {"platform": "lovelace_l40s", "gpus": condition.gpus, "gpu_model": "NVIDIA L40S 48GB", "interconnect": "PCIe"}
        if l40s_only
        else {
            "platform": "lovelace_h100",
            "gpus": condition.gpus,
            "gpu_model": "NVIDIA H100 SXM 80GB",
            "interconnect": "NVLink",
        }
    )
    profilers = ["nvidia_smi", "torch_memory"] + (["nccl"] if condition.gpus > 1 else [])
    metadata: dict[str, Any] = {
        "matrix_id": experiment_id,
        "phase": PHASE,
        "priority": "Must Have",
        "gpu_hours_estimate": round(gpu_hours(condition), 3),
        "serial_submission": True,
        "objective": "; ".join(s.objective for s in used_by),
        "scaling_studies": [f"{s.key}:{s.wave}" for s in used_by],
        "research_questions": sorted({q for s in used_by for q in s.research_questions}),
        "analysis_tasks": ["A01", "A03", "A04", "A05", "A07"],
        "key_metrics": ["throughput", "scaling_efficiency", "step_time_p95", "gpu_hours", "memory", "input_identity"],
        "estimated_peak_memory_gb": round(estimated_memory_gb(condition), 1),
        "comparison_scope": "controlled_scaling_deep_dive_v1_not_legacy_static_input",
    }
    if condition.allocation_gpus:
        metadata["allocation_gpus"] = condition.allocation_gpus
    return {
        "experiment_id": experiment_id,
        "experiment": f"scaling_deep_dive_{slug(condition)}",
        "mode": "training",
        "workload": {
            "name": condition.workload,
            "model": workload["model"],
            "model_revision": workload["revision"],
            "dataset": REAL_TEXT_DATASET if condition.text_batching else INPUT_PROTOCOL,
            "parameters": p,
        },
        "hardware": hardware,
        "distributed": {"strategy": condition.strategy, "nodes": 1, "gpus_per_node": condition.gpus, "backend": "nccl"},
        "runtime": {
            "precision": precision,
            "max_steps": condition.updates,
            "seed": 1337,
            "output_dir": "results/raw",
            "environment": {"OMP_NUM_THREADS": "1", **dict(condition.environment)},
        },
        "profiling": {"enabled": True, "profilers": profilers, "parameters": {"sample_interval_seconds": 0.1}},
        "metadata": metadata,
    }


def gpu_hours(condition: Condition) -> float:
    """Planning estimate: ~5 minutes of start-up plus the measured loop, per reserved GPU."""
    reserved = max(condition.gpus, condition.allocation_gpus)
    minutes = 5 + 2 * condition.updates / MEASURED_UPDATES * condition.accumulation
    return reserved * minutes / 60


def waves(entries: list[tuple[str, Condition, list[Study]]]) -> list[dict[str, Any]]:
    """One serial wave per owning study and platform; a condition's first study owns it."""
    grouped: dict[tuple[str, str], list[str]] = {}
    owners: dict[str, Study] = {}
    for experiment_id, condition, used_by in entries:
        owner = used_by[0]
        owners[owner.wave] = owner
        for platform in condition.run_platforms:
            grouped.setdefault((owner.wave, platform), []).append(experiment_id)
    result = []
    for (wave, platform), ids in grouped.items():
        owner = owners[wave]
        result.append(
            {
                "platform": platform,
                "wave": WAVE_PREFIX + wave,
                "name": f"{platform.upper()} scaling deep dive {owner.key}: {owner.title}",
                "experiment_ids": compress_ids(ids),
                "required_trials": 1 if owner.key == "C" else TRIALS,
                "comparability": owner.comparability,
            }
        )
    return sorted(result, key=lambda w: (_wave_order(w["wave"]), w["platform"]))


def _wave_order(wave: str) -> int:
    order = [
        "canary",
        "accumulation",
        "split",
        "model-size",
        "ddp-comm",
        "stacks",
        "topology",
        "real-text",
        "precision",
        "run-length",
    ]
    return order.index(wave.removeprefix(WAVE_PREFIX))


def compress_ids(ids: list[str]) -> list[str]:
    numbers = sorted(int(i[1:]) for i in ids)
    ranges, start, previous = [], numbers[0], numbers[0]
    for number in numbers[1:] + [None]:
        if number is not None and number == previous + 1:
            previous = number
            continue
        ranges.append(f"E{start}" if start == previous else f"E{start}-E{previous}")
        if number is not None:
            start = previous = number
    return ranges
