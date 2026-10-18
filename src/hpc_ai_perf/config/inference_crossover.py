"""Controlled equal-budget inference layouts; separate from historical TP rows."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

STUDY = "inference_layout_crossover_v1"
CANARY_WAVE = "inference-crossover-canary"
LAYOUTS = {
    "l40s": [(1, 8), (2, 4), (4, 2), (8, 1)],
    "h100": [(1, 4), (2, 2), (4, 1)],
}


def serving_runtime_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in (
        "serving_layout.py",
        "workloads/transformer.py",
        "workloads/vllm_router.py",
        "profiling/measurement.py",
        "config/inference_crossover.py",
    ):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def design() -> list[tuple[str, str, str, dict]]:
    base = json.loads(Path("configs/inference/E711_tinyllama_tp2dp2_concurrency4.json").read_text())
    specifications = []
    for platform, layouts in LAYOUTS.items():
        for tp, dp in layouts:
            specifications.append((platform, CANARY_WAVE, tp, dp, 1280, 8, True))
    for platform, layouts in LAYOUTS.items():
        for prompt in (128, 1280):
            for concurrency in (1, 8):
                for tp, dp in layouts:
                    wave = "inference-crossover-short" if prompt == 128 else "inference-crossover-long"
                    specifications.append((platform, wave, tp, dp, prompt, concurrency, False))
    rows = []
    controls = {}
    for number, (platform, wave, tp, dp, prompt, concurrency, canary) in enumerate(specifications, 715):
        eid = f"E{number:03d}"
        layout = f"tp{tp}_dp{dp}" if dp > 1 else f"tp{tp}"
        data = copy.deepcopy(base)
        data["experiment_id"] = eid
        data["experiment"] = (
            f"{STUDY}_{platform}_{layout}_p{prompt}_c{concurrency}_{'canary' if canary else 'performance'}"
        )
        p = data["workload"]["parameters"]
        p.update(
            {
                "inference_study_type": STUDY,
                "serving_layout": layout,
                "tensor_parallel_size": tp,
                "data_parallel_size": dp,
                "replica_count": dp,
                "request_routing": "round_robin" if dp > 1 else "single_process",
                "prompt_length": prompt,
                "concurrent_requests": concurrency,
                "max_new_tokens": 128,
                "warmup_steps": 2 if canary else 5,
                "ignore_eos": True,
                "max_model_len": 2048,
                "enable_prefix_caching": False,
                "gpu_memory_utilization": 0.85,
                "comparison_scope": f"{STUDY}_{platform}_{'canary' if canary else 'performance'}",
            }
        )
        p.pop("comparison_control_id", None)
        key = (platform, prompt, concurrency)
        if not canary:
            controls.setdefault(key, eid)
            if controls[key] != eid:
                p["comparison_control_id"] = controls[key]
        gpus = tp * dp
        data["hardware"] = {
            "platform": f"lovelace_{platform}",
            "gpus": gpus,
            "gpu_model": "NVIDIA L40S 48GB" if platform == "l40s" else "NVIDIA H100 SXM 80GB",
            "interconnect": "PCIe" if platform == "l40s" else "NVLink",
        }
        data["distributed"]["gpus_per_node"] = gpus
        data["runtime"]["max_steps"] = 2 if canary else 30
        data["metadata"].update(
            {
                "matrix_id": eid,
                "study_family": STUDY,
                "platform_wave": wave,
                "research_questions": ["RQ1", "RQ4", "RQ5", "RQ6", "RQ7"],
                "analysis_tasks": ["A02", "A04", "A06", "A08"],
                "phase": "Inference layout acceptance" if canary else "Inference layout crossover",
                "priority": "Must Have",
                "comparison_scope": p["comparison_scope"],
                "source_experiment_ids": [],
                "serial_submission": True,
                "matrix_stack_strategy": f"vLLM {layout}",
                "objective": (
                    "Compare TP versus independent replicas at a fixed node GPU budget "
                    "across prompt and concurrency levels"
                ),
                "gpu_hours_estimate": round(gpus * (0.1 if canary else 0.5), 2),
                "limitation": (
                    "Planning allowance only; benchmark timings determine scheduling. "
                    "Closed-round synthetic prompts; no production arrivals, quality "
                    "or collective-time attribution."
                ),
            }
        )
        rows.append((eid, platform, wave, data))
    return rows
