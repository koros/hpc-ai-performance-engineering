"""Focused L40S larger-model follow-up; no changes to historical TinyLlama inputs."""

import copy
import json
from pathlib import Path

CANARY_WAVE = "qwen7b-layout-canary"
PERFORMANCE_WAVE = "qwen7b-layout-performance"
CONFIG_ROOT = "configs/qwen_layout"
STUDY = "qwen7b_equal_budget_layout_v1"


def design():
    templates = {
        1: json.loads(next(Path("configs/inference_crossover").glob("E715_*.json")).read_text()),
        2: json.loads(next(Path("configs/inference_crossover").glob("E716_*.json")).read_text()),
    }
    qwen = json.loads(next(Path("configs/inference").glob("E301_*.json")).read_text())["workload"]
    specifications = [(True, 1280, 1), (True, 1280, 2)]
    specifications += [(False, prompt, tp) for prompt in (128, 1280) for tp in (1, 2)]
    rows = []
    for number, (canary, prompt, tp) in enumerate(specifications, 750):
        data = copy.deepcopy(templates[tp])
        eid = f"E{number}"
        wave = CANARY_WAVE if canary else PERFORMANCE_WAVE
        data["experiment_id"] = eid
        data["experiment"] = f"{STUDY}_tp{tp}_dp{8 // tp}_p{prompt}_c8_{'canary' if canary else 'performance'}"
        for key in ("name", "model", "model_revision"):
            data["workload"][key] = qwen[key]
        p = data["workload"]["parameters"]
        p.update(
            inference_study_type=STUDY,
            prompt_length=prompt,
            warmup_steps=2 if canary else 5,
            comparison_scope=f"{STUDY}_{'canary' if canary else 'performance'}",
        )
        if not canary and tp == 2:
            p["comparison_control_id"] = f"E{number - 1}"
        data["runtime"]["max_steps"] = 2 if canary else 30
        data["runtime"]["environment"].update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        data["metadata"].update(
            matrix_id=eid,
            study_family=STUDY,
            platform_wave=wave,
            phase="Qwen7B layout acceptance" if canary else "Qwen7B layout follow-up",
            comparison_scope=p["comparison_scope"],
            source_experiment_ids=[],
            matrix_stack_strategy=f"vLLM tp{tp}_dp{8 // tp}",
            objective="Test whether the TinyLlama throughput-energy trade-off persists with Qwen7B",
            gpu_hours_estimate=8 * (0.25 if canary else 1.0),
        )
        rows.append((eid, "l40s", wave, data))
    return rows
