"""Focused L40S larger-model follow-up; no changes to historical TinyLlama inputs."""

import copy
import json
from pathlib import Path

CANARY_WAVE = "qwen7b-layout-canary"
PERFORMANCE_WAVE = "qwen7b-layout-performance"
EXTENSION_CANARY_WAVE = "qwen7b-layout-extension-canary"
EXTENSION_PERFORMANCE_WAVE = "qwen7b-layout-extension-performance"
LOW_CONCURRENCY_CANARY_WAVE = "qwen7b-layout-low-concurrency-canary"
LOW_CONCURRENCY_PERFORMANCE_WAVE = "qwen7b-layout-low-concurrency-performance"
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
    completed = rows + extension_design(rows)
    return completed + low_concurrency_design(completed)


def extension_design(original_rows):
    """Add valid TP4; 28 query heads rule out TP8 for this pinned model."""
    rows = []
    specifications = [(True, 1280)]
    specifications += [(False, prompt) for prompt in (128, 1280)]
    for number, (canary, prompt) in enumerate(specifications, 756):
        original = original_rows[0 if canary else (2 if prompt == 128 else 4)][3]
        data = copy.deepcopy(original)
        template_id = "E717"
        template = json.loads(next(Path("configs/inference_crossover").glob(f"{template_id}_*.json")).read_text())
        p = data["workload"]["parameters"]
        for key in ("serving_layout", "tensor_parallel_size", "data_parallel_size", "replica_count", "request_routing"):
            p[key] = template["workload"]["parameters"][key]
        if not canary:
            p["comparison_control_id"] = "E752" if prompt == 128 else "E754"
        eid = f"E{number}"
        wave = EXTENSION_CANARY_WAVE if canary else EXTENSION_PERFORMANCE_WAVE
        data["experiment_id"] = eid
        data["experiment"] = f"{STUDY}_{p['serving_layout']}_p{prompt}_c8_{'canary' if canary else 'performance'}"
        data["metadata"].update(
            matrix_id=eid,
            platform_wave=wave,
            matrix_stack_strategy=f"vLLM {p['serving_layout']}",
            objective="Extend the Qwen7B throughput-energy comparison with TP4 replicas",
        )
        rows.append((eid, "l40s", wave, data))
    return rows


def low_concurrency_design(completed):
    """Concurrency-one extension with enough acceptance rounds to exercise every replica."""
    by_id = {row[0]: row[3] for row in completed}
    rows = []
    specs = [(True, 1), (True, 2), (True, 4), (False, 1), (False, 2), (False, 4)]
    references = {1: "E754", 2: "E755", 4: "E758"}
    for number, (canary, tp) in enumerate(specs, 759):
        data = copy.deepcopy(by_id[references[tp]])
        eid = f"E{number}"
        wave = LOW_CONCURRENCY_CANARY_WAVE if canary else LOW_CONCURRENCY_PERFORMANCE_WAVE
        p = data["workload"]["parameters"]
        p.update(concurrent_requests=1, warmup_steps=2 if canary else 5)
        p.pop("comparison_control_id", None)
        if not canary and tp != 1:
            p["comparison_control_id"] = "E762"
        # Eight measured singleton rounds guarantee acceptance covers all replicas.
        data["runtime"]["max_steps"] = 8 if canary else 30
        p["comparison_scope"] = f"{STUDY}_{'canary' if canary else 'performance'}"
        data["experiment_id"] = eid
        data["experiment"] = f"{STUDY}_{p['serving_layout']}_p1280_c1_{'canary' if canary else 'performance'}"
        data["metadata"].update(
            matrix_id=eid,
            platform_wave=wave,
            comparison_scope=p["comparison_scope"],
            concurrency_reference_id=references[tp],
            objective="Measure Qwen7B layout preference at concurrency one",
            phase="Qwen7B layout acceptance" if canary else "Qwen7B layout follow-up",
            gpu_hours_estimate=8 * (0.25 if canary else 1.0),
        )
        rows.append((eid, "l40s", wave, data))
    return rows
