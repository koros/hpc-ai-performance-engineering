"""Focused fresh acceptance for the original S7 performance matrix."""

import copy
import json
from pathlib import Path

from hpc_ai_perf.config.scaling_deep_dive import design as deep_dive_design

CONFIG_ROOT = "configs/s7_acceptance"
CANARY_WAVE = "s7-real-text-canary"


def design():
    sources = [(eid, c) for eid, c, _ in deep_dive_design() if c.role == "canary" and c.text_batching]
    rows = []
    for number, (source, _) in enumerate(sources, 810):
        data = copy.deepcopy(json.loads(next(Path("configs/scaling_deep_dive").glob(f"{source}_*.json")).read_text()))
        eid = f"E{number}"
        data["experiment_id"] = eid
        data["experiment"] = "s7_acceptance_" + data["experiment"]
        data["hardware"].update(platform="lovelace_l40s", gpu_model="NVIDIA L40S 48GB", interconnect="PCIe")
        data["runtime"]["environment"].update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        data["metadata"].update(
            matrix_id=eid,
            platform_wave=CANARY_WAVE,
            phase="S7 fresh real-text acceptance",
            source_experiment_ids=[source],
            objective="Certify S7 batching and token accounting in recovered runtime",
        )
        rows.append((eid, data))
    return rows
