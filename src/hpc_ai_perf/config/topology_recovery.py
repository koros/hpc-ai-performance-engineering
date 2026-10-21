"""Independent recovery cohort for interrupted L40S S6 comparisons."""

import copy
import hashlib
import json
from pathlib import Path

CONFIG_ROOT = "configs/topology_recovery"
CANARY_WAVE = "topology-recovery-canary"
PERFORMANCE_WAVE = "topology-recovery-performance"
STUDY_ID = "l40s_topology_recovery_v1"
# Fresh controls are confined to comparisons touched by interruption/Tree failure.
PERFORMANCE_SOURCES = (
    "E389",
    "E616",
    "E617",
    "E623",
    "E625",
    "E626",
    "E627",
    "E628",
    "E629",
    "E630",
    "E421",
    "E631",
    "E632",
    "E633",
)
CANARY_SOURCES = ("E617", "E628", "E630", "E632", "E633")


def design():
    rows = []
    specs = [(True, source) for source in CANARY_SOURCES]
    specs += [(False, source) for source in PERFORMANCE_SOURCES]
    for number, (canary, source) in enumerate(specs, 791):
        data = copy.deepcopy(json.loads(next(Path("configs/scaling_deep_dive").glob(f"{source}_*.json")).read_text()))
        eid = f"E{number}"
        wave = CANARY_WAVE if canary else PERFORMANCE_WAVE
        data["experiment_id"] = eid
        data["experiment"] = f"{STUDY_ID}_{source}_{'canary' if canary else 'performance'}"
        data["hardware"].update(platform="lovelace_l40s", gpu_model="NVIDIA L40S 48GB", interconnect="PCIe")
        p = data["workload"]["parameters"]
        p.update(
            scaling_study_id=STUDY_ID,
            scaling_role="canary" if canary else "performance",
            scaling_type="weak",
            fixed_workload=False,
            comparison_scope=STUDY_ID,
        )
        data["runtime"]["max_steps"] = 20 if canary else 200
        env = data["runtime"]["environment"]
        if env.get("NCCL_ALGO") in {"Ring", "Tree"}:
            env["NCCL_ALGO"] = "allreduce:" + env["NCCL_ALGO"]
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        if canary:
            env.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,ENV,GRAPH")
        data["metadata"].update(
            matrix_id=eid,
            phase="L40S topology recovery",
            platform_wave=wave,
            study_family=STUDY_ID,
            comparison_scope=STUDY_ID,
            source_experiment_ids=[source],
            objective="Recover interrupted topology comparisons with current-runtime matched controls",
            study_design_note="recovery of interrupted S6; collective-scoped algorithm correction",
            historical_evidence_policy="preserve separately; never pool across historical/current runtime cohorts",
            historical_source_config_sha256=hashlib.sha256(
                next(Path("configs/scaling_deep_dive").glob(f"{source}_*.json")).read_bytes()
            ).hexdigest(),
            algorithm_scope="allreduce only; other collectives use automatic selection",
        )
        # Historical estimates are retained as conservative planning figures, not new timing claims.
        if canary:
            data["metadata"]["gpu_hours_estimate"] *= 0.5
        controls = {
            "E616": "E389",
            "E617": "E616",
            "E625": "E623",
            "E627": "E626",
            "E628": "E626",
            "E629": "E623",
            "E630": "E625",
            "E631": "E421",
            "E632": "E631",
            "E633": "E421",
        }
        if not canary and source in controls:
            control = f"E{796 + PERFORMANCE_SOURCES.index(controls[source])}"
            p["comparison_control_id"] = control
        rows.append((eid, wave, data))
    return rows
