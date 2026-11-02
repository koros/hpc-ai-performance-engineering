#!/usr/bin/env python3
"""Aggregate current study findings from cohort-aware plot and report exports.

Adapted from the dashboard3 prototype. Updates dashboard/studies-data.json and
its offline HTML embed. Run with --check to detect stale generated outputs.
"""

# Narrative chart captions retain readable sentences from the study specification.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scipy.special import stdtrit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DASH = ROOT / "dashboard"
PLOT = ROOT / "results" / "plot_data"
REPORT = ROOT / "results" / "report_data"
PLATFORMS = ("h100", "l40s")
GPU_NAME = {"h100": "H100", "l40s": "L40S"}

WORKLOAD_LABELS = {
    "gpt2_small": "GPT-2 Small",
    "gpt2_medium": "GPT-2 Medium",
    "gpt2_large": "GPT-2 Large",
    "distilbert": "DistilBERT",
    "tinyllama": "TinyLlama-1.1B",
    "environment_validation": "Environment validation",
    "qwen2_5_7b": "Qwen2.5-7B (serving)",
    "qwen2_5_7b_long_context": "Qwen2.5-7B (long-context)",
    "qwen2_5_7b_training": "Qwen2.5-7B (training)",
    "qwen2_5_7b_bf16_serving": "Qwen2.5-7B (BF16 serving)",
    "qwen2_5_3b": "Qwen2.5-3B",
}
PARAMS_M = {"distilbert": 66, "gpt2_small": 124, "gpt2_medium": 355, "gpt2_large": 774, "tinyllama": 1100}
LABEL_TO_KEY = {v: k for k, v in WORKLOAD_LABELS.items()}
SHORT = {
    "gpt2_small": "GPT-2",
    "distilbert": "DistilBERT",
    "tinyllama": "TinyLlama",
    "gpt2_medium": "GPT-2 M",
    "gpt2_large": "GPT-2 L",
}

DEEP_DIVE_PHASES = {
    "C": "Deep dive · acceptance canaries",
    "S1": "Deep dive S1 · Gradient accumulation",
    "S2": "Deep dive S2 · Batch × sequence split",
    "S3": "Deep dive S3 · Model size",
    "S4": "Deep dive S4 · DDP communication settings",
    "S5": "Deep dive S5 · Training stacks (DDP vs FSDP)",
    "S6": "Deep dive S6 · Placement, P2P and NCCL",
    "S7": "Deep dive S7 · Real text batching",
    "S8": "Deep dive S8 · Numeric precision",
    "RL": "Deep dive · run-length check",
}
THEMES = [
    (
        "environment",
        "Environment & Baselines",
        ["Environment validation", "Baseline profiling", "Workload characterisation", "Telemetry revalidation"],
    ),
    ("scaling", "Scaling Efficiency (RQ1)", ["Strong scaling", "Weak scaling"]),
    ("deepdive", "Scaling Deep Dive (S1–S8)", list(DEEP_DIVE_PHASES.values())),
    (
        "followups",
        "Training Follow-ups",
        ["Microbatch accumulation follow-up", "L40S topology recovery", "S7 fresh real-text acceptance"],
    ),
    ("strategy", "Distributed Strategy (RQ2)", ["Distributed training strategy"]),
    (
        "precision",
        "Precision & Quantization (RQ3)",
        ["Precision study", "Inference precision/quantization", "Low-bit quality robustness"],
    ),
    ("communication", "Communication & Data Movement (RQ4)", ["Communication analysis", "Data movement"]),
    ("power", "Power & Energy (RQ5)", ["Hardware energy revalidation", "Inference energy anchors"]),
    ("hardware", "Hardware Comparison (RQ6)", ["Hardware comparison"]),
    (
        "serving",
        "Inference Serving",
        [
            "Inference batching",
            "Inference concurrency",
            "Inference prompt length",
            "Inference framework",
            "Inference multi-GPU",
            "Inference long-context prompt length",
        ],
    ),
    (
        "layouts",
        "Inference Serving Layouts (TP vs replicas)",
        [
            "Inference TP prompt crossover",
            "Inference TP concurrency crossover",
            "Inference TP precision crossover",
            "Inference TP output crossover",
            "Inference layout acceptance",
            "Inference layout crossover",
            "Qwen7B layout acceptance",
            "Qwen7B layout follow-up",
        ],
    ),
    ("memory", "Memory & Capacity", ["Memory optimisation"]),
    (
        "largemodel",
        "Large Model · Qwen2.5-7B",
        ["Large-model distributed strategy", "Large-model training update controls", "Training update validation"],
    ),
]
CLUSTER_FIELDS = [
    "model",
    "mode",
    "strategy",
    "strategy_detail",
    "fsdp_wrap_policy",
    "scaling_type",
    "precision",
    "precision_backend",
    "gpus",
    "batch_size",
    "global_batch_size",
    "per_gpu_batch_size",
    "gradient_accumulation_steps",
    "sequence_length",
    "framework",
    "serving_layout",
    "tensor_parallel_size",
    "replica_count",
    "prompt_length",
    "concurrent_requests",
    "max_new_tokens",
    "quantization_method",
    "text_batching",
    "ddp_settings",
    "nccl_settings",
    "visible_devices",
    "activation_checkpointing",
    "pinned_memory",
    "num_workers",
    "prefetch_factor",
    "role",
    "comparability",
    "platform_list",
]


# ---------------------------------------------------------------- helpers
def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def num(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def r(value, digits=4):
    return None if value is None else round(value, digits)


def tcrit(dof: float) -> float:
    """Use the Student-t quantile without rounding degrees of freedom."""
    if not math.isfinite(dof) or dof <= 0:
        raise ValueError("Student-t degrees of freedom must be finite and positive")
    return float(stdtrit(dof, 0.975))


def ratio_ci(ma, sa, na, mb, sb, nb):
    """Delta-method ratio of means with t(na+nb-2), matching the report pipeline's ratio_ci()."""
    if None in (ma, mb) or not mb or not ma:
        return None, None, None
    ratio = ma / mb
    if None in (sa, sb, na, nb) or not na or not nb:
        return ratio, None, None
    se = ratio * math.sqrt(sa**2 / (na * ma**2) + sb**2 / (nb * mb**2))
    t = tcrit(max(na + nb - 2, 1))
    return ratio, ratio - t * se, ratio + t * se


def report_conditions(name, perf, platform):
    """Keep curated report columns but refresh intervals from verified trial exports."""
    rows = read_csv(REPORT / name)
    for row in rows:
        current = perf[(row["experiment_id"], platform)]
        for key in list(row):
            if "ci95" not in key:
                continue
            source = key.replace("throughput_ci95", "throughput_tokens_sec_ci95")
            if source in current:
                row[key] = current[source]
    return rows


def goodput_rows():
    """Recompute request-SLO goodput and its interval from individual trials."""
    rows = read_csv(REPORT / "qwen7b_goodput_by_latency_slo.csv")
    trials = defaultdict(list)
    for trial in read_csv(PLOT / "trial_level/current_performance_trials.csv"):
        if trial["platform"] == "l40s":
            trials[trial["experiment_id"]].append(trial)
    latencies = defaultdict(list)
    for sample in read_csv(PLOT / "request_distributions/inference_samples_long.csv"):
        if sample["platform"] == "l40s" and sample["metric"] == "latency_seconds":
            latencies[sample["run_id"]].append(float(sample["value"]))
    for row in rows:
        values = []
        for trial in trials[row["experiment_id"]]:
            samples = latencies[trial["run_id"]]
            if not samples:
                raise ValueError(f"Missing request latencies for {trial['run_id']}")
            values.append(sum(x <= float(row["slo_s"]) for x in samples) / float(trial["measured_seconds"]))
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if len(values) > 1 else None
        margin = tcrit(len(values) - 1) * sd / math.sqrt(len(values)) if sd is not None else None
        row.update(
            n=len(values),
            goodput_mean=mean,
            goodput_sd=sd,
            goodput_ci95_low=mean - margin if margin is not None else None,
            goodput_ci95_high=mean + margin if margin is not None else None,
        )
    return rows


def expand(ids: list[str]) -> list[str]:
    out = []
    for item in ids:
        m = re.fullmatch(r"E(\d+)-E(\d+)", item)
        out += [f"E{n:03d}" for n in range(int(m[1]), int(m[2]) + 1)] if m else [item]
    return out


def compress(ids) -> str:
    nums = sorted({int(i[1:]) for i in ids})
    if not nums:
        return ""
    parts, start, prev = [], nums[0], nums[0]
    for n in nums[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        parts.append(f"E{start:03d}" if start == prev else f"E{start:03d}–E{prev:03d}")
        if n is not None:
            start = prev = n
    return ", ".join(parts)


def sign_pct(x, d=1):
    return "n/a" if x is None else f"{x:+.{d}f}%"


# ---------------------------------------------------------------- sources
def load_sources():
    configs = {}
    for path in sorted(ROOT.glob("configs/*/E*.json")):
        c = json.loads(path.read_text())
        eid = c.get("experiment_id", "")
        if re.fullmatch(r"E\d{3}", eid):
            configs[eid] = (c, path.parent.name)
    matrix = json.loads((ROOT / "platforms" / "execution_matrix.json").read_text())
    routing = defaultdict(dict)  # id -> platform -> wave info
    for wave in matrix["waves"]:
        for eid in expand(wave["experiment_ids"]):
            routing[eid].setdefault(wave["platform"], wave)
    coverage = {
        (x["experiment_id"], x["platform"]): x
        for x in read_csv(PLOT / "matrix_coverage/platform_condition_coverage.csv")
    }
    perf = {
        (x["experiment_id"], x["platform"]): x
        for x in read_csv(PLOT / "trial_level/performance_condition_summary.csv")
        if x["selected_current_cohort"] == "true" and x["analysis_eligible"] == "true"
    }
    manifest = json.loads((PLOT / "manifest.json").read_text())
    return configs, matrix, routing, coverage, perf, manifest


# ---------------------------------------------------------------- Part 1 catalog
def phase_of(config) -> str:
    m = config["metadata"]
    phase = m.get("phase", "Unassigned")
    if phase == "Scaling deep dive":
        key = (m.get("scaling_studies") or ["C:canary"])[0].split(":")[0]
        return DEEP_DIVE_PHASES.get(key, phase)
    return phase


def experiment_record(eid, config, folder, routing, coverage, perf, capacity):
    w, m, rt = config["workload"], config["metadata"], config["runtime"]
    p = w.get("parameters", {})
    env = rt.get("environment") or {}
    ddp = (
        ",".join(
            f"{k.removeprefix('ddp_')}={p[k]}"
            for k in ("ddp_bucket_cap_mb", "ddp_comm_hook", "ddp_gradient_as_bucket_view", "ddp_static_graph")
            if k in p and not (k == "ddp_comm_hook" and p[k] == "none")
        )
        or None
    )
    nccl = (
        ",".join(f"{k}={v}" for k, v in sorted(env.items()) if k.startswith("NCCL_") and not k.startswith("NCCL_DEBUG"))
        or None
    )
    role = p.get("scaling_role") or (
        "canary" if "canary" in str(m.get("platform_wave", "")) or "canary" in folder else None
    )
    if role is None and re.search(r"canary", Path(str(config.get("experiment", ""))).name):
        role = "canary"
    platform_list = sorted(routing.get(eid, {}))
    rec = {
        "id": eid,
        "experiment": config.get("experiment"),
        "phase": phase_of(config),
        "workload": w["name"],
        "workload_label": WORKLOAD_LABELS.get(w["name"], w["name"]),
        "model": w.get("model"),
        "mode": config.get("mode"),
        "strategy": config["distributed"].get("strategy"),
        "strategy_detail": p.get("strategy_detail"),
        "fsdp_wrap_policy": p.get("fsdp_wrap_policy"),
        "scaling_type": p.get("scaling_type"),
        "precision": rt.get("precision"),
        "precision_backend": p.get("precision_backend"),
        "gpus": str(config["hardware"].get("gpus")),
        "batch_size": _s(p.get("batch_size")),
        "global_batch_size": _s(p.get("global_batch_size")),
        "per_gpu_batch_size": _s(p.get("per_gpu_batch_size")),
        "gradient_accumulation_steps": _s(p.get("gradient_accumulation_steps")),
        "sequence_length": _s(p.get("sequence_length")),
        "framework": p.get("framework"),
        "serving_layout": p.get("serving_layout"),
        "tensor_parallel_size": _s(p.get("tensor_parallel_size")),
        "replica_count": _s(p.get("replica_count")),
        "prompt_length": _s(p.get("prompt_length")),
        "concurrent_requests": _s(p.get("concurrent_requests")),
        "max_new_tokens": _s(p.get("max_new_tokens")),
        "quantization_method": p.get("quantization_method"),
        "text_batching": p.get("text_batching"),
        "ddp_settings": ddp,
        "nccl_settings": nccl,
        "visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "activation_checkpointing": _s(p.get("activation_checkpointing")),
        "pinned_memory": _s(p.get("pinned_memory")),
        "num_workers": _s(p.get("num_workers")),
        "prefetch_factor": _s(p.get("prefetch_factor")),
        "role": role,
        "priority": m.get("priority"),
        "gpu_hours_estimate": _s(m.get("gpu_hours_estimate")),
        "objective": m.get("objective"),
        "study_design_note": m.get("study_design_note"),
        "research_questions": m.get("research_questions") or [],
        "comparability": next((wv.get("comparability") for wv in routing.get(eid, {}).values()), None),
        "platform_list": platform_list,
        "platforms": {},
        "coverage": {},
    }
    completed_at = []
    for platform in PLATFORMS:
        cov = coverage.get((eid, platform))
        if cov:
            # Deferred rows have zero required trials, so the export marks them "complete" vacuously.
            status = "deferred" if cov["policy_status"] == "deferred" else cov["coverage_status"]
            rec["coverage"][platform] = {
                "status": status,
                "policy": cov["policy_status"],
                "completed_trials": int(cov["completed_trials"] or 0),
                "required_trials": int(cov["required_trials"] or 0),
                "failed_trials": int(cov["failed_trials"] or 0),
            }
        row = perf.get((eid, platform))
        if row:
            rec["platforms"][platform] = platform_metrics(row)
            completed_at.append(row.get("source_commit") or "")
    rec["status"] = overall_status(rec, capacity.get(eid, {}))
    # Drop empty fields: the pages treat a missing field as "—", and the catalog is embedded inline.
    return {k: v for k, v in rec.items() if v not in (None, [], {}, "") or k in ("platforms", "platform_list")}


def _s(value):
    return None if value is None else str(value)


def platform_metrics(row) -> dict:
    out = {
        "n": int(row["trial_count"] or 0),
        "gpu_model": row.get("gpu_model"),
        "throughput_unit": row.get("throughput_unit"),
        "runtime_cohort_id": (row.get("runtime_cohort_id") or "")[:12],
    }
    mapping = {
        "throughput_tokens_sec": "throughput_tokens_sec",
        "throughput_samples_sec": "throughput_samples_sec",
        "memory_used_gb": "peak_memory_gib",
        "energy_joules": "energy_j_measured_region",
        "latency_p50_ms": "latency_p50_ms",
        "latency_p95_ms": "latency_p95_ms",
        "time_to_first_token_p95_ms": "ttft_p95_ms",
        "time_per_output_token_p95_ms": "time_per_output_token_p95_ms",
        "joules_per_output_token": "joules_per_output_token",
        "joules_per_processed_token": "joules_per_processed_token",
        "gpu_utilization": "gpu_utilization_pct",
        "runtime_seconds": "measured_seconds",
    }
    for key, source in mapping.items():
        value = num(row.get(f"{source}_mean"))
        if value is not None:
            out[key] = r(value, 6)
    lo, hi = num(row.get("throughput_tokens_sec_ci95_low")), num(row.get("throughput_tokens_sec_ci95_high"))
    if lo is not None and hi is not None:
        out["throughput_ci95"] = [r(lo, 3), r(hi, 3)]
    if row.get("energy_status") and row["energy_status"] != "sufficient":
        for key in ("energy_joules", "joules_per_output_token", "joules_per_processed_token"):
            out.pop(key, None)
        out["energy_status"] = row["energy_status"]
    return out


def overall_status(rec, history) -> str:
    covs = rec["coverage"]
    if any(c["status"] == "complete" and c["completed_trials"] > 0 for c in covs.values()):
        return "Completed"
    if any(c["status"] == "partial" for c in covs.values()):
        return "Partial"
    deferred = [p for p, c in covs.items() if c["policy"] == "deferred"] or [
        p for p, h in history.items() if h["state"] == "deferred"
    ]
    if covs and len(deferred) == len(covs):
        return "Deferred"
    return "Configured"


def capacity_history(matrix) -> dict:
    history: dict = defaultdict(dict)
    for entry in matrix.get("deferred", []):
        for eid in expand(entry["experiment_ids"]):
            for platform in entry["platforms"]:
                history[eid][platform] = {
                    "state": "deferred",
                    "measuredOom": entry.get("observed_failure") == "cuda_oom",
                    "note": entry.get("reason", ""),
                }
    for entry in matrix.get("recovered_failures", []):
        history[entry["experiment_id"]][entry["platform"]] = {
            "state": "recovered",
            "measuredOom": entry.get("observed_failure") == "cuda_oom",
            "note": entry.get("reason", ""),
        }
    return dict(history)


def build_catalog(configs, routing, coverage, perf, capacity, manifest):
    experiments = {
        eid: experiment_record(eid, c, folder, routing, coverage, perf, capacity)
        for eid, (c, folder) in sorted(configs.items())
    }
    phase_theme = {ph: (tid, label) for tid, label, phases in THEMES for ph in phases}
    unknown = sorted({e["phase"] for e in experiments.values()} - set(phase_theme))
    if unknown:
        raise SystemExit(f"Unmapped phases: {unknown}")
    themes = []
    for tid, label, phases in THEMES:
        theme_phases = []
        for phase in phases:
            ids = [eid for eid, e in experiments.items() if e["phase"] == phase]
            if not ids:
                continue
            groups = defaultdict(list)
            for eid in ids:
                groups[experiments[eid]["workload_label"]].append(eid)
            clusters = []
            for wl, cids in sorted(groups.items(), key=lambda kv: kv[1][0]):
                constant, varied = {}, {}
                for field in CLUSTER_FIELDS:
                    vals = [experiments[i].get(field) for i in cids]
                    keyed = sorted({json.dumps(v) for v in vals if v not in (None, [], "")})
                    if len(keyed) == 1 and all(v not in (None, [], "") for v in vals):
                        constant[field] = json.loads(keyed[0])
                    elif len(keyed) > 1:
                        varied[field] = [json.loads(k) for k in keyed]
                constant["workload_label"] = wl
                clusters.append(
                    {
                        "key": f"{phase}:{wl}",
                        "label": wl,
                        "totals": totals(cids, experiments),
                        "constant": constant,
                        "varied": varied,
                        "experiment_ids": sorted(cids),
                    }
                )
            theme_phases.append(
                {
                    "id": re.sub(r"[^a-z0-9]+", "-", phase.lower()).strip("-"),
                    "label": phase,
                    "totals": totals(ids, experiments),
                    "clusters": clusters,
                }
            )
        for eid in (i for ph in theme_phases for cl in ph["clusters"] for i in cl["experiment_ids"]):
            experiments[eid]["theme_id"], experiments[eid]["theme_label"] = tid, label
        all_ids = [i for ph in theme_phases for cl in ph["clusters"] for i in cl["experiment_ids"]]
        themes.append({"id": tid, "label": label, "totals": totals(all_ids, experiments), "phases": theme_phases})
    status = Counter(e["status"] for e in experiments.values())
    opts = {
        "theme": [{"id": t["id"], "label": t["label"]} for t in themes],
        "workload": sorted({(e["workload"], e["workload_label"]) for e in experiments.values()}),
        "platform": list(PLATFORMS),
        "precision": sorted({e["precision"] for e in experiments.values() if e["precision"]}),
        "strategy": sorted({e["strategy"] for e in experiments.values() if e["strategy"]}),
        "status": ["Completed", "Partial", "Configured", "Deferred"],
        "gpus": sorted({int(e["gpus"]) for e in experiments.values() if e["gpus"] not in (None, "None")}),
        "mode": sorted({e["mode"] for e in experiments.values() if e["mode"]}),
    }
    meta = {
        "generated_at": manifest["latest_completed_at"][:10],
        "evidence_latest_completed_at": manifest["latest_completed_at"],
        "cohort_policy": manifest["cohort_policy"],
        "total": len(experiments),
        **{k.lower(): status.get(k, 0) for k in ("Completed", "Partial", "Configured", "Deferred")},
        "platform_conditions": {
            p: dict(Counter(e["coverage"].get(p, {}).get("status", "not_routed") for e in experiments.values()))
            for p in PLATFORMS
        },
    }
    return {"meta": meta, "themes": themes, "experiments": experiments, "filter_options": opts}


def totals(ids, experiments):
    c = Counter(experiments[i]["status"] for i in ids)
    return {
        "completed": c["Completed"],
        "partial": c["Partial"],
        "configured": c["Configured"],
        "deferred": c["Deferred"],
        "total": len(ids),
    }


# ---------------------------------------------------------------- Part 2 drawer refresh
# ---------------------------------------------------------------- Part 3 studies
class Study:
    def __init__(self, sid, group, title, question, plain, ids, design, sources):
        self.d = {
            "id": sid,
            "group": group,
            "title": title,
            "question": question,
            "plain": plain,
            "ids": sorted(set(ids)),
            "design": design,
            "sources": sources,
            "findings": [],
            "caveats": [],
            "charts": [],
            "tables": [],
        }

    def finding(self, text):
        self.d["findings"].append(text)

    def caveat(self, text):
        self.d["caveats"].append(text)

    def chart(self, **chart):
        self.d["charts"].append(chart)

    def table(self, title, caption, columns, rows):
        self.d["tables"].append({"title": title, "caption": caption, "columns": columns, "rows": rows})


def col(key, label, fmt="text", align=None):
    return {"key": key, "label": label, "fmt": fmt, "align": align or ("left" if fmt in ("text", "id") else "right")}


def coverage_summary(ids, coverage):
    out = {}
    for platform in PLATFORMS:
        rows = [coverage[(i, platform)] for i in ids if (i, platform) in coverage]
        if not rows:
            continue
        c = Counter("deferred" if x["policy_status"] == "deferred" else x["coverage_status"] for x in rows)
        out[platform] = {
            "planned": len(rows),
            "complete": c["complete"],
            "partial": c["partial"],
            "missing": c["missing"],
            "deferred": c["deferred"],
            "trials": sum(int(x["completed_trials"] or 0) for x in rows),
        }
    return out


def effects_by(lever):
    rows = read_csv(PLOT / "training_parameter_effects/all_controlled_effects.csv")
    return [x for x in rows if x["lever"] == lever]


def deep_conditions():
    return {
        (x["platform_id"], x["experiment_id"]): x
        for x in read_csv(PLOT / "training_scaling/deep_dive_condition_metrics.csv")
    }


SETTING_LABELS = {
    "bucket_cap_mb=5": "5 MB buckets",
    "bucket_cap_mb=100": "100 MB buckets",
    "comm_hook=bf16_compress": "BF16 compression",
    "bucket_cap_mb=100,comm_hook=bf16_compress": "100 MB + BF16 compression",
    "gradient_as_bucket_view=True,static_graph=True": "bucket view + static graph",
    "default": "default DDP",
    "ddp": "DDP",
    "fsdp:sharded_grad_op:transformer_layer": "FSDP grad-op shard (per-layer)",
    "fsdp:full_shard:transformer_layer": "FSDP full shard (per-layer)",
    "fsdp:full_shard:root": "FSDP full shard (root wrap)",
    "fp32_params_bf16_autocast": "BF16 autocast",
    "fp32_params_fp32": "FP32",
    "transformer_engine_bf16": "TE BF16",
    "transformer_engine_fp8": "TE FP8",
    "wikitext103:pad_to_max": "pad to max",
    "wikitext103:dynamic_pad": "dynamic pad",
    "wikitext103:length_bucketed": "length-bucketed",
    "wikitext103:pack": "pack",
}


def pretty(setting: str) -> str:
    if setting in SETTING_LABELS:
        return SETTING_LABELS[setting]
    if re.fullmatch(r"\d+", setting):
        return f"accum {setting}"
    env = dict(part.split("=", 1) for part in setting.split(",") if "=" in part) if "=" in setting else {}
    if env:
        bits = []
        if env.get("CUDA_VISIBLE_DEVICES"):
            bits.append(f"GPUs {setting.split('CUDA_VISIBLE_DEVICES=')[1].split(',NCCL')[0]}")
        if env.get("NCCL_ALGO"):
            bits.append(f"NCCL {env['NCCL_ALGO']}")
        if env.get("NCCL_P2P_DISABLE"):
            bits.append("P2P off")
        return ", ".join(bits) or setting
    return setting.replace("x", "×") if re.fullmatch(r"\d+x\d+", setting) else setting


def eff_row(e, conds, extra=None):
    c = conds.get((e["platform_id"], e["control_experiment_id"]), {})
    t = conds.get((e["platform_id"], e["treatment_experiment_id"]), {})
    row = {
        "platform": GPU_NAME[e["platform_id"]],
        "workload": WORKLOAD_LABELS.get(e["workload"], e["workload"]),
        "gpus": int(e["gpus"]),
        "shape": f"{c.get('per_gpu_batch_size', '?')}×{c.get('sequence_length', '?')}"
        + (
            f", accum {c.get('gradient_accumulation_steps')}"
            if c.get("gradient_accumulation_steps") not in (None, "", "1")
            else ""
        ),
        "control": pretty(e["control_setting"]),
        "treatment": pretty(e["treatment_setting"]),
        "control_raw": e["control_setting"],
        "treatment_raw": e["treatment_setting"],
        "change": r(num(e["throughput_change_pct"]), 2),
        "lo": r(num(e["change_ci95_low"]), 2),
        "hi": r(num(e["change_ci95_high"]), 2),
        "eff_points": r(_pts(e["efficiency_change_points"]), 1),
        "verdict": e["verdict"],
        "ids": f"{e['control_experiment_id']} → {e['treatment_experiment_id']}",
        "control_id": e["control_experiment_id"],
        "treatment_id": e["treatment_experiment_id"],
        "control_tps": r(num(c.get("mean_throughput_tokens_sec")), 1),
        "treatment_tps": r(num(t.get("mean_throughput_tokens_sec")), 1),
        "treatment_mem": r(num(t.get("peak_memory_gb")), 2),
        "control_mem": r(num(c.get("peak_memory_gb")), 2),
    }
    row.update(extra or {})
    return row


def _pts(value):
    """The export already reports efficiency changes in percentage points."""
    return num(value)


EFFECT_COLUMNS = [
    col("platform", "Platform"),
    col("workload", "Workload"),
    col("shape", "Shape"),
    col("gpus", "GPUs", "int"),
    col("control", "Control"),
    col("treatment", "Treatment"),
    col("change", "Throughput change", "signed1"),
    col("lo", "95% CI low", "signed1"),
    col("hi", "95% CI high", "signed1"),
    col("eff_points", "Δ efficiency (pts)", "signed1"),
    col("verdict", "Verdict", "verdict"),
    col("ids", "Experiments", "id"),
]


def forest(rows, label_fn, title, caption, group_fn=None):
    return {
        "type": "forest",
        "title": title,
        "caption": caption,
        "x": {"label": "Throughput change vs control (%)"},
        "rows": [
            {
                "label": label_fn(x),
                "group": group_fn(x) if group_fn else None,
                "y": x["change"],
                "lo": x["lo"],
                "hi": x["hi"],
                "verdict": x["verdict"],
                "ids": x["ids"],
            }
            for x in rows
        ],
    }


def max_gpu_rows(rows):
    best = {}
    for x in rows:
        key = (x["platform"], x["workload"], x["shape"], x["treatment"])
        if key not in best or x["gpus"] > best[key]["gpus"]:
            best[key] = x
    return sorted(best.values(), key=lambda x: (x["platform"], x["workload"], x["treatment"]))


def build_studies(coverage, perf, configs):
    conds = deep_conditions()
    studies = []

    def eff_curve_panel(platform, workload, shape, series_field, rows, label):
        series = defaultdict(list)
        for x in rows:
            if x["platform_id"] != platform or x["workload"] != workload:
                continue
            if f"{x['per_gpu_batch_size']}x{x['sequence_length']}" != shape:
                continue
            e = num(x["scaling_efficiency"])
            if e is None:
                continue
            series[x[series_field]].append(
                {
                    "x": int(x["gpus"]),
                    "y": r(e, 4),
                    "lo": r(num(x["efficiency_ci95_low"]), 4),
                    "hi": r(num(x["efficiency_ci95_high"]), 4),
                    "id": x["experiment_id"],
                }
            )
        return {
            "title": f"{GPU_NAME[platform]} · {WORKLOAD_LABELS[workload]} ({shape.replace('x', '×')})",
            "series": [
                {"name": label(k), "points": sorted(v, key=lambda p: p["x"])}
                for k, v in sorted(
                    series.items(), key=lambda kv: float(kv[0]) if re.fullmatch(r"[\d.]+", kv[0]) else kv[0]
                )
            ],
        }

    # ---- S1 gradient accumulation
    acc = read_csv(PLOT / "training_parameter_studies/gradient_accumulation_conditions.csv")
    s = Study(
        "s1-accumulation",
        "Training · scaling deep dive",
        "S1 · Gradient accumulation",
        "If each GPU does several micro-batches before the gradients are exchanged, does scaling efficiency recover?",
        "Every training step ends with all GPUs swapping their gradients. Accumulation lets each GPU do 2, 4 or 8 "
        "small steps before that swap, so the swap is paid less often. If a slow link is the bottleneck, this should help most where the link is slowest.",
        [x["experiment_id"] for x in acc],
        [
            "Fixed micro-batch per workload: GPT-2 8×1024, DistilBERT 32×512, TinyLlama 4×1024 (sequences × tokens).",
            "Accumulation 1/2/4/8 at 1/2/4/8 GPUs on L40S; accumulation 1 and 4 at 1/2/4 GPUs on H100 (compact block).",
            "Efficiency = throughput at N GPUs ÷ (N × throughput of the matched 1-GPU run, same accumulation, same runtime cohort).",
        ],
        [
            "results/plot_data/training_parameter_studies/gradient_accumulation_conditions.csv",
            "results/plot_data/training_parameter_effects/gradient_accumulation_effects.csv",
        ],
    )
    panels = []
    for platform in PLATFORMS:
        for wl, shape in (("gpt2_small", "8x1024"), ("distilbert", "32x512"), ("tinyllama", "4x1024")):
            panel = eff_curve_panel(platform, wl, shape, "gradient_accumulation_steps", acc, lambda k: f"accum {k}")
            if any(len(sr["points"]) > 1 for sr in panel["series"]):
                panels.append(panel)
    s.chart(
        type="line",
        title="Scaling efficiency by GPU count and accumulation",
        caption="Points are condition means of 3 trials; whiskers are 95% CIs from the deep-dive export. The dashed line is perfect (100%) scaling.",
        x={"label": "GPUs", "ticks": [1, 2, 4, 8]},
        y={"label": "Scaling efficiency", "fmt": "pct0", "ref": 1.0},
        panels=panels,
    )
    er = [eff_row(e, conds) for e in effects_by("gradient_accumulation")]
    s1_rows = [x for x in er if x["shape"].split(",")[0] in ("8×1024", "32×512", "4×1024")]
    s.table(
        "Accumulation versus accumulation 1, same GPU count",
        "Throughput change of each accumulation setting against accumulation 1 at the same GPU count and micro-batch. Verdict: helps/hurts needs ≥3% and a 95% CI excluding zero.",
        EFFECT_COLUMNS,
        sorted(s1_rows, key=lambda x: (x["platform"], x["workload"], x["treatment"], x["gpus"])),
    )
    for platform in PLATFORMS:
        for wl in ("tinyllama", "gpt2_small", "distilbert"):
            pts = {
                (int(x["gradient_accumulation_steps"]), int(x["gpus"])): num(x["scaling_efficiency"])
                for x in acc
                if x["platform_id"] == platform and x["workload"] == wl
            }
            gmax = max((g for (_, g) in pts), default=None)
            amax = max((a for (a, g) in pts if g == gmax), default=None)
            if gmax and amax and amax > 1 and pts.get((1, gmax)) and pts.get((amax, gmax)):
                gain = [
                    x
                    for x in s1_rows
                    if x["platform"] == GPU_NAME[platform]
                    and x["workload"] == WORKLOAD_LABELS[wl]
                    and x["gpus"] == gmax
                    and x["treatment_raw"] == str(amax)
                ]
                s.finding(
                    f"{GPU_NAME[platform]} {WORKLOAD_LABELS[wl]} at {gmax} GPUs: efficiency {pts[(1, gmax)]:.0%} with no accumulation → "
                    f"{pts[(amax, gmax)]:.0%} with accumulation {amax}"
                    + (
                        f" (throughput {sign_pct(gain[0]['change'])}, CI {sign_pct(gain[0]['lo'])} to {sign_pct(gain[0]['hi'])})."
                        if gain
                        else "."
                    )
                )
    verdicts = Counter(x["verdict"] for x in s1_rows)
    s.finding(
        f"Across all {len(s1_rows)} matched contrasts accumulation never hurt throughput ({verdicts['helps']} helps, "
        f"{verdicts['inconclusive']} inconclusive, {verdicts['hurts']} hurts). The gain grows with GPU count and model size, "
        "which is what a gradient-exchange bottleneck predicts."
    )
    s.caveat(
        "Accumulation changes the global batch, so these are system-throughput results, not time-to-accuracy results."
    )
    s.caveat(
        "H100 ran a compact block (accumulation 1 and 4 only, up to 4 GPUs); compare platforms at matching settings only."
    )
    # Efficiency against work per optimizer step (L40S)
    work = read_csv(REPORT / "l40s_efficiency_vs_work_per_step.csv")
    series = defaultdict(list)
    for x in work:
        if x["gpus"] == "8" and num(x["scaling_efficiency"]) is not None:
            series[WORKLOAD_LABELS.get(x["workload"], x["workload"])].append(
                {
                    "x": num(x["tokens_per_step"]),
                    "y": r(num(x["scaling_efficiency"]), 4),
                    "lo": r(num(x["efficiency_ci95_low"]), 4),
                    "hi": r(num(x["efficiency_ci95_high"]), 4),
                    "id": x["experiment_id"],
                }
            )
    s.chart(
        type="scatter",
        shared=True,
        title="L40S 8-GPU efficiency against tokens processed per GPU per optimizer step",
        caption="Every L40S 8-GPU deep-dive condition with a matched 1-GPU baseline (accumulation, split and model-size studies). "
        "Work per exchange explains most of the spread: the more tokens each GPU processes before a gradient exchange, the closer to 100%.",
        x={"label": "Tokens per GPU per optimizer step (log scale)", "log": True},
        y={"label": "Scaling efficiency at 8 GPUs", "fmt": "pct0", "ref": 1.0},
        # One panel per workload (shared axes): a scatter keeps each panel to a single series.
        panels=[
            {"title": k, "series": [{"name": k, "points": sorted(v, key=lambda p: p["x"])}]}
            for k, v in sorted(series.items(), key=lambda kv: PARAMS_M.get(LABEL_TO_KEY.get(kv[0], ""), 0))
        ],
    )
    s.d["coverage"] = coverage_summary(s.d["ids"], coverage)
    studies.append(s.d)

    # ---- S2 batch x sequence split
    shapes = read_csv(PLOT / "training_parameter_studies/batch_sequence_shapes_conditions.csv")
    split = [
        eff_row(e, conds) for e in effects_by("batch_sequence_split") if e["control_setting"] != e["treatment_setting"]
    ]
    s = Study(
        "s2-split",
        "Training · scaling deep dive",
        "S2 · Batch × sequence split",
        "At the same number of tokens per GPU, does it matter whether they arrive as many short sequences or a few long ones?",
        "Each GPU processes the same amount of text per step, but split differently: 32 sequences of 256 tokens versus 8 of 1,024, for example. "
        "Attention cost grows faster than linearly with sequence length, so the split could change both speed and scaling.",
        [x["experiment_id"] for x in shapes],
        [
            "Two token budgets per workload (GPT-2 8,192 and 2,048; DistilBERT 8,192 and 2,048; TinyLlama 4,096 and 2,048 tokens per GPU per micro-step).",
            "Three splits per budget, 1/2/4/8 GPUs. The control is the longest-sequence split of each budget.",
            "L40S has the full grid; H100 has only shared controls so far.",
        ],
        [
            "results/plot_data/training_parameter_studies/batch_sequence_shapes_conditions.csv",
            "results/plot_data/training_parameter_effects/batch_sequence_split_effects.csv",
        ],
    )
    split_sorted = sorted(split, key=lambda x: (x["platform"], x["workload"], x["shape"], x["treatment"], x["gpus"]))
    s.chart(
        **forest(
            [x for x in split_sorted if x["gpus"] in (1, 8)],
            lambda x: f"{SHORT_BY_LABEL(x['workload'])} {x['treatment']} vs {x['control']} · {x['gpus']} GPU",
            "Shorter sequences at the same tokens per GPU (L40S, 1 and 8 GPUs)",
            "Each row compares a shorter-sequence split with the longest-sequence split of the same token budget at the same GPU count. "
            "Bars are 95% CIs; the shaded band is the ±3% materiality threshold.",
            group_fn=lambda x: x["workload"],
        )
    )
    s.table(
        "Every split contrast",
        "Throughput change of each split against the longest-sequence split of the same token budget, same GPU count.",
        EFFECT_COLUMNS,
        split_sorted,
    )
    gp = [x for x in split if x["workload"] == "GPT-2 Small" and x["gpus"] == 1]
    s.finding(
        "GPT-2 gains most from shorter sequences: "
        + ", ".join(f"{x['treatment']} {sign_pct(x['change'])}" for x in sorted(gp, key=lambda x: -x["change"])[:3])
        + " at 1 GPU, shrinking at 8 GPUs ("
        + ", ".join(
            f"{x['treatment']} {sign_pct(x['change'])}"
            for x in split
            if x["workload"] == "GPT-2 Small" and x["gpus"] == 8
        )
        + ")."
    )
    db = [x for x in split if x["workload"] == "DistilBERT"]
    s.finding(
        f"DistilBERT is insensitive to the split: all {len(db)} contrasts are within ±{max(abs(x['change']) for x in db):.1f}%."
    )
    tl = [x for x in split if x["workload"] == "TinyLlama-1.1B" and x["gpus"] == 8]
    s.finding(
        "TinyLlama's single-GPU benefit from shorter sequences disappears at 8 GPUs ("
        + ", ".join(f"{x['treatment']} {sign_pct(x['change'])}" for x in tl)
        + "): once communication dominates, how the tokens are shaped stops mattering."
    )
    s.finding(
        "Rule of thumb supported: tokens per GPU per step, not the batch/sequence split, is what governs scaling efficiency."
    )
    s.caveat(
        "Two H100 contrasts in the export compare 8×1024 with 8×1024. They are the run-length check (1,000 vs 200 updates), not a split, and are excluded here (see Run length)."
    )
    s.d["coverage"] = coverage_summary(s.d["ids"], coverage)
    studies.append(s.d)

    # ---- S3 model size
    msz = read_csv(PLOT / "training_parameter_studies/model_size_conditions.csv")
    s = Study(
        "s3-model-size",
        "Training · scaling deep dive",
        "S3 · Model size and gradient volume",
        "At the same work per GPU, how does scaling efficiency change as the model gets bigger?",
        "A bigger model has more gradients to exchange after every step, but each GPU's compute per step also grows. "
        "This study holds the work per GPU fixed (4 sequences × 512 tokens) and changes only the model, from 66M to 1.1B parameters.",
        [x["experiment_id"] for x in msz],
        [
            "Models: DistilBERT 66M, GPT-2 Small 124M, GPT-2 Medium 355M, GPT-2 Large 774M, TinyLlama 1.1B.",
            "4×512 per GPU, accumulation 1 and 4, 1/2/4/8 GPUs (L40S). H100 has not run this block yet.",
        ],
        ["results/plot_data/training_parameter_studies/model_size_conditions.csv"],
    )
    order = ["distilbert", "gpt2_small", "gpt2_medium", "gpt2_large", "tinyllama"]
    rows, groups = [], []
    pool = msz + [
        y
        for y in conds.values()
        if y["per_gpu_batch_size"] == "4"
        and y["sequence_length"] == "512"
        and y["stack"] == "ddp"
        and y["ddp_settings"] == "default"
        and y["input_contract"] == "synthetic"
        and y["runtime_environment"] == "default"
        and y["precision_contract"] == "fp32_params_bf16_autocast"
    ]
    for wl in order:
        bars = []
        for acc_steps in ("1", "4"):
            x = next(
                (
                    y
                    for y in pool
                    if y["platform_id"] == "l40s"
                    and y["workload"] == wl
                    and y["gpus"] == "8"
                    and y["gradient_accumulation_steps"] == acc_steps
                ),
                None,
            )
            if not x:
                continue
            e = num(x["scaling_efficiency"])
            bars.append(
                {
                    "name": f"accum {acc_steps}",
                    "y": r(e, 4),
                    "lo": r(num(x["efficiency_ci95_low"]), 4),
                    "hi": r(num(x["efficiency_ci95_high"]), 4),
                    "id": x["experiment_id"],
                }
            )
            rows.append(
                {
                    "workload": WORKLOAD_LABELS[wl],
                    "params": PARAMS_M[wl],
                    "accum": int(acc_steps),
                    "gpus": 8,
                    "tps": r(num(x["mean_throughput_tokens_sec"]), 0),
                    "eff": r(e, 4),
                    "elo": r(num(x["efficiency_ci95_low"]), 4),
                    "ehi": r(num(x["efficiency_ci95_high"]), 4),
                    "comm": r(num(x["communication_fraction"]), 3),
                    "mem": r(num(x["peak_memory_gb"]), 1),
                    "id": x["experiment_id"],
                }
            )
        groups.append({"label": f"{WORKLOAD_LABELS[wl]} ({PARAMS_M[wl]}M)", "bars": bars})
    s.chart(
        type="bar",
        title="L40S scaling efficiency at 8 GPUs by model size",
        caption="4×512 tokens per GPU per micro-step. Whiskers are 95% CIs.",
        y={"label": "Scaling efficiency at 8 GPUs", "fmt": "pct0", "ref": 1.0},
        groups=groups,
    )
    s.table(
        "Model-size ladder at 8 L40S GPUs",
        "Efficiency against the matched 1-GPU run; communication fraction = exposed communication time ÷ step time.",
        [
            col("workload", "Model"),
            col("params", "Params (M)", "int"),
            col("accum", "Accum", "int"),
            col("tps", "Tokens/s", "int"),
            col("eff", "Efficiency", "pct1"),
            col("elo", "CI low", "pct1"),
            col("ehi", "CI high", "pct1"),
            col("comm", "Comm. fraction", "pct0"),
            col("mem", "Peak GiB", "num1"),
            col("id", "Experiment", "id"),
        ],
        rows,
    )
    a1 = {x["workload"]: x for x in rows if x["accum"] == 1}
    a4 = {x["workload"]: x for x in rows if x["accum"] == 4}
    s.finding(
        "Without accumulation, 8-GPU L40S efficiency: "
        + ", ".join(f"{k} {v['eff']:.0%}" for k, v in a1.items())
        + ". Within the GPT-2 family it falls as the model grows; DistilBERT and TinyLlama are lowest, so size alone does not predict it — "
        "work per step relative to gradient volume does (DistilBERT does very little compute per step at this shape)."
    )
    s.finding(
        "Accumulation 4 at the same micro-batch: " + ", ".join(f"{k} {v['eff']:.0%}" for k, v in a4.items()) + "."
    )
    gains = {k: a4[k]["eff"] - a1[k]["eff"] for k in a1 if k in a4}
    s.finding(
        "Efficiency gained from accumulation 4: "
        + ", ".join(f"{k} {v * 100:+.0f} pts" for k, v in gains.items())
        + " — a similar gain for every model at this small shape."
    )
    s.caveat(
        "4×512 is small work per GPU by design, so absolute efficiencies here are lower than in S1's larger shapes."
    )
    s.d["coverage"] = coverage_summary(s.d["ids"], coverage)
    studies.append(s.d)

    # ---- S4 DDP communication settings
    ddp = [eff_row(e, conds) for e in effects_by("ddp_communication")]
    ddp_ids = [i for x in ddp for i in (x["control_id"], x["treatment_id"])]
    s = Study(
        "s4-ddp",
        "Training · scaling deep dive",
        "S4 · DDP communication settings",
        "Which DDP settings recover efficiency on PCIe, and do they matter on NVLink?",
        "PyTorch DDP sends gradients in 'buckets' while the backward pass is still running. Bucket size changes how well "
        "sending overlaps computing; BF16 compression halves the bytes sent. These are one-line settings, so a clear win would be cheap to adopt.",
        ddp_ids,
        [
            "Settings: 5 MB and 100 MB buckets (default 25 MB), BF16 gradient compression, compression + 100 MB, bucket view + static graph.",
            "Primary shapes (GPT-2 8×1024, DistilBERT 32×512, TinyLlama 4×1024), accumulation 1, 2/4/8 GPUs.",
            "Control: default DDP at the same GPU count.",
        ],
        [
            "results/plot_data/training_parameter_effects/ddp_communication_effects.csv",
            "results/plot_data/training_parameter_studies/ddp_communication_conditions.csv",
        ],
    )
    top = max_gpu_rows(ddp)
    s.chart(
        **forest(
            top,
            lambda x: f"{x['platform']} {SHORT_BY_LABEL(x['workload'])} · {x['treatment']} · {x['gpus']} GPU",
            "DDP settings at the largest GPU count (8 L40S, 4 H100)",
            "Throughput change against default DDP at the same GPU count. Shaded band: ±3%.",
            group_fn=lambda x: x["platform"],
        )
    )
    s.table(
        "Every DDP-setting contrast",
        "Against default DDP (25 MB buckets, no compression) at the same GPU count and shape.",
        EFFECT_COLUMNS,
        sorted(ddp, key=lambda x: (x["platform"], x["workload"], x["treatment"], x["gpus"])),
    )
    for platform, wl in (
        ("L40S", "TinyLlama-1.1B"),
        ("H100", "TinyLlama-1.1B"),
        ("L40S", "GPT-2 Small"),
        ("L40S", "DistilBERT"),
    ):
        rr = [x for x in top if x["platform"] == platform and x["workload"] == wl]
        if rr:
            best = max(rr, key=lambda x: x["change"])
            worst = min(rr, key=lambda x: x["change"])
            s.finding(
                f"{platform} {wl} at {best['gpus']} GPUs: best setting {best['treatment']} {sign_pct(best['change'])} ({best['verdict'].replace('_', ' ')}); "
                f"worst {worst['treatment']} {sign_pct(worst['change'])} ({worst['verdict'].replace('_', ' ')})."
            )
    s.finding(
        "Pattern: compression pays where the link is the bottleneck (large model on PCIe) and costs a little where it isn't (NVLink); "
        "bucket size and static graph are second-order."
    )
    s.caveat("H100 has TinyLlama only (2 and 4 GPUs).")
    s.d["coverage"] = coverage_summary(s.d["ids"], coverage)
    studies.append(s.d)

    # ---- S5 stacks
    st = [eff_row(e, conds) for e in effects_by("training_stack")]
    s = Study(
        "s5-stacks",
        "Training · scaling deep dive",
        "S5 · Training stacks: DDP versus FSDP",
        "Does sharding the model (FSDP) scale better or worse than replicating it (DDP), with identical inputs?",
        "DDP keeps a full copy of the model on every GPU and exchanges gradients. FSDP splits the model's state across GPUs, "
        "which changes what is sent and when: more, smaller messages that can overlap computation. Wrapping each layer separately ('per-layer') or the whole model ('root') changes that overlap again.",
        [i for x in st for i in (x["control_id"], x["treatment_id"])],
        [
            "FSDP sharded_grad_op (per-layer wrap), full_shard (per-layer wrap), full_shard (root wrap) versus DDP.",
            "Primary shapes, accumulation 1, 2/4/8 GPUs. Plus TinyLlama 8×1024 under FSDP, a shape DDP cannot fit on L40S.",
        ],
        [
            "results/plot_data/training_parameter_effects/training_stack_effects.csv",
            "results/plot_data/training_parameter_studies/distributed_strategies_conditions.csv",
        ],
    )
    top = max_gpu_rows(st)
    s.chart(
        **forest(
            top,
            lambda x: f"{x['platform']} {SHORT_BY_LABEL(x['workload'])} · {x['treatment']} · {x['gpus']} GPU",
            "FSDP against DDP at the largest GPU count",
            "Throughput change against DDP at the same GPU count and shape. Shaded band: ±3%.",
            group_fn=lambda x: x["platform"],
        )
    )
    s.table(
        "Every stack contrast",
        "Against DDP at the same GPU count and shape; memory columns are peak GiB per GPU.",
        EFFECT_COLUMNS + [col("control_mem", "DDP GiB", "num1"), col("treatment_mem", "FSDP GiB", "num1")],
        sorted(st, key=lambda x: (x["platform"], x["workload"], x["treatment"], x["gpus"])),
    )
    for platform in ("L40S", "H100"):
        rr = [x for x in top if x["platform"] == platform and x["workload"] == "TinyLlama-1.1B"]
        if rr:
            best = max(rr, key=lambda x: x["change"])
            s.finding(
                f"{platform} TinyLlama at {best['gpus']} GPUs: {best['treatment']} {sign_pct(best['change'])} over DDP "
                f"(CI {sign_pct(best['lo'])} to {sign_pct(best['hi'])})."
            )
    rootd = [
        x
        for x in st
        if x["workload"] == "DistilBERT" and x["platform"] == "L40S" and x["treatment_raw"] == "fsdp:full_shard:root"
    ]
    if rootd:
        s.finding(
            "Wrapping matters: for DistilBERT on L40S, root-wrapped full_shard "
            + ", ".join(f"{x['gpus']} GPU {sign_pct(x['change'])}" for x in rootd)
            + "; per-layer wrapping helps at 8 GPUs (see matched contrasts below)."
        )
    per_layer = [
        x for x in st if x["platform"] == "L40S" and x["gpus"] == 8 and x["treatment_raw"].endswith("transformer_layer")
    ]
    s.finding(
        f"On L40S at 8 GPUs, per-layer FSDP beats DDP in all {len(per_layer)} contrasts ("
        f"{min(x['change'] for x in per_layer):+.0f}% to {max(x['change'] for x in per_layer):+.0f}%); the gap grows with model size."
    )
    s.d["coverage"] = coverage_summary(s.d["ids"], coverage)
    studies.append(s.d)

    # ---- S6 placement / NCCL + recovery
    pl = [eff_row(e, conds) for e in effects_by("nccl_and_placement")]
    rec_ids = [f"E{n}" for n in range(796, 810)]
    s = Study(
        "s6-topology",
        "Training · scaling deep dive",
        "S6 · GPU placement, peer-to-peer and NCCL algorithm",
        "Does it matter which L40S GPUs a job lands on, or how NCCL moves the data?",
        "On the L40S node, GPUs come in pairs that share a PCIe switch; other pairs talk through the CPU interconnect, and half the GPUs sit on a second CPU socket. "
        "If placement mattered, unpinned jobs would vary from run to run depending on where Slurm put them.",
        [i for x in pl for i in (x["control_id"], x["treatment_id"])] + rec_ids,
        [
            "Pinned GPU sets: same PCIe switch (0,1), same socket (0,2), across sockets (0,4); 4-GPU sets 0-3, 0,1,4,5 and 0,2,4,6.",
            "P2P disabled; NCCL Ring vs Tree. Original deep-dive runs E608–E637 compare with unpinned default runs.",
            "Recovery set E796–E809 re-ran the interrupted conditions in one current runtime cohort with matched defaults (E796 GPT-2, E806 TinyLlama). "
            "Its Tree/Ring settings apply to all-reduce only (allreduce:Tree), replacing the original invalid global Tree setting (E617 failed).",
        ],
        [
            "results/plot_data/training_parameter_effects/nccl_and_placement_effects.csv",
            "results/plot_data/trial_level/performance_condition_summary.csv (E796–E809)",
        ],
    )
    s.table(
        "Original deep-dive contrasts against unpinned default",
        "Throughput change against the unpinned default at the same GPU count.",
        EFFECT_COLUMNS,
        sorted(pl, key=lambda x: (x["workload"], x["gpus"], x["treatment"])),
    )
    rec_rows = []

    def env_label(eid):
        env = configs[eid][0]["runtime"].get("environment", {})
        parts = []
        if env.get("CUDA_VISIBLE_DEVICES"):
            parts.append(f"GPUs {env['CUDA_VISIBLE_DEVICES']}")
        if env.get("NCCL_ALGO"):
            parts.append(env["NCCL_ALGO"])
        if env.get("NCCL_P2P_DISABLE"):
            parts.append("P2P off")
        return ", ".join(parts) or "default (unpinned)"

    controls = {
        ("gpt2_small", "8"): "E796",
        ("tinyllama", "8"): "E806",
        ("tinyllama", "2"): "E799",
        ("tinyllama", "4"): "E801",
    }
    for eid in rec_ids:
        x = perf.get((eid, "l40s"))
        if not x:
            continue
        cid = controls.get((x["workload"], x["gpus"]))
        c = perf.get((cid, "l40s"))
        ratio, lo, hi = ratio_ci(
            num(x["throughput_tokens_sec_mean"]),
            num(x["throughput_tokens_sec_sd"]),
            num(x["trial_count"]),
            num(c["throughput_tokens_sec_mean"]),
            num(c["throughput_tokens_sec_sd"]),
            num(c["trial_count"]),
        )
        change, clo, chi = (ratio - 1) * 100, (lo - 1) * 100 if lo else None, (hi - 1) * 100 if hi else None
        verdict = verdict_of(change, clo, chi) if eid != cid else "control"
        rec_rows.append(
            {
                "platform": "L40S",
                "workload": WORKLOAD_LABELS[x["workload"]],
                "gpus": int(x["gpus"]),
                "setting": env_label(eid),
                "control": env_label(cid),
                "tps": r(num(x["throughput_tokens_sec_mean"]), 0),
                "change": r(change, 2) if eid != cid else None,
                "lo": r(clo, 2) if eid != cid else None,
                "hi": r(chi, 2) if eid != cid else None,
                "verdict": verdict,
                "ids": eid if eid == cid else f"{cid} → {eid}",
            }
        )
    s.table(
        "Recovery set (one runtime cohort, matched controls)",
        "2- and 4-GPU placements are compared with the same-switch (0,1) or single-socket (0–3) set; 8-GPU settings with the unpinned default. CI: delta-method ratio, t with 4 d.f.",
        [
            col("workload", "Workload"),
            col("gpus", "GPUs", "int"),
            col("setting", "Setting"),
            col("control", "Compared with"),
            col("tps", "Tokens/s", "int"),
            col("change", "Change", "signed1"),
            col("lo", "CI low", "signed1"),
            col("hi", "CI high", "signed1"),
            col("verdict", "Verdict", "verdict"),
            col("ids", "Experiments", "id"),
        ],
        rec_rows,
    )
    s.chart(
        **forest(
            [x for x in rec_rows if x["change"] is not None],
            lambda x: f"{SHORT_BY_LABEL(x['workload'])} {x['gpus']} GPU · {x['setting']}",
            "Recovery set: placement, P2P and NCCL settings (L40S)",
            "Throughput change against the matched control in the same runtime cohort. Shaded band: ±3%.",
        )
    )
    nm = Counter(x["verdict"] for x in pl)
    s.finding(
        f"Original contrasts: {nm['no_material_effect']} of {len(pl)} show no material effect — pinning to a shared PCIe switch, a single socket, "
        "or across sockets changed GPT-2 and TinyLlama throughput by under 3% at 2 and 4 GPUs."
    )
    big = [x for x in rec_rows if x["verdict"] in ("helps", "hurts")]
    if big:
        s.finding(
            "Recovery set material effects: "
            + "; ".join(f"{x['workload']} {x['gpus']} GPU {x['setting']} {sign_pct(x['change'])}" for x in big)
            + "."
        )
    s.finding(
        "Placement and NCCL effects are at most a few percent (all-reduce Tree is the exception at −12%), an order of magnitude smaller than the "
        "+33% to +117% from compression, FSDP and accumulation (S1, S4, S5). Placement does not explain the earlier poor TinyLlama scaling on L40S."
    )
    s.caveat(
        "Topology is the one captured by E215 on this node: switch pairs (0,1), (2,3), (4,5), (6,7); GPUs 4–7 on socket 1."
    )
    s.d["coverage"] = coverage_summary(s.d["ids"], coverage)
    studies.append(s.d)

    # ---- S7 real text
    rt_rows = report_conditions("l40s_real_text_batching.csv", perf, "l40s")
    s = Study(
        "s7-real-text",
        "Training · scaling deep dive",
        "S7 · Real text: padding, packing and stragglers",
        "Do realistic variable-length inputs cost speed or scaling that synthetic fixed-length data hides?",
        "Real paragraphs are short and uneven. Padding every one to 1,024 tokens wastes compute; padding each batch only to its own longest item wastes less but makes GPUs finish at different times; "
        "packing joins paragraphs into full windows and wastes nothing. Throughput here counts only real (non-padding) tokens.",
        [x["experiment_id"] for x in rt_rows],
        [
            "WikiText-103 (pinned revision), GPT-2 8×1024 and TinyLlama 4×1024, 1/2/4/8 GPUs on L40S.",
            "Modes: pad to max, dynamic padding, length-bucketed dynamic padding, packing. Control: pad to max.",
        ],
        [
            "results/report_data/l40s_real_text_batching.csv",
            "results/plot_data/training_parameter_effects/text_batching_effects.csv",
        ],
    )
    MODE = {
        "wikitext103:pad_to_max": "pad to max",
        "wikitext103:dynamic_pad": "dynamic pad",
        "wikitext103:length_bucketed": "length-bucketed",
        "wikitext103:pack": "pack",
    }
    panels = []
    for wl in ("gpt2_small", "tinyllama"):
        series = defaultdict(list)
        for x in rt_rows:
            if x["workload"] == wl:
                series[MODE[x["input_contract"]]].append(
                    {
                        "x": int(x["gpus"]),
                        "y": r(num(x["mean_throughput_tokens_sec"]), 1),
                        "lo": r(num(x["throughput_ci95_low"]), 1),
                        "hi": r(num(x["throughput_ci95_high"]), 1),
                        "id": x["experiment_id"],
                    }
                )
        panels.append(
            {
                "title": f"L40S · {WORKLOAD_LABELS[wl]}",
                "series": [{"name": k, "points": sorted(v, key=lambda p: p["x"])} for k, v in series.items()],
            }
        )
    s.chart(
        type="line",
        title="Useful tokens per second by batching mode",
        caption="Only non-padding tokens count. Whiskers: 95% CIs. Log scale.",
        x={"label": "GPUs", "ticks": [1, 2, 4, 8]},
        y={"label": "Useful tokens / s", "fmt": "si", "log": True},
        panels=panels,
    )
    tab = []
    for x in sorted(rt_rows, key=lambda x: (x["workload"], int(x["gpus"]), list(MODE).index(x["input_contract"]))):
        tab.append(
            {
                "workload": WORKLOAD_LABELS[x["workload"]],
                "gpus": int(x["gpus"]),
                "mode": MODE[x["input_contract"]],
                "tps": r(num(x["mean_throughput_tokens_sec"]), 0),
                "pad": r(num(x["padding_fraction"]), 3),
                "imb": r(num(x["rank_token_imbalance"]), 3),
                "syn": r(num(x["synthetic_same_shape_tokens_sec"]), 0),
                "id": x["experiment_id"],
            }
        )
    s.table(
        "Real-text conditions",
        "Padding fraction = share of processed tokens that were padding. Rank imbalance = busiest GPU's tokens ÷ mean, averaged over steps (1.0 = balanced). "
        "Synthetic column = the same shape with synthetic fixed-length data.",
        [
            col("workload", "Workload"),
            col("gpus", "GPUs", "int"),
            col("mode", "Mode"),
            col("tps", "Useful tokens/s", "int"),
            col("pad", "Padding", "pct0"),
            col("imb", "Rank imbalance", "num2"),
            col("syn", "Synthetic tokens/s", "int"),
            col("id", "Experiment", "id"),
        ],
        tab,
    )
    for wl in ("GPT-2 Small", "TinyLlama-1.1B"):
        g8 = {x["mode"]: x for x in tab if x["workload"] == wl and x["gpus"] == 8}
        if "pack" in g8 and "pad to max" in g8:
            s.finding(
                f"{wl}, 8 GPUs: packing delivers {g8['pack']['tps'] / g8['pad to max']['tps']:.1f}× the useful tokens/s of pad-to-max "
                f"(padding {g8['pad to max']['pad']:.0%} vs 0%); dynamic padding {g8['dynamic pad']['tps'] / g8['pad to max']['tps']:.1f}×, "
                f"rank imbalance {g8['dynamic pad']['imb']:.2f}."
            )
    s.finding(
        "Bucketing reduces padding relative to dynamic padding but the gain over dynamic padding is small and, for TinyLlama at 2 GPUs, inconclusive."
    )
    for wl in ("GPT-2 Small", "TinyLlama-1.1B"):
        drops = [
            m
            for m in MODE.values()
            if any(x["mode"] == m and x["gpus"] == 2 for x in tab)
            and next(x["tps"] for x in tab if x["workload"] == wl and x["mode"] == m and x["gpus"] == 2)
            < next(x["tps"] for x in tab if x["workload"] == wl and x["mode"] == m and x["gpus"] == 1)
        ]
        if drops:
            s.caveat(
                f"{wl} at 2 GPUs is slower than at 1 GPU in {len(drops)} of 4 modes ({', '.join(drops)}), the same 2-GPU L40S anomaly "
                "first seen in E291; its 2-GPU points are shown but should not be read as a scaling trend."
            )
    s.caveat(
        "Large percentages are real but reflect the metric: pad-to-max at 1,024 tokens processes mostly padding for short WikiText paragraphs."
    )
    s.d["coverage"] = coverage_summary(s.d["ids"], coverage)
    studies.append(s.d)

    # ---- S8 precision
    pr = [eff_row(e, conds) for e in effects_by("precision")] + [eff_row(e, conds) for e in effects_by("fp8")]
    s = Study(
        "s8-precision",
        "Training · scaling deep dive",
        "S8 · Numeric precision under scaling",
        "Does faster arithmetic (BF16, FP8) leave communication more exposed, and how much speed does each format buy?",
        "Lower precision makes each GPU's maths faster but leaves the gradient exchange the same size, so scaling efficiency could drop even as throughput rises. "
        "FP8 here uses NVIDIA Transformer Engine, compared with Transformer Engine BF16 so the only change is the number format.",
        [i for x in pr for i in (x["control_id"], x["treatment_id"])],
        [
            "H100 only. FP32 (no autocast) and Transformer Engine BF16 versus the default BF16 autocast; FP8 versus Transformer Engine BF16.",
            "GPT-2 8×1024, DistilBERT 32×512, TinyLlama 2×1024 (FP32 memory), 1/2/4 GPUs.",
        ],
        [
            "results/plot_data/training_parameter_effects/precision_effects.csv",
            "results/plot_data/training_parameter_effects/fp8_effects.csv",
            "results/report_data/h100_matched_precision_ladder.csv",
        ],
    )
    s.chart(
        **forest(
            sorted(pr, key=lambda x: (x["workload"], x["treatment"], x["gpus"])),
            lambda x: f"{SHORT_BY_LABEL(x['workload'])} {x['treatment']} vs {x['control']} · {x['gpus']} GPU",
            "Precision contrasts (H100)",
            "Throughput change against the stated control at the same GPU count. Shaded band: ±3%.",
            group_fn=lambda x: x["workload"],
        )
    )
    s.table(
        "Every precision contrast",
        "Δ efficiency = change in scaling efficiency (percentage points) between treatment and control at that GPU count.",
        EFFECT_COLUMNS,
        sorted(pr, key=lambda x: (x["workload"], x["treatment"], x["gpus"])),
    )
    for wl in ("GPT-2 Small", "DistilBERT", "TinyLlama-1.1B"):
        f32 = [x for x in pr if x["workload"] == wl and x["treatment_raw"] == "fp32_params_fp32"]
        f8 = [x for x in pr if x["workload"] == wl and x["treatment_raw"] == "transformer_engine_fp8"]
        parts = []
        if f32:
            parts.append(
                "FP32 is "
                + ", ".join(f"{sign_pct(x['change'], 0)}" for x in sorted(f32, key=lambda x: x["gpus"]))
                + " vs BF16 (1/2/4 GPUs)"
            )
        if f8:
            parts.append(
                "FP8 is "
                + ", ".join(f"{sign_pct(x['change'])}" for x in sorted(f8, key=lambda x: x["gpus"]))
                + " vs TE BF16"
            )
        if parts:
            s.finding(f"{wl}: " + "; ".join(parts) + ".")
    f32_4 = [
        x for x in pr if x["treatment_raw"] == "fp32_params_fp32" and x["gpus"] == 4 and x["eff_points"] is not None
    ]
    if f32_4:
        s.finding(
            "Slower arithmetic hides communication: at 4 GPUs FP32 scales better than BF16 ("
            + ", ".join(f"{SHORT_BY_LABEL(x['workload'])} {x['eff_points']:+.1f} pts" for x in f32_4)
            + " of efficiency), yet it is still 56–79% slower in absolute throughput. Faster formats trade a little scaling for a lot of speed."
        )
    f8_4 = [
        x
        for x in pr
        if x["treatment_raw"] == "transformer_engine_fp8" and x["gpus"] == 4 and x["eff_points"] is not None
    ]
    if f8_4:
        s.finding(
            "FP8 versus TE BF16 at 4 GPUs changes efficiency by "
            + ", ".join(f"{SHORT_BY_LABEL(x['workload'])} {x['eff_points']:+.1f} pts" for x in f8_4)
            + " — the same direction, much smaller."
        )
    s.caveat("No L40S precision runs yet, so the PCIe version of this question is open.")
    s.d["coverage"] = coverage_summary(s.d["ids"], coverage)
    studies.append(s.d)

    # ---- Run length
    rl = read_csv(REPORT / "h100_run_length_robustness.csv")
    for row in rl:
        current, base = perf[(row["extended"], "h100")], perf[(row["standard"], "h100")]
        _, row["ratio_lo"], row["ratio_hi"] = ratio_ci(
            num(current["throughput_tokens_sec_mean"]),
            num(current["throughput_tokens_sec_sd"]),
            num(current["trial_count"]),
            num(base["throughput_tokens_sec_mean"]),
            num(base["throughput_tokens_sec_sd"]),
            num(base["trial_count"]),
        )
    s = Study(
        "run-length",
        "Training · scaling deep dive",
        "Run-length check",
        "Are 200 measured updates enough, or does throughput drift over longer runs?",
        "Every deep-dive number comes from 200 measured optimizer updates. This check repeats two conditions for 1,000 updates.",
        [x["standard"] for x in rl] + [x["extended"] for x in rl],
        ["H100, 4 GPUs: GPT-2 8×1024 and TinyLlama 4×1024, 200 vs 1,000 measured updates."],
        ["results/report_data/h100_run_length_robustness.csv"],
    )
    s.table(
        "200 vs 1,000 updates",
        "Ratio = 1,000-update throughput ÷ 200-update throughput, with 95% CI.",
        [
            col("workload", "Workload"),
            col("std", "200 updates (tokens/s)", "int"),
            col("ext", "1,000 updates (tokens/s)", "int"),
            col("ratio", "Ratio", "num4"),
            col("lo", "CI low", "num4"),
            col("hi", "CI high", "num4"),
            col("ids", "Experiments", "id"),
        ],
        [
            {
                "workload": x["workload"],
                "std": r(num(x["m_s"]), 0),
                "ext": r(num(x["m_e"]), 0),
                "ratio": r(num(x["ratio"]), 4),
                "lo": r(num(x["ratio_lo"]), 4),
                "hi": r(num(x["ratio_hi"]), 4),
                "ids": f"{x['standard']} → {x['extended']}",
            }
            for x in rl
        ],
    )
    s.finding(
        "Both ratios are within 0.1% of 1.000 and their CIs include 1: 200 updates measure steady-state throughput."
    )
    s.d["coverage"] = coverage_summary(s.d["ids"], coverage)
    studies.append(s.d)

    # ---- Microbatch vs accumulation follow-up
    mb = report_conditions("l40s_fixed_effective_batch_microbatch.csv", perf, "l40s")
    s = Study(
        "microbatch",
        "Training · follow-ups",
        "Micro-batch versus accumulation at a fixed effective batch",
        "With 16 sequences per GPU per update either way, is it better to use bigger micro-batches or more accumulation steps?",
        "Accumulation (S1) helped TinyLlama a lot. This follow-up asks whether the same 16 sequences per update are processed faster as 4 big micro-batches or 16 tiny ones.",
        [x["experiment_id"] for x in mb],
        [
            "L40S, TinyLlama 1024-token sequences, effective batch 16 per GPU: micro-batch 1 × 16, 2 × 8, 4 × 4 accumulation steps; 1/2/4/8 GPUs.",
            "Follow-up motivated by observed results (planned 2026-10-21 after E418–E433).",
        ],
        ["results/report_data/l40s_fixed_effective_batch_microbatch.csv"],
    )
    series = defaultdict(list)
    for x in mb:
        series[f"micro-batch {x['per_gpu_batch_size']} × accum {x['gradient_accumulation_steps']}"].append(
            {"x": int(x["gpus"]), "y": r(num(x["per_gpu"]), 1), "id": x["experiment_id"]}
        )
    s.chart(
        type="line",
        title="Tokens per second per GPU",
        caption="Total throughput ÷ GPUs. Flat lines would mean perfect weak scaling.",
        x={"label": "GPUs", "ticks": [1, 2, 4, 8]},
        y={"label": "Tokens / s per GPU", "fmt": "si"},
        panels=[
            {
                "title": "L40S · TinyLlama-1.1B, effective batch 16 per GPU",
                "series": [{"name": k, "points": sorted(v, key=lambda p: p["x"])} for k, v in series.items()],
            }
        ],
    )
    tab = [
        {
            "gpus": int(x["gpus"]),
            "mb": int(x["per_gpu_batch_size"]),
            "acc": int(x["gradient_accumulation_steps"]),
            "tps": r(num(x["throughput_tokens_sec_mean"]), 0),
            "lo": r(num(x["throughput_tokens_sec_ci95_low"]), 0),
            "hi": r(num(x["throughput_tokens_sec_ci95_high"]), 0),
            "per": r(num(x["per_gpu"]), 0),
            "mem": r(num(x["peak_memory_gib_mean"]), 1),
            "id": x["experiment_id"],
        }
        for x in mb
    ]
    s.table(
        "Conditions",
        "Throughput with 95% CI; peak memory per GPU.",
        [
            col("gpus", "GPUs", "int"),
            col("mb", "Micro-batch", "int"),
            col("acc", "Accum", "int"),
            col("tps", "Tokens/s", "int"),
            col("lo", "CI low", "int"),
            col("hi", "CI high", "int"),
            col("per", "Per GPU", "int"),
            col("mem", "Peak GiB", "num1"),
            col("id", "Experiment", "id"),
        ],
        tab,
    )
    by = {(x["mb"], x["gpus"]): x for x in tab}
    s.finding(
        f"At 8 GPUs, micro-batch 4 × accumulation 4 gives {by[(4, 8)]['tps']:,.0f} tokens/s, {by[(4, 8)]['tps'] / by[(1, 8)]['tps'] - 1:+.0%} over micro-batch 1 × 16 "
        f"({by[(1, 8)]['tps']:,.0f}), for {by[(4, 8)]['mem'] - by[(1, 8)]['mem']:.1f} GiB more memory per GPU."
    )
    s.finding(
        f"Weak scaling from 1 to 8 GPUs keeps {by[(4, 8)]['per'] / by[(4, 1)]['per']:.0%} (micro-batch 4), {by[(2, 8)]['per'] / by[(2, 1)]['per']:.0%} (2) and "
        f"{by[(1, 8)]['per'] / by[(1, 1)]['per']:.0%} (1) of the single-GPU rate: larger micro-batches are faster but scale slightly less well."
    )
    s.d["coverage"] = coverage_summary(s.d["ids"], coverage)
    studies.append(s.d)

    # ---- Inference: TinyLlama equal-budget layouts
    tl = read_csv(PLOT / "inference_layout_comparisons/tinyllama_conditions.csv")
    s = Study(
        "tinyllama-layouts",
        "Inference · serving layouts",
        "TinyLlama: tensor parallel versus independent replicas",
        "With a fixed number of GPUs, is it better to split one model copy across them (tensor parallel) or run independent copies?",
        "Tensor parallelism (TP) splits each request's maths across GPUs, which can cut per-request latency but adds communication. "
        "Replicas each serve their own requests with no communication. The node's GPU budget is fixed: 8 L40S or 4 H100.",
        [x["experiment_id"] for x in tl],
        [
            "Layouts L40S: TP1×8 replicas, TP2×4, TP4×2, TP8. H100: TP1×4, TP2×2, TP4.",
            "Prompt 128 or 1,280 tokens × 1 or 8 concurrent requests; exactly 128 output tokens; BF16; vLLM; 30 measured rounds, 3 trials.",
        ],
        [
            "results/plot_data/inference_layout_comparisons/tinyllama_conditions.csv",
            "results/report_data/h100_tinyllama_serving_layouts.csv",
        ],
    )
    add_layout_study(s, tl, "TinyLlama-1.1B")
    s.d["coverage"] = coverage_summary(s.d["ids"], coverage)
    studies.append(s.d)

    # ---- Inference: Qwen 7B layouts and concurrency
    qw = read_csv(PLOT / "inference_layout_comparisons/qwen7b_conditions.csv")
    s = Study(
        "qwen-layouts",
        "Inference · serving layouts",
        "Qwen2.5-7B: layouts, concurrency and goodput",
        "Does the TinyLlama answer hold for a model 7× larger, and how does the best layout change with load?",
        "A 7B model nearly fills one L40S, leaving little memory for each request's working state (KV cache). Splitting it across 2 or 4 GPUs frees that memory. "
        "Goodput counts only requests that finish within a latency target.",
        [x["experiment_id"] for x in qw],
        [
            "L40S only, 8-GPU budget: TP1×8, TP2×4, TP4×2. Prompt 128 or 1,280 × concurrency 1/2/4/8; 128 output tokens.",
            "Concurrency 2 and 4 are follow-ups motivated by observed results (planned 2026-10-21).",
        ],
        [
            "results/plot_data/inference_layout_comparisons/qwen7b_conditions.csv",
            "results/plot_data/inference_concurrency/qwen7b_long_prompt_conditions.csv",
            "results/report_data/qwen7b_goodput_by_latency_slo.csv (SLO grid)",
            "results/plot_data/request_distributions/inference_samples_long.csv (goodput recomputed)",
            "results/plot_data/trial_level/current_performance_trials.csv (measured durations)",
        ],
    )
    add_layout_study(s, qw, "Qwen2.5-7B")
    # concurrency sweep at 1280
    series = defaultdict(list)
    for x in qw:
        if x["prompt_tokens"] == "1280":
            series[layout_label(x["serving_layout"])].append(
                {
                    "x": int(x["concurrency"]),
                    "y": r(num(x["throughput_tokens_sec_mean"]), 2),
                    "lo": r(num(x["throughput_tokens_sec_ci95_low"]), 2),
                    "hi": r(num(x["throughput_tokens_sec_ci95_high"]), 2),
                    "id": x["experiment_id"],
                }
            )
    s.chart(
        type="line",
        title="Qwen2.5-7B throughput against concurrency (prompt 1,280, L40S)",
        caption="Generated tokens per second; whiskers 95% CIs.",
        x={"label": "Concurrent requests", "ticks": [1, 2, 4, 8]},
        y={"label": "Generated tokens / s", "fmt": "si"},
        panels=[
            {
                "title": "L40S · 8-GPU budget",
                "series": [{"name": k, "points": sorted(v, key=lambda p: p["x"])} for k, v in sorted(series.items())],
            }
        ],
    )
    gp = goodput_rows()
    series = defaultdict(list)
    for x in gp:
        if x["concurrency"] == "8" and float(x["slo_s"]) <= 4.0001:
            series[f"{layout_label(x['serving_layout'])} ({x['experiment_id']})"].append(
                {
                    "x": round(float(x["slo_s"]), 2),
                    "y": r(num(x["goodput_mean"]), 4),
                    "lo": r(max(0.0, num(x["goodput_ci95_low"])), 4)
                    if num(x["goodput_ci95_low"]) is not None
                    else None,
                    "hi": r(num(x["goodput_ci95_high"]), 4),
                }
            )
    onset = {}
    for x in gp:
        if x["concurrency"] != "8":
            continue
        key = (x["experiment_id"], layout_label(x["serving_layout"]))
        g = num(x["goodput_mean"]) or 0.0
        o = onset.setdefault(key, {"first": None, "peak": 0.0})
        if g > 0 and (o["first"] is None or float(x["slo_s"]) < o["first"]):
            o["first"] = float(x["slo_s"])
        o["peak"] = max(o["peak"], g)
    s.finding(
        "Goodput at concurrency 8 (prompt 1,280): "
        + "; ".join(
            f"{lay} first meets a {o['first']:.2f} s target and peaks at {o['peak']:.1f} requests/s ({eid})"
            for (eid, lay), o in sorted(onset.items(), key=lambda kv: kv[1]["first"] or 99)
        )
        + ". Tighter latency targets favour TP2 replicas."
    )
    s.chart(
        type="line",
        title="Goodput against latency target, concurrency 8",
        caption="Goodput = requests per second that complete within the end-to-end latency target (definition from the report pipeline, 30_fig_inference.R); "
        "95% CIs floored at zero. Prompt 1,280, 128 output tokens.",
        x={"label": "Latency target (s)"},
        y={"label": "Requests within target per second", "fmt": "num2"},
        panels=[
            {
                "title": "L40S · Qwen2.5-7B, concurrency 8",
                "series": [{"name": k, "points": sorted(v, key=lambda p: p["x"])} for k, v in sorted(series.items())],
            }
        ],
    )
    s.d["coverage"] = coverage_summary(s.d["ids"], coverage)
    studies.append(s.d)

    # ---- Core studies revisited
    core = read_csv(REPORT / "h100_core_scaling.csv")
    for row in core:
        current = perf[(row["experiment_id"], "h100")]
        baseline = perf[(row["base_id"], "h100")]
        ratio, lo, hi = ratio_ci(
            num(current["throughput_tokens_sec_mean"]),
            num(current["throughput_tokens_sec_sd"]),
            num(current["trial_count"]),
            num(baseline["throughput_tokens_sec_mean"]),
            num(baseline["throughput_tokens_sec_sd"]),
            num(baseline["trial_count"]),
        )
        scale = float(row["gpu_ratio"])
        row["efficiency_lo"] = 1.0 if row["experiment_id"] == row["base_id"] else lo / scale if lo is not None else None
        row["efficiency_hi"] = 1.0 if row["experiment_id"] == row["base_id"] else hi / scale if hi is not None else None
    s = Study(
        "core-scaling",
        "Core studies · current cohorts",
        "Core scaling on H100 (revisited)",
        "Do the headline strong/weak-scaling numbers survive the current runtime-cohort selection?",
        "Re-derived from the latest submission cohort per condition. Strong scaling splits a fixed job across more GPUs; weak scaling gives every GPU the same work.",
        [x["experiment_id"] for x in core],
        ["H100, BF16 DDP, 1/2/4 GPUs. GPT-2 strong scaling starts at 2 GPUs (the 1-GPU shape OOMs)."],
        ["results/report_data/h100_core_scaling.csv"],
    )
    series = defaultdict(list)
    for x in core:
        series[(x["kind"], x["workload_label"])].append(
            {
                "x": int(x["gpus"]),
                "y": r(num(x["efficiency"]), 4),
                "lo": r(num(x["efficiency_lo"]), 4),
                "hi": r(num(x["efficiency_hi"]), 4),
                "id": x["experiment_id"],
            }
        )
    s.chart(
        type="line",
        title="H100 scaling efficiency",
        caption="Relative to the smallest GPU count in each series; whiskers 95% CIs.",
        x={"label": "GPUs", "ticks": [1, 2, 4]},
        y={"label": "Scaling efficiency", "fmt": "pct0", "ref": 1.0},
        panels=[
            {
                "title": f"H100 · {kind}",
                "series": [
                    {"name": wl, "points": sorted(v, key=lambda p: p["x"])}
                    for (k, wl), v in sorted(series.items())
                    if k == kind
                ],
            }
            for kind in ("Weak scaling", "Strong scaling")
        ],
    )
    s.table(
        "Series",
        "Speedup and efficiency against the first GPU count of each series, with 95% CIs. TinyLlama strong scaling mixes runtime cohorts (flagged).",
        [
            col("kind", "Type"),
            col("workload", "Workload"),
            col("gpus", "GPUs", "int"),
            col("tps", "Tokens/s", "int"),
            col("speedup", "Speedup", "x2"),
            col("eff", "Efficiency", "pct1"),
            col("lo", "CI low", "pct1"),
            col("hi", "CI high", "pct1"),
            col("cohort", "Same cohort"),
            col("id", "Experiment", "id"),
        ],
        [
            {
                "kind": x["kind"],
                "workload": x["workload_label"],
                "gpus": int(x["gpus"]),
                "tps": r(num(x["throughput_tokens_sec_mean"]), 0),
                "speedup": r(num(x["speedup"]), 3),
                "eff": r(num(x["efficiency"]), 4),
                "lo": r(num(x["efficiency_lo"]), 4),
                "hi": r(num(x["efficiency_hi"]), 4),
                "cohort": "yes" if x["same_runtime_cohort"] == "TRUE" else "no",
                "id": x["experiment_id"],
            }
            for x in core
        ],
    )
    w4 = {x["workload_label"]: x for x in core if x["kind"] == "Weak scaling" and x["gpus"] == "4"}
    s.finding(
        "Weak scaling at 4 H100 GPUs: "
        + ", ".join(f"{k} {num(v['efficiency']):.0%}" for k, v in w4.items())
        + " — unchanged from the September snapshot."
    )
    s4 = {x["workload_label"]: x for x in core if x["kind"] == "Strong scaling" and x["gpus"] == "4"}
    s.finding(
        "Strong scaling at 4 GPUs: "
        + ", ".join(f"{k} {num(v['efficiency']):.0%}" for k, v in s4.items())
        + "; splitting a fixed job costs more efficiency than growing it."
    )
    s.caveat("TinyLlama strong scaling (E044–E046) spans two runtime cohorts; treat its efficiency as indicative.")
    s.d["coverage"] = coverage_summary(s.d["ids"], coverage)
    studies.append(s.d)

    stg = report_conditions("h100_strategy_tradeoff.csv", perf, "h100")
    s = Study(
        "core-strategy",
        "Core studies · current cohorts",
        "Distributed strategies on H100 (revisited)",
        "Which way of splitting training across GPUs gives the best speed for the memory it uses?",
        "DDP, FSDP (two modes) and DeepSpeed ZeRO stages 1–3 on the same job. Compare strategies only at the same GPU count.",
        [x["experiment_id"] for x in stg],
        [
            "H100, GPT-2 Small and DistilBERT, BF16; all six strategies at 2 GPUs, DDP and FSDP full shard also at 4 GPUs."
        ],
        ["results/report_data/h100_strategy_tradeoff.csv"],
    )
    order = ["DDP", "FSDP full shard", "FSDP grad-op shard", "ZeRO-1", "ZeRO-2", "ZeRO-3"]
    groups = []
    for wl in ("GPT-2 Small", "DistilBERT"):
        for g in ("2", "4"):
            members = sorted(
                [x for x in stg if x["workload_label"] == wl and x["gpus"] == g], key=lambda x: order.index(x["stack"])
            )
            if members:
                groups.append(
                    {
                        "label": f"{wl} · {g} GPUs",
                        "bars": [
                            {
                                "name": x["stack"],
                                "y": r(num(x["throughput_tokens_sec_mean"]), 0),
                                "lo": r(num(x["throughput_tokens_sec_ci95_low"]), 0),
                                "hi": r(num(x["throughput_tokens_sec_ci95_high"]), 0),
                                "id": x["experiment_id"],
                            }
                            for x in members
                        ],
                    }
                )
    s.chart(
        type="bar",
        title="Throughput by strategy and GPU count (H100)",
        caption="Whiskers 95% CIs. Bars are comparable only within a group (same workload and GPU count).",
        y={"label": "Tokens / s", "fmt": "si"},
        groups=groups,
    )
    s.table(
        "Strategies",
        "Peak memory per GPU; energy per 1,000 processed tokens from measured-region telemetry.",
        [
            col("workload", "Workload"),
            col("gpus", "GPUs", "int"),
            col("stack", "Strategy"),
            col("tps", "Tokens/s", "int"),
            col("mem", "Peak GiB", "num1"),
            col("j", "J / 1k tokens", "num3"),
            col("id", "Experiment", "id"),
        ],
        [
            {
                "workload": x["workload_label"],
                "gpus": int(x["gpus"]),
                "stack": x["stack"],
                "tps": r(num(x["throughput_tokens_sec_mean"]), 0),
                "mem": r(num(x["peak_memory_gib_mean"]), 2),
                "j": r(num(x["joules_per_processed_token_mean"]) * 1000, 3)
                if num(x["joules_per_processed_token_mean"])
                else None,
                "id": x["experiment_id"],
            }
            for x in sorted(stg, key=lambda x: (x["workload_label"], int(x["gpus"]), order.index(x["stack"])))
        ],
    )
    for wl in ("GPT-2 Small", "DistilBERT"):
        rows_w = [x for x in stg if x["workload_label"] == wl and x["gpus"] == "2"]
        best = max(rows_w, key=lambda x: num(x["throughput_tokens_sec_mean"]))
        least = min(rows_w, key=lambda x: num(x["peak_memory_gib_mean"]))
        ddp = next(x for x in rows_w if x["stack"] == "DDP")
        s.finding(
            f"{wl}, 2 GPUs: fastest {best['stack']} ({num(best['throughput_tokens_sec_mean']):,.0f} tok/s, "
            f"{num(best['throughput_tokens_sec_mean']) / num(ddp['throughput_tokens_sec_mean']) - 1:+.1%} vs DDP); "
            f"least memory {least['stack']} ({num(least['peak_memory_gib_mean']):.1f} GiB vs DDP {num(ddp['peak_memory_gib_mean']):.1f} GiB)."
        )
    z3 = []
    for x in (y for y in stg if y["stack"] == "ZeRO-3"):
        ddp = next(
            y
            for y in stg
            if y["workload_label"] == x["workload_label"] and y["gpus"] == x["gpus"] and y["stack"] == "DDP"
        )
        z3.append(
            f"{x['workload_label']} {num(x['throughput_tokens_sec_mean']) / num(ddp['throughput_tokens_sec_mean']):.0%}"
        )
    s.finding(
        "ZeRO-3 throughput relative to DDP at 2 GPUs: "
        + ", ".join(z3)
        + " — full parameter sharding is model-dependent at this scale, not uniformly slower."
    )
    s.d["coverage"] = coverage_summary(s.d["ids"], coverage)
    studies.append(s.d)

    prec = read_csv(REPORT / "h100_precision.csv")
    for row in prec:
        base_id = next(
            x["experiment_id"]
            for x in prec
            if x["workload_label"] == row["workload_label"] and x["precision"] == "FP32"
        )
        current, base = perf[(row["experiment_id"], "h100")], perf[(base_id, "h100")]
        _, lo, hi = ratio_ci(
            num(current["throughput_tokens_sec_mean"]),
            num(current["throughput_tokens_sec_sd"]),
            num(current["trial_count"]),
            num(base["throughput_tokens_sec_mean"]),
            num(base["throughput_tokens_sec_sd"]),
            num(base["trial_count"]),
        )
        row["speedup_lo"] = 1.0 if row["experiment_id"] == base_id else lo
        row["speedup_hi"] = 1.0 if row["experiment_id"] == base_id else hi
    s = Study(
        "core-precision",
        "Core studies · current cohorts",
        "Single-GPU precision (revisited)",
        "How much faster is each number format than FP32 on one H100?",
        "FP32, FP16, BF16 and FP8 on one GPU.",
        [x["experiment_id"] for x in prec],
        ["H100, 1 GPU, GPT-2 Small 16×1024 and DistilBERT 32×512."],
        ["results/report_data/h100_precision.csv"],
    )
    groups = [
        {
            "label": wl,
            "bars": [
                {
                    "name": x["precision"],
                    "y": r(num(x["speedup_vs_fp32"]), 3),
                    "lo": r(num(x["speedup_lo"]), 3),
                    "hi": r(num(x["speedup_hi"]), 3),
                    "id": x["experiment_id"],
                }
                for x in prec
                if x["workload_label"] == wl
            ],
        }
        for wl in ("GPT-2 Small", "DistilBERT")
    ]
    s.chart(
        type="bar",
        title="Speed-up over FP32 (1 H100)",
        caption="Ratio of mean throughput to FP32, 95% CIs.",
        y={"label": "× FP32 throughput", "fmt": "x2", "ref": 1.0},
        groups=groups,
    )
    s.table(
        "Precision",
        "Speed-up against FP32 with 95% CI.",
        [
            col("workload", "Workload"),
            col("precision", "Format"),
            col("tps", "Tokens/s", "int"),
            col("speedup", "× FP32", "x2"),
            col("lo", "CI low", "x2"),
            col("hi", "CI high", "x2"),
            col("mem", "Peak GiB", "num1"),
            col("id", "Experiment", "id"),
        ],
        [
            {
                "workload": x["workload_label"],
                "precision": x["precision"],
                "tps": r(num(x["throughput_tokens_sec_mean"]), 0),
                "speedup": r(num(x["speedup_vs_fp32"]), 3),
                "lo": r(num(x["speedup_lo"]), 3),
                "hi": r(num(x["speedup_hi"]), 3),
                "mem": r(num(x["peak_memory_gib_mean"]), 1),
                "id": x["experiment_id"],
            }
            for x in prec
        ],
    )
    for wl in ("GPT-2 Small", "DistilBERT"):
        f8 = next(x for x in prec if x["workload_label"] == wl and x["precision"] == "FP8")
        s.finding(f"{wl}: FP8 is {num(f8['speedup_vs_fp32']):.2f}× FP32 on one H100.")
    s.d["coverage"] = coverage_summary(s.d["ids"], coverage)
    studies.append(s.d)

    hw = read_csv(REPORT / "hardware_h100_vs_l40s.csv")
    s = Study(
        "core-hardware",
        "Core studies · current cohorts",
        "H100 versus L40S on matched single-GPU jobs",
        "How much faster and more energy-efficient is the H100 per GPU?",
        "Same job, one GPU each, both platforms.",
        [x["h100"] for x in hw] + [x["l40s"] for x in hw],
        ["One GPU, BF16, GPT-2 Small and DistilBERT, training and inference."],
        ["results/report_data/hardware_h100_vs_l40s.csv"],
    )
    s.table(
        "Matched pairs",
        "Throughput ratio H100 ÷ L40S; energy ratio H100 ÷ L40S per token (below 1 = H100 uses less energy).",
        [
            col("workload", "Workload"),
            col("mode", "Mode"),
            col("ratio", "Throughput H100 ÷ L40S", "x2"),
            col("eratio", "Energy/token H100 ÷ L40S", "x2"),
            col("ids", "Experiments", "id"),
        ],
        [
            {
                "workload": x["workload"],
                "mode": x["mode"],
                "ratio": r(num(x["thr_ratio"]), 3),
                "eratio": r(num(x["energy_ratio"]), 3),
                "ids": f"{x['h100']} / {x['l40s']}",
            }
            for x in hw
        ],
    )
    for x in hw:
        s.finding(
            f"{x['workload']} {x['mode'].lower()}: H100 is {num(x['thr_ratio']):.2f}× L40S throughput"
            + (
                f", using {num(x['energy_ratio']):.2f}× the energy per token."
                if num(x["energy_ratio"])
                else " (energy not measured)."
            )
        )
    s.d["coverage"] = coverage_summary(s.d["ids"], coverage)
    studies.append(s.d)

    lb = read_csv(REPORT / "qwen7b_lowbit_vs_bf16.csv")
    s = Study(
        "core-lowbit",
        "Core studies · current cohorts",
        "Qwen2.5-7B: 4-bit formats versus BF16",
        "What do 4-bit model formats cost in quality and buy in speed, memory and energy?",
        "Publisher AWQ and GPTQ 4-bit releases and on-the-fly NF4 against the BF16 model, served with vLLM.",
        [x["experiment_id"] for x in lb],
        ["H100, 1 GPU, WikiText-2 perplexity; 10% perplexity tolerance."],
        ["results/report_data/qwen7b_lowbit_vs_bf16.csv"],
    )
    s.table(
        "Formats against BF16",
        "Changes are relative to BF16 (E270).",
        [
            col("method", "Format"),
            col("tps", "Tokens/s", "num1"),
            col("thr", "Throughput", "signed1"),
            col("lat", "p95 latency", "signed1"),
            col("mem", "Model memory saved", "pct0d"),
            col("energy", "Energy/token", "signed1"),
            col("ppl", "Perplexity", "num2"),
            col("pplc", "Perplexity change", "signed1"),
            col("id", "Experiment", "id"),
        ],
        [
            {
                "method": LOWBIT.get(x["method"], x["method"]),
                "tps": r(num(x["mean_throughput_tokens_sec"]), 1),
                "thr": r(num(x["throughput_change_pct"]), 1),
                "lat": r(num(x["latency_p95_change_pct"]), 1),
                "mem": r(num(x["model_memory_saving_pct"]), 1),
                "energy": r(num(x["energy_per_token_change_pct"]), 1),
                "ppl": r(num(x["mean_quality_perplexity"]), 2),
                "pplc": r(num(x["perplexity_change_pct"]), 1),
                "id": x["experiment_id"],
            }
            for x in lb
        ],
    )
    for x in lb[1:]:
        s.finding(
            f"{LOWBIT.get(x['method'], x['method'])}: throughput {sign_pct(num(x['throughput_change_pct']))}, memory −{num(x['model_memory_saving_pct']):.0f}%, energy/token {sign_pct(num(x['energy_per_token_change_pct']))}, "
            f"perplexity {sign_pct(num(x['perplexity_change_pct']))} (within the 10% tolerance)."
        )
    s.d["coverage"] = coverage_summary(s.d["ids"], coverage)
    studies.append(s.d)
    for study in studies:
        if study["id"] in {
            "s7-real-text",
            "run-length",
            "microbatch",
            "core-scaling",
            "core-strategy",
            "core-precision",
        }:
            study["sources"].append(
                "results/plot_data/trial_level/performance_condition_summary.csv (refreshed intervals)"
            )
    return studies


LOWBIT = {"bf16": "BF16", "awq": "AWQ 4-bit", "gptq": "GPTQ 4-bit", "bitsandbytes_nf4": "bitsandbytes NF4"}


def SHORT_BY_LABEL(label):
    return {
        "GPT-2 Small": "GPT-2",
        "DistilBERT": "DistilBERT",
        "TinyLlama-1.1B": "TinyLlama",
        "GPT-2 Medium": "GPT-2 M",
        "GPT-2 Large": "GPT-2 L",
    }.get(label, label)


def verdict_of(change, lo, hi, material=3.0):
    if None in (change, lo, hi):
        return "inconclusive"
    if lo > 0 and change >= material:
        return "helps"
    if hi < 0 and change <= -material:
        return "hurts"
    if -material < lo and hi < material:
        return "no_material_effect"
    return "inconclusive"


def layout_label(layout):
    m = re.fullmatch(r"tp(\d+)(?:_dp(\d+))?", layout or "")
    if not m:
        return layout
    return f"TP{m[1]} × {m[2]} replicas" if m[2] else f"TP{m[1]}, single engine"


def add_layout_study(s: Study, rows, model):
    rows = [x for x in rows if x["role"] == "performance"]

    def shape(x):
        return f"prompt {int(x['prompt_tokens']):,} · {x['concurrency']} concurrent"

    for platform in PLATFORMS:
        pr = [x for x in rows if x["platform"] == platform]
        if not pr:
            continue
        for metric, label, fmt, title in (
            ("throughput_tokens_sec", "Generated tokens / s", "si", "Throughput"),
            ("ttft_p95_ms", "Time to first token, p95 (ms)", "num0", "Time to first token (p95)"),
            ("joules_per_output_token", "Joules per output token", "num2", "Energy per output token"),
        ):
            groups = []
            for key in sorted({(int(x["prompt_tokens"]), int(x["concurrency"])) for x in pr}):
                members = sorted(
                    [x for x in pr if (int(x["prompt_tokens"]), int(x["concurrency"])) == key],
                    key=lambda x: int(x["tensor_parallel_size"]),
                )
                groups.append(
                    {
                        "label": f"prompt {key[0]:,} · c{key[1]}",
                        "bars": [
                            {
                                "name": layout_label(x["serving_layout"]),
                                "y": r(num(x[f"{metric}_mean"]), 3),
                                "lo": r(num(x[f"{metric}_ci95_low"]), 3),
                                "hi": r(num(x[f"{metric}_ci95_high"]), 3),
                                "id": x["experiment_id"],
                            }
                            for x in members
                        ],
                    }
                )
            s.chart(
                type="bar",
                title=f"{GPU_NAME[platform]} · {model}: {title} by layout",
                caption="Condition means of 3 trials; whiskers 95% CIs. Lower is better for latency and energy.",
                y={"label": label, "fmt": fmt},
                groups=groups,
            )
    s.table(
        "Layout conditions",
        "Throughput = generated tokens/s across the whole node. TTFT = time to first token; TPOT = time per output token; p95 = trial-level 95th percentile, averaged over 3 trials.",
        [
            col("platform", "Platform"),
            col("shape", "Workload shape"),
            col("layout", "Layout"),
            col("tps", "Tokens/s", "num1"),
            col("ttft", "TTFT p95 (ms)", "num1"),
            col("tpot", "TPOT p95 (ms)", "num2"),
            col("lat", "Latency p95 (ms)", "int"),
            col("j", "J / output token", "num3"),
            col("id", "Experiment", "id"),
        ],
        [
            {
                "platform": GPU_NAME[x["platform"]],
                "shape": shape(x),
                "layout": layout_label(x["serving_layout"]),
                "tps": r(num(x["throughput_tokens_sec_mean"]), 1),
                "ttft": r(num(x["ttft_p95_ms_mean"]), 1),
                "tpot": r(num(x["time_per_output_token_p95_ms_mean"]), 2),
                "lat": r(num(x["latency_p95_ms_mean"]), 0),
                "j": r(num(x["joules_per_output_token_mean"]), 3),
                "id": x["experiment_id"],
            }
            for x in sorted(
                rows,
                key=lambda x: (
                    x["platform"],
                    int(x["prompt_tokens"]),
                    int(x["concurrency"]),
                    int(x["tensor_parallel_size"]),
                ),
            )
        ],
    )
    for platform in PLATFORMS:
        pr = [x for x in rows if x["platform"] == platform]
        for key in sorted({(int(x["prompt_tokens"]), int(x["concurrency"])) for x in pr}):
            members = [x for x in pr if (int(x["prompt_tokens"]), int(x["concurrency"])) == key]
            if len(members) < 2:
                continue
            fast = max(members, key=lambda x: num(x["throughput_tokens_sec_mean"]))
            slow = min(members, key=lambda x: num(x["throughput_tokens_sec_mean"]))
            eco = min(
                (x for x in members if num(x["joules_per_output_token_mean"])),
                key=lambda x: num(x["joules_per_output_token_mean"]),
                default=None,
            )
            spread = num(fast["throughput_tokens_sec_mean"]) / num(slow["throughput_tokens_sec_mean"])
            if spread < 1.03:
                text = (
                    f"{GPU_NAME[platform]}, prompt {key[0]:,}, {key[1]} concurrent: layouts within {100 * (spread - 1):.1f}% of each other "
                    f"(~{num(fast['throughput_tokens_sec_mean']):,.0f} tok/s)"
                )
            else:
                text = (
                    f"{GPU_NAME[platform]}, prompt {key[0]:,}, {key[1]} concurrent: fastest {layout_label(fast['serving_layout'])} "
                    f"({num(fast['throughput_tokens_sec_mean']):,.0f} tok/s, {spread:.2f}× the slowest, {layout_label(slow['serving_layout'])})"
                )
            if eco is not None:
                text += f"; lowest energy per token {layout_label(eco['serving_layout'])} ({num(eco['joules_per_output_token_mean']):.2f} J)"
            s.finding(text + ".")


# ---------------------------------------------------------------- Part 2 status
NEW_EVIDENCE = [  # (study id, which finding, plain-language lead)
    ("s1-accumulation", r"^L40S TinyLlama", "The budget chips' scaling problem is mostly fixable in software"),
    ("s5-stacks", r"^L40S TinyLlama", "Sharding the model (FSDP) more than doubles TinyLlama's 8-GPU speed on L40S"),
    ("s4-ddp", r"^L40S TinyLlama", "Compressing gradients helps on the slow link and hurts on the fast one"),
    ("s6-topology", r"^Placement and NCCL", "Which GPUs a job lands on barely matters"),
    ("s8-precision", r"^Slower arithmetic", "Faster number formats scale slightly worse, but are far faster overall"),
    ("s7-real-text", r"^GPT-2 Small", "With real text, how you batch it matters more than anything else"),
    (
        "tinyllama-layouts",
        r"^L40S, prompt 1,280, 8 concurrent",
        "For a small model, independent copies beat splitting it across GPUs",
    ),
    ("qwen-layouts", r"^Goodput", "For a 7B model, splitting it across 2 GPUs per copy is the sweet spot"),
    ("core-scaling", r"^Weak scaling", "The September headline numbers still hold"),
]


def replace_embedded(text: str, variable: str, payload: str) -> str:
    lines = text.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.lstrip().startswith(f"var {variable} = ")]
    if len(hits) != 1:
        raise SystemExit(f"Expected one `var {variable} =` line, found {len(hits)}")
    indent = lines[hits[0]][: len(lines[hits[0]]) - len(lines[hits[0]].lstrip())]
    lines[hits[0]] = f"{indent}var {variable} = {payload};\n"
    return "".join(lines)


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the current aggregated dashboard studies")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    configs, matrix, routing, coverage, perf, manifest = load_sources()
    catalog = build_catalog(configs, routing, coverage, perf, capacity_history(matrix), manifest)
    summaries = build_studies(coverage, perf, configs)
    ids = {eid for study in summaries for eid in study["ids"]}
    unique_trials = sum(int(row["completed_trials"] or 0) for (eid, platform), row in coverage.items() if eid in ids)
    studies = {
        "meta": {
            **catalog["meta"],
            "generator": "scripts/build_dashboard_studies.py",
            "study_conditions": len(ids),
            "study_completed_trials": unique_trials,
        },
        "studies": summaries,
    }
    payload = dumps(studies).replace("</", "<\\/")
    page = DASH / "studies.html"
    outputs = {
        DASH / "studies-data.json": payload + "\n",
        page: replace_embedded(page.read_text(encoding="utf-8"), "STUDIES", payload),
    }
    if args.check:
        stale = [p.name for p, content in outputs.items() if not p.exists() or p.read_text(encoding="utf-8") != content]
        if stale:
            raise SystemExit("Stale dashboard studies: " + ", ".join(stale))
        print("Dashboard studies match the evidence exports.")
        return
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
    print(f"Aggregated {len(studies['studies'])} studies into {page}")


if __name__ == "__main__":
    main()
