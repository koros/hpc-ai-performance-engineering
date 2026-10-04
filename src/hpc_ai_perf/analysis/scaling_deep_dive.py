"""Lever-by-lever analysis of the controlled scaling deep dive (scaling_deep_dive_v1).

Every physical trial is analysed once as a *condition* (one experiment on one
platform), then reused by every lever contrast it can support: one 8 x 1024
GPT-2 run is both the accumulation-1 control and a batch/sequence split point.
Levers are inferred from the parameters actually run, never from a label, so a
trial cannot be counted under a lever it does not isolate.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from typing import Any

from hpc_ai_perf.config.scaling import DEEP_DIVE_STUDY_ID

# Usable capacity observed by PyTorch on each platform (GiB), from the OOM records.
CAPACITY_GB = {"h100": 79.19, "l40s": 44.52}
MEMORY_RULE = 0.85
MATERIAL_PCT = 3.0

CONDITION_COLUMNS = [
    "platform_id",
    "experiment_id",
    "workload",
    "gpus",
    "per_gpu_batch_size",
    "sequence_length",
    "tokens_per_gpu_micro_step",
    "gradient_accumulation_steps",
    "global_batch_size",
    "stack",
    "ddp_settings",
    "precision_contract",
    "input_contract",
    "runtime_environment",
    "completed_trials",
    "failed_trials",
    "oom_trials",
    "mean_throughput_tokens_sec",
    "throughput_ci95_low",
    "throughput_ci95_high",
    "throughput_cv_pct",
    "baseline_experiment_id",
    "scaling_efficiency",
    "efficiency_ci95_low",
    "efficiency_ci95_high",
    "step_time_p50_ms",
    "step_time_p95_ms",
    "straggler_ratio_p95_p50",
    "exposed_communication_ms",
    "communication_fraction",
    "gpu_hours_per_million_tokens",
    "peak_memory_gb",
    "memory_fraction_of_capacity",
    "memory_rule_status",
    "padding_fraction",
    "rank_token_imbalance",
    "placement_classes",
]

EFFECT_COLUMNS = [
    "lever",
    "platform_id",
    "workload",
    "gpus",
    "control_experiment_id",
    "treatment_experiment_id",
    "control_setting",
    "treatment_setting",
    "throughput_change_pct",
    "change_ci95_low",
    "change_ci95_high",
    "efficiency_change_points",
    "verdict",
]

# One field per lever: (field, control value, treatment values or None for any other value).
LEVER_FIELDS = {
    "gradient_accumulation": ("gradient_accumulation_steps", "1", None),
    "ddp_communication": ("ddp_settings", "default", None),
    "training_stack": ("stack", "ddp", None),
    "precision": ("precision_contract", "fp32_params_bf16_autocast", {"fp32_params_fp32", "transformer_engine_bf16"}),
    "fp8": ("precision_contract", "transformer_engine_bf16", {"transformer_engine_fp8"}),
    "text_batching": (
        "input_contract",
        "wikitext103:pad_to_max",
        {"wikitext103:dynamic_pad", "wikitext103:length_bucketed", "wikitext103:pack"},
    ),
    "real_vs_synthetic": ("input_contract", "synthetic", {"wikitext103:pack"}),
    "nccl_and_placement": ("runtime_environment", "default", None),
}


def deep_dive_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("parameter_scaling_study_id") == DEEP_DIVE_STUDY_ID]


def build_scaling_deep_dive_conditions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [r for r in deep_dive_rows(rows) if r.get("parameter_scaling_role") != "canary"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[(str(row.get("platform_id")), str(row.get("experiment_id")))].append(row)
    conditions = [_condition(platform, eid, trials) for (platform, eid), trials in sorted(groups.items())]
    by_key = {(c["platform_id"], c["_baseline_key"]): c for c in conditions if c["gpus"] == 1 and c["_is_default"]}
    for condition in conditions:
        baseline = by_key.get((condition["platform_id"], condition["_baseline_key"]))
        _attach_baseline(condition, baseline)
    return conditions


def build_scaling_deep_dive_effects(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Conditions whose trials all failed (for example OOM) have no throughput to contrast.
    conditions = [c for c in conditions if c["_throughputs"]]
    effects = []
    for lever, (field, control_value, treatments) in LEVER_FIELDS.items():
        effects.extend(_field_effects(conditions, lever, field, control_value, treatments))
    effects.extend(_split_effects(conditions))
    return sorted(
        effects, key=lambda e: (e["lever"], e["platform_id"], e["workload"], e["gpus"], e["treatment_setting"])
    )


def public_rows(rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    return [{column: row.get(column, "") for column in columns} for row in rows]


def render_scaling_deep_dive_report(conditions: list[dict[str, Any]], effects: list[dict[str, Any]]) -> str:
    lines = [
        "# Scaling deep dive: generated findings",
        "",
        f"Study `{DEEP_DIVE_STUDY_ID}`. Generated from processed trial rows; do not edit by hand.",
        "Effects are throughput changes against the lever's control at the same GPU count. A change counts as",
        f"material at {MATERIAL_PCT:.0f}% or more, and only when its 95% confidence interval excludes zero.",
        "",
        "## Coverage",
        "",
    ]
    by_platform = Counter(c["platform_id"] for c in conditions)
    failed = sum(int(c["failed_trials"]) for c in conditions)
    ooms = sum(int(c["oom_trials"]) for c in conditions)
    for platform, count in sorted(by_platform.items()):
        trials = sum(int(c["completed_trials"]) for c in conditions if c["platform_id"] == platform)
        lines.append(f"- {platform.upper()}: {count} conditions, {trials} completed trials.")
    lines.append(f"- Failed trials: {failed} (of which CUDA OOM: {ooms}).")
    over = [c for c in conditions if c["memory_rule_status"] == "over_rule"]
    if over:
        names = ", ".join(f"{c['experiment_id']}@{c['platform_id']}" for c in over)
        lines.append(f"- Above the {MEMORY_RULE:.0%} memory rule: {names}.")
    lines += ["", "## Lever verdicts", ""]
    lines += ["| Lever | Platform | Helps | Hurts | No material effect | Inconclusive |", "|---|---|---|---|---|---|"]
    tally: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for effect in effects:
        tally[(effect["lever"], effect["platform_id"])][effect["verdict"]] += 1
    for (lever, platform), counts in sorted(tally.items()):
        lines.append(
            f"| {lever} | {platform.upper()} | {counts['helps']} | {counts['hurts']} | "
            f"{counts['no_material_effect']} | {counts['inconclusive']} |"
        )
    material = [e for e in effects if e["verdict"] in {"helps", "hurts"}]
    material.sort(key=lambda e: -abs(float(e["throughput_change_pct"])))
    if material:
        lines += ["", "## Largest material effects", ""]
        lines += ["| Lever | Platform | Workload | GPUs | Change | Setting |", "|---|---|---|---|---|---|"]
        for effect in material[:20]:
            lines.append(
                f"| {effect['lever']} | {effect['platform_id'].upper()} | {effect['workload']} | {effect['gpus']} | "
                f"{float(effect['throughput_change_pct']):+.1f}% | {effect['control_setting']} -> "
                f"{effect['treatment_setting']} |"
            )
    lines += ["", "## Scaling efficiency at the largest GPU count, default DDP", ""]
    lines += [
        "| Platform | Workload | Shape | GPUs | Efficiency | Exposed comm. (ms/step) |",
        "|---|---|---|---|---|---|",
    ]
    best: dict[tuple, dict[str, Any]] = {}
    for c in conditions:
        if not c["_is_default"] or c["scaling_efficiency"] in ("", None):
            continue
        key = (
            c["platform_id"],
            c["workload"],
            c["per_gpu_batch_size"],
            c["sequence_length"],
            c["gradient_accumulation_steps"],
        )
        if key not in best or c["gpus"] > best[key]["gpus"]:
            best[key] = c
    for (platform, workload, batch, seq, accumulation), c in sorted(best.items()):
        if c["gpus"] == 1:
            continue
        comm = c["exposed_communication_ms"]
        lines.append(
            f"| {platform.upper()} | {workload} | {batch}x{seq}, accum {accumulation} | {c['gpus']} | "
            f"{float(c['scaling_efficiency']):.0%} | {'' if comm in ('', None) else f'{float(comm):.1f}'} |"
        )
    placement = _placement_summary(conditions)
    if placement:
        lines += ["", "## L40S placement classes (multi-GPU default runs)", ""]
        lines += ["| Placement | Trials | Mean throughput vs condition mean |", "|---|---|---|"]
        for name, (count, mean_ratio) in sorted(placement.items()):
            lines.append(f"| {name} | {count} | {mean_ratio:+.1%} |")
    stragglers = [
        c for c in conditions if _float(c["straggler_ratio_p95_p50"]) and float(c["straggler_ratio_p95_p50"]) > 1.25
    ]
    if stragglers:
        lines += ["", "## Straggler flags (p95/p50 step time > 1.25)", ""]
        for c in stragglers:
            lines.append(f"- {c['experiment_id']}@{c['platform_id']}: {float(c['straggler_ratio_p95_p50']):.2f}")
    lines.append("")
    return "\n".join(lines)


def _condition(platform: str, experiment_id: str, trials: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [t for t in trials if t.get("status") == "completed"]
    failed = [t for t in trials if t.get("status") != "completed"]
    first = (completed or trials)[0]
    p = {
        k.removeprefix("parameter_"): v for k, v in first.items() if k.startswith("parameter_") and v not in (None, "")
    }
    gpus = int(float(first.get("gpus") or 0))
    batch, seq = int(float(p.get("per_gpu_batch_size", 0))), int(float(p.get("sequence_length", 0)))
    accumulation = str(int(float(p.get("gradient_accumulation_steps", 1))))
    stack = str(first.get("strategy") or p.get("stack") or "")
    if stack == "fsdp":
        stack = f"fsdp:{p.get('strategy_detail', '')}:{p.get('fsdp_wrap_policy', '')}"
    ddp = (
        ",".join(
            f"{key.removeprefix('ddp_')}={p[key]}"
            for key in ("ddp_bucket_cap_mb", "ddp_comm_hook", "ddp_gradient_as_bucket_view", "ddp_static_graph")
            if key in p and not (key == "ddp_comm_hook" and p[key] == "none")
        )
        or "default"
    )
    precision = str(first.get("precision") or "")
    if p.get("precision_backend") == "transformer_engine":
        precision_contract = f"transformer_engine_{precision}"
    else:
        precision_contract = "fp32_params_bf16_autocast" if precision == "bf16" else f"fp32_params_{precision}"
    input_contract = f"wikitext103:{p['text_batching']}" if p.get("text_batching") else "synthetic"
    environment = _environment(first.get("runtime_environment"))
    throughputs = [_float(t.get("throughput_tokens_sec")) for t in completed]
    mean_tps = _mean(throughputs)
    low, high = _ci95(throughputs)
    p50 = _mean([_float(t.get("metric_step_time_p50_ms")) for t in completed])
    p95 = _mean([_float(t.get("metric_step_time_p95_ms")) for t in completed])
    memory = _max([_float(t.get("memory_used_gb")) for t in completed])
    capacity = CAPACITY_GB.get(platform)
    fraction = memory / capacity if memory is not None and capacity else None
    is_default = ddp == "default" and stack == "ddp" and environment == "default"
    return {
        "platform_id": platform,
        "experiment_id": experiment_id,
        "workload": first.get("workload", ""),
        "gpus": gpus,
        "per_gpu_batch_size": batch,
        "sequence_length": seq,
        "tokens_per_gpu_micro_step": batch * seq,
        "gradient_accumulation_steps": accumulation,
        "global_batch_size": p.get("global_batch_size", ""),
        "stack": stack,
        "ddp_settings": ddp,
        "precision_contract": precision_contract,
        "input_contract": input_contract,
        "runtime_environment": environment,
        "completed_trials": len(completed),
        "failed_trials": len(failed),
        "oom_trials": sum("out of memory" in str(t.get("error", "")).lower() for t in failed),
        "mean_throughput_tokens_sec": _round(mean_tps),
        "throughput_ci95_low": _round(low),
        "throughput_ci95_high": _round(high),
        "throughput_cv_pct": _round(_cv(throughputs)),
        "step_time_p50_ms": _round(p50),
        "step_time_p95_ms": _round(p95),
        "straggler_ratio_p95_p50": _round(p95 / p50 if p50 and p95 else None),
        "gpu_hours_per_million_tokens": _round(gpus * 1e6 / (3600 * mean_tps) if mean_tps else None),
        "peak_memory_gb": _round(memory),
        "memory_fraction_of_capacity": _round(fraction),
        "memory_rule_status": "" if fraction is None else ("over_rule" if fraction > MEMORY_RULE else "within_rule"),
        "padding_fraction": _round(_mean([_float(t.get("metric_padding_fraction")) for t in completed])),
        "rank_token_imbalance": _round(_mean([_float(t.get("metric_rank_token_imbalance")) for t in completed])),
        "placement_classes": ";".join(
            f"{name}={count}" for name, count in sorted(Counter(placement_class(t) for t in completed).items())
        ),
        "_is_default": is_default,
        "_throughputs": [v for v in throughputs if v is not None],
        "_trial_rows": completed,
        # The 1-GPU reference runs the same per-GPU work, numerics and inputs with default DDP.
        "_baseline_key": (first.get("workload"), batch, seq, accumulation, precision_contract, input_contract),
    }


def _attach_baseline(condition: dict[str, Any], baseline: dict[str, Any] | None) -> None:
    condition.update(
        {
            "baseline_experiment_id": "",
            "scaling_efficiency": "",
            "efficiency_ci95_low": "",
            "efficiency_ci95_high": "",
            "exposed_communication_ms": "",
            "communication_fraction": "",
        }
    )
    if baseline is None or not baseline["mean_throughput_tokens_sec"]:
        return
    gpus = condition["gpus"]
    base = float(baseline["mean_throughput_tokens_sec"])
    condition["baseline_experiment_id"] = baseline["experiment_id"]
    if condition["mean_throughput_tokens_sec"]:
        condition["scaling_efficiency"] = _round(float(condition["mean_throughput_tokens_sec"]) / (gpus * base))
    # Conservative interval: pair each throughput bound with the opposite baseline bound.
    if condition["throughput_ci95_low"] not in ("", None) and baseline["throughput_ci95_high"] not in ("", None):
        condition["efficiency_ci95_low"] = _round(
            float(condition["throughput_ci95_low"]) / (gpus * float(baseline["throughput_ci95_high"]))
        )
        condition["efficiency_ci95_high"] = _round(
            float(condition["throughput_ci95_high"]) / (gpus * float(baseline["throughput_ci95_low"]))
        )
    step, base_step = _float(condition["step_time_p50_ms"]), _float(baseline["step_time_p50_ms"])
    if step and base_step and gpus > 1:
        exposed = max(0.0, step - base_step)
        condition["exposed_communication_ms"] = _round(exposed)
        condition["communication_fraction"] = _round(exposed / step)


def _field_effects(conditions, lever: str, field: str, control_value: str, treatments) -> list[dict[str, Any]]:
    shared = [k for k in _identity_fields() if k != field]
    controls = {_key(c, shared): c for c in conditions if str(c[field]) == control_value}
    effects = []
    for treatment in conditions:
        value = str(treatment[field])
        if value == control_value or (treatments is not None and value not in treatments):
            continue
        control = controls.get(_key(treatment, shared))
        if control is not None:
            effects.append(_effect(lever, control, treatment, str(control[field]), str(treatment[field])))
    return effects


def _split_effects(conditions) -> list[dict[str, Any]]:
    """Same tokens per GPU, different batch x sequence split; control is the longest sequence."""
    shared = [k for k in _identity_fields() if k not in {"per_gpu_batch_size", "sequence_length"}]
    shared.append("tokens_per_gpu_micro_step")
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for c in conditions:
        groups[_key(c, shared)].append(c)
    effects = []
    for members in groups.values():
        if len(members) < 2:
            continue
        control = max(members, key=lambda c: c["sequence_length"])
        for treatment in members:
            if treatment is not control:
                effects.append(
                    _effect(
                        "batch_sequence_split",
                        control,
                        treatment,
                        f"{control['per_gpu_batch_size']}x{control['sequence_length']}",
                        f"{treatment['per_gpu_batch_size']}x{treatment['sequence_length']}",
                    )
                )
    return effects


def _identity_fields() -> list[str]:
    return [
        "platform_id",
        "workload",
        "gpus",
        "per_gpu_batch_size",
        "sequence_length",
        "gradient_accumulation_steps",
        "stack",
        "ddp_settings",
        "precision_contract",
        "input_contract",
        "runtime_environment",
    ]


def _key(condition: dict[str, Any], fields: list[str]) -> tuple:
    return tuple(str(condition[f]) for f in fields)


def _effect(
    lever: str, control: dict[str, Any], treatment: dict[str, Any], control_setting: str, treatment_setting: str
):
    change, low, high = _relative_change(control["_throughputs"], treatment["_throughputs"])
    efficiency_delta = None
    if control["scaling_efficiency"] not in ("", None) and treatment["scaling_efficiency"] not in ("", None):
        efficiency_delta = 100 * (float(treatment["scaling_efficiency"]) - float(control["scaling_efficiency"]))
    return {
        "lever": lever,
        "platform_id": control["platform_id"],
        "workload": control["workload"],
        "gpus": control["gpus"],
        "control_experiment_id": control["experiment_id"],
        "treatment_experiment_id": treatment["experiment_id"],
        "control_setting": control_setting,
        "treatment_setting": treatment_setting,
        "throughput_change_pct": _round(change),
        "change_ci95_low": _round(low),
        "change_ci95_high": _round(high),
        "efficiency_change_points": _round(efficiency_delta),
        "verdict": _verdict(change, low, high),
    }


def _relative_change(control: list[float], treatment: list[float]) -> tuple[float | None, float | None, float | None]:
    """Percent change of means with a Welch 95% interval on the difference."""
    if not control or not treatment:
        return None, None, None
    c_mean, t_mean = statistics.fmean(control), statistics.fmean(treatment)
    if c_mean == 0:
        return None, None, None
    change = 100 * (t_mean / c_mean - 1)
    if len(control) < 2 or len(treatment) < 2:
        return change, None, None
    vc, vt = statistics.variance(control) / len(control), statistics.variance(treatment) / len(treatment)
    se = math.sqrt(vc + vt)
    if se == 0:
        return change, change, change
    dof = (vc + vt) ** 2 / ((vc**2) / (len(control) - 1) + (vt**2) / (len(treatment) - 1))
    margin = _t95(dof) * se / c_mean * 100
    return change, change - margin, change + margin


def _verdict(change, low, high) -> str:
    if change is None or low is None or high is None:
        return "inconclusive"
    if low > 0 and change >= MATERIAL_PCT:
        return "helps"
    if high < 0 and change <= -MATERIAL_PCT:
        return "hurts"
    if low > -MATERIAL_PCT and high < MATERIAL_PCT:
        return "no_material_effect"
    return "inconclusive"


def placement_class(row: dict[str, Any]) -> str:
    """Classify where a trial's ranks ran, from the topology it captured itself."""
    try:
        ranks = json.loads(row.get("metric_training_rank_placement") or "[]")
    except (TypeError, ValueError):
        return "unknown"
    uuids = [str(r.get("gpu_uuid")) for r in ranks if isinstance(r, dict)]
    if _float(row.get("gpus")) == 1 or len(uuids) == 1:
        return "single_gpu"
    if not uuids:
        return "unknown"
    inventory = [
        line.split(",")[0].strip() for line in str(row.get("metric_gpu_pcie_link_inventory") or "").splitlines()
    ]
    topology = _parse_topology(str(row.get("metric_gpu_topology_matrix") or ""))
    if not inventory or not topology or any(u not in inventory for u in uuids):
        return "unknown"
    indices = [inventory.index(u) for u in uuids]
    links = {topology["links"][a][b] for a in indices for b in indices if a != b}
    if all(link.startswith("NV") for link in links):
        return "nvlink"
    if links == {"PIX"}:
        return "same_pcie_switch"
    numa = {topology["numa"][i] for i in indices}
    return "same_socket" if len(numa) == 1 else "cross_socket"


def _parse_topology(text: str) -> dict[str, Any] | None:
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in text.splitlines()]
    header = next((line for line in lines if "GPU0" in line and "CPU Affinity" in line), None)
    if header is None:
        return None
    gpu_count = len(re.findall(r"\bGPU\d+\b", header))
    nic_count = len(re.findall(r"\bNIC\d+\b", header))
    links, numa = [], []
    for line in lines:
        cells = line.split()
        if line is header or not cells or not re.fullmatch(r"GPU\d+", cells[0]):
            continue
        links.append(cells[1 : 1 + gpu_count])
        tail = cells[1 + gpu_count + nic_count :]
        numa.append(tail[1] if len(tail) > 1 else "unknown")
    return {"links": links, "numa": numa} if len(links) == gpu_count else None


def _placement_summary(conditions) -> dict[str, tuple[int, float]]:
    ratios: dict[str, list[float]] = defaultdict(list)
    for c in conditions:
        if c["platform_id"] != "l40s" or c["gpus"] < 2 or not c["_is_default"] or not c["mean_throughput_tokens_sec"]:
            continue
        mean = float(c["mean_throughput_tokens_sec"])
        for trial in c["_trial_rows"]:
            value = _float(trial.get("throughput_tokens_sec"))
            if value:
                ratios[placement_class(trial)].append(value / mean - 1)
    return {name: (len(values), statistics.fmean(values)) for name, values in ratios.items()}


def _environment(value: Any) -> str:
    try:
        environment = json.loads(value) if isinstance(value, str) and value else {}
    except ValueError:
        environment = {}
    relevant = {k: v for k, v in sorted(environment.items()) if k not in {"OMP_NUM_THREADS"}}
    return ",".join(f"{k}={v}" for k, v in relevant.items()) or "default"


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values) -> float | None:
    numbers = [v for v in values if v is not None]
    return statistics.fmean(numbers) if numbers else None


def _max(values) -> float | None:
    numbers = [v for v in values if v is not None]
    return max(numbers) if numbers else None


def _cv(values) -> float | None:
    numbers = [v for v in values if v is not None]
    if len(numbers) < 2 or not statistics.fmean(numbers):
        return None
    return 100 * statistics.stdev(numbers) / statistics.fmean(numbers)


def _ci95(values) -> tuple[float | None, float | None]:
    numbers = [v for v in values if v is not None]
    if len(numbers) < 2:
        return (numbers[0], numbers[0]) if numbers else (None, None)
    mean = statistics.fmean(numbers)
    margin = _t95(len(numbers) - 1) * statistics.stdev(numbers) / math.sqrt(len(numbers))
    return mean - margin, mean + margin


def _t95(dof: float) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 8: 2.306, 10: 2.228, 20: 2.086, 30: 2.042}
    for limit in sorted(table):
        if dof <= limit:
            return table[limit]
    return 1.96


def _round(value: float | None, digits: int = 6) -> float | str:
    return "" if value is None else round(value, digits)
