"""Reproducible tidy CSV/JSON datasets for R and JavaScript plotting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from hpc_ai_perf.analysis.scaling_deep_dive import (
    CONDITION_COLUMNS,
    EFFECT_COLUMNS,
    _t95,
    build_scaling_deep_dive_conditions,
    build_scaling_deep_dive_effects,
)
from hpc_ai_perf.config.platform import load_platform_conditions
from hpc_ai_perf.results import verify_result_evidence
from hpc_ai_perf.results.aggregate import flatten_result_record

METRICS = (
    "throughput_tokens_sec",
    "throughput_samples_sec",
    "energy_j_measured_region",
    "joules_per_output_token",
    "joules_per_processed_token",
    "latency_p50_ms",
    "latency_p95_ms",
    "ttft_p95_ms",
    "time_per_output_token_p95_ms",
    "decode_time_p95_ms",
    "peak_memory_gib",
    "measured_seconds",
    "runtime_seconds",
    "gpu_utilization_pct",
)
PARAMETERS = (
    "batch_size",
    "activation_checkpointing",
    "num_workers",
    "pinned_memory",
    "prefetch_factor",
    "sequence_length",
    "gradient_accumulation_steps",
    "prompt_length",
    "concurrent_requests",
    "max_new_tokens",
    "per_gpu_batch_size",
    "global_batch_size",
    "tensor_parallel_size",
    "replica_count",
    "max_model_len",
    "enable_prefix_caching",
    "ddp_bucket_cap_mb",
    "ddp_comm_hook",
    "ddp_gradient_as_bucket_view",
    "ddp_static_graph",
    "model_load_dtype",
    "precision_backend",
    "attention_backend",
    "input_protocol",
    "text_batching",
    "strategy_detail",
    "fsdp_wrap_policy",
)
RENAME = {
    "prompt_length": "prompt_tokens",
    "concurrent_requests": "concurrency",
    "max_new_tokens": "output_tokens_requested",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def number(value):
    return value if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) else None


def role(record):
    c = record.get("config") or {}
    p = c.get("workload", {}).get("parameters", {})
    meta = c.get("metadata", {})
    description = " ".join(str(meta.get(k, "")) for k in ("platform_wave", "phase"))
    if p.get("scaling_role") == "canary" or any(word in description.lower() for word in ("canary", "acceptance")):
        return "canary"
    if any(word in description.lower() for word in ("diagnostic", "environment validation", "pilot")):
        return "diagnostic"
    return "performance"


def runtime_identity(record):
    env, m = record.get("environment") or {}, record.get("metrics") or {}
    fingerprints = {
        k: v
        for k, v in m.items()
        if k in ("serving_runtime_sha256", "scaling_runtime_sha256", "training_update_runtime_sha256")
    }
    versions = {k: v for k, v in env.items() if k.endswith("_version")}
    return digest(
        {
            "fingerprints": fingerprints,
            "versions": versions,
            "source_commit_fallback": record.get("git_commit") if not fingerprints else None,
        }
    )


def normalize(record, path, source_root):
    c, m = record.get("config") or {}, record.get("metrics") or {}
    workload = c.get("workload", {})
    p, meta = workload.get("parameters", {}), c.get("metadata", {})
    env = record.get("environment") or {}
    runtime = runtime_identity(record)
    # Bundle-specific config hashes deliberately keep separate submission cohorts apart.
    config_id = record.get("config_sha256") or digest(c)
    row = dict(
        run_id=record["run_id"],
        experiment_id=record["experiment_id"],
        platform=record.get("platform_id") or path.parent.parent.name,
        repetition=record.get("repetition_number"),
        status=record.get("status"),
        error=record.get("error"),
        role=role(record),
        mode=record.get("mode"),
        workload=record.get("workload"),
        model=record.get("model") or workload.get("model"),
        model_revision=record.get("model_revision") or workload.get("model_revision"),
        dataset_revision=record.get("dataset_revision"),
        dataset=workload.get("dataset"),
        seed=c.get("runtime", {}).get("seed"),
        warmup_iterations=record.get("warmup_iterations"),
        measured_iterations=record.get("measured_iterations"),
        study=meta.get("study_family") or p.get("scaling_study_id"),
        wave=meta.get("platform_wave"),
        comparison_scope=p.get("comparison_scope") or meta.get("comparison_scope"),
        gpus=record.get("gpus"),
        precision=record.get("precision"),
        comparability=record.get("comparability") or meta.get("comparability"),
        throughput_unit=record.get("throughput_unit"),
        scaling_protocol_recorded=p.get("scaling_type"),
        strategy=c.get("distributed", {}).get("strategy"),
        serving_layout=p.get("serving_layout"),
        gpu_model=c.get("hardware", {}).get("gpu_model"),
        source_commit=record.get("git_commit"),
        git_dirty=record.get("git_dirty", env.get("git_dirty")),
        config_sha256=config_id,
        runtime_cohort_id=runtime,
        submission_cohort_id=digest([config_id, runtime]),
        started_at=record.get("started_at"),
        completed_at=record.get("completed_at"),
        source_record=path.relative_to(source_root).as_posix(),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        study_design_note=meta.get("study_design_note"),
        motivating_experiment_ids=";".join(meta.get("motivating_experiment_ids", [])) or None,
        follow_up_planned_on=meta.get("follow_up_planned_on"),
        planned_after_results_commit=meta.get("planned_after_results_commit"),
        torch_version=env.get("torch_version"),
        vllm_version=env.get("vllm_version"),
        cuda_version=env.get("cuda_version"),
    )
    for key in PARAMETERS:
        row[RENAME.get(key, key)] = p.get(key)
    for key in METRICS:
        row[key] = number(m.get(key, record.get(key)))
    row.update(
        energy_j_measured_region=number(m.get("energy_joules_measured_region")),
        ttft_p95_ms=number(m.get("time_to_first_token_p95_ms")),
        peak_memory_gib=number(m.get("memory_used_gb", record.get("memory_used_gb"))),
        gpu_utilization_pct=number(m.get("gpu_utilization", record.get("gpu_utilization"))),
        energy_status=m.get("energy_measurement_status"),
        telemetry_status=m.get("nvidia_smi.measurement_status"),
        generated_tokens=number(m.get("generated_tokens")),
        processed_tokens=number(m.get("total_tokens")) if record.get("mode") == "training" else None,
        requests=number(m.get("requests")),
    )
    if row["energy_status"] not in (None, "sufficient"):
        row["energy_j_measured_region"] = None
    if row["energy_j_measured_region"] is not None and row["generated_tokens"] and row["generated_tokens"] > 0:
        row["joules_per_output_token"] = row["energy_j_measured_region"] / row["generated_tokens"]
    row["energy_scope"] = "measured_region" if row["energy_j_measured_region"] is not None else None
    if row["energy_j_measured_region"] is not None and row["processed_tokens"] and row["processed_tokens"] > 0:
        row["joules_per_processed_token"] = row["energy_j_measured_region"] / row["processed_tokens"]
    return row


def select_cohorts(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[(row["platform"], row["experiment_id"])][row["submission_cohort_id"]].append(row)
    for cohorts in groups.values():
        latest = max(cohorts, key=lambda key: max(r["completed_at"] or r["started_at"] or "" for r in cohorts[key]))
        for key, members in cohorts.items():
            for row in members:
                row["selected_current_cohort"] = key == latest
                row["analysis_eligible"] = (
                    key == latest and row["status"] == "completed" and not row["error"] and row["git_dirty"] is not True
                )
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["analysis_eligible"] and row["role"] == "performance":
            groups[(row["platform"], row["experiment_id"], row["submission_cohort_id"])].append(row)
    summaries = []
    for members in groups.values():
        first = members[0]
        excluded = {"run_id", "repetition", "source_record", "source_sha256", "started_at", "completed_at", "error"}
        result = {k: v for k, v in first.items() if k not in excluded and k not in METRICS}
        result.update(trial_count=len(members), source_run_ids=";".join(r["run_id"] for r in members))
        for metric in METRICS:
            values = [r[metric] for r in members if r.get(metric) is not None]
            n = len(values)
            mean = statistics.fmean(values) if n else None
            sd = statistics.stdev(values) if n > 1 else None
            margin = _t95(n - 1) * sd / math.sqrt(n) if sd is not None else None
            result.update(
                {
                    f"{metric}_n": n,
                    f"{metric}_mean": mean,
                    f"{metric}_sd": sd,
                    f"{metric}_ci95_low": mean - margin if margin is not None else None,
                    f"{metric}_ci95_high": mean + margin if margin is not None else None,
                }
            )
        summaries.append(result)
    return summaries


def training_tables(selected, raw):
    cohorts = defaultdict(list)
    for row in selected:
        if row["mode"] == "training" and row["study"] == "scaling_deep_dive_v1":
            cohorts[(row["platform"], row["runtime_cohort_id"])].append(row)
    conditions, effects = [], []
    for (_, runtime), members in cohorts.items():
        flattened = [flatten_result_record(raw[r["source_record"]], r["source_record"]) for r in members]
        cs = build_scaling_deep_dive_conditions(flattened)
        for c in cs:
            conditions.append(
                {
                    **{k: (None if c.get(k) == "" else c.get(k)) for k in CONDITION_COLUMNS},
                    "runtime_cohort_id": runtime,
                    "scaling_basis": "matched_per_gpu_workload",
                }
            )
        for e in build_scaling_deep_dive_effects(cs):
            effects.append(
                {**{k: (None if e.get(k) == "" else e.get(k)) for k in EFFECT_COLUMNS}, "runtime_cohort_id": runtime}
            )
    integer_fields = (
        "gpus",
        "per_gpu_batch_size",
        "sequence_length",
        "tokens_per_gpu_micro_step",
        "gradient_accumulation_steps",
        "global_batch_size",
        "completed_trials",
        "failed_trials",
        "oom_trials",
    )
    for condition in conditions:
        for key in integer_fields:
            if condition.get(key) is not None:
                condition[key] = int(float(condition[key]))
    return conditions, effects


def sample_rows(selected, raw):
    rows = []
    for trial in selected:
        if trial["mode"] != "inference" or int(trial["experiment_id"][1:]) < 715:
            continue
        for metric, values in raw[trial["source_record"]].get("samples", {}).items():
            if not isinstance(values, list):
                continue
            for index, value in enumerate(values):
                if number(value) is None:
                    continue
                rows.append(
                    dict(
                        run_id=trial["run_id"],
                        experiment_id=trial["experiment_id"],
                        platform=trial["platform"],
                        model=trial["model"],
                        serving_layout=trial["serving_layout"],
                        concurrency=trial["concurrency"],
                        runtime_cohort_id=trial["runtime_cohort_id"],
                        metric=metric,
                        sample_index=index,
                        value=value,
                        unit="seconds" if metric.endswith("_seconds") else "unknown",
                    )
                )
    return rows


def coverage(rows, matrix):
    groups = defaultdict(list)
    for row in rows:
        if row["selected_current_cohort"]:
            groups[(row["platform"], row["experiment_id"])].append(row)
    result = []
    for platform in ("l40s", "h100"):
        for c in load_platform_conditions(matrix, platform):
            trials = groups.get((platform, c.matrix_id), [])
            done = sum(r["analysis_eligible"] for r in trials)
            result.append(
                dict(
                    platform=platform,
                    experiment_id=c.matrix_id,
                    wave=c.wave,
                    policy_status=c.status,
                    required_trials=c.required_trials,
                    completed_trials=done,
                    failed_trials=sum(r["status"] != "completed" for r in trials),
                    coverage_status="complete" if done >= c.required_trials else "partial" if done else "missing",
                )
            )
    return result


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def column_unit(name):
    base = name
    for suffix in ("_ci95_low", "_ci95_high", "_mean", "_sd"):
        base = base.removesuffix(suffix)
    if base.endswith("_n") or base in ("trial_count", "requests", "gpus", "repetition"):
        return "count"
    if base.endswith("_ms"):
        return "milliseconds"
    if base.endswith("_seconds"):
        return "seconds"
    if base.endswith("_gib") or base == "peak_memory_gb":
        return "GiB"
    if base in ("joules_per_output_token", "joules_per_processed_token"):
        return "joules/token"
    if base == "energy_j_measured_region":
        return "joules"
    if "throughput_tokens_sec" in base:
        return "tokens/second"
    if "throughput_samples_sec" in base:
        return "samples/second"
    if base in ("scaling_efficiency", "efficiency") or "fraction" in base:
        return "fraction_of_ideal"
    if base.endswith("_pct") or base == "change":
        return "percent"
    if base == "efficiency_change_points":
        return "percentage_points"
    return None


def column_schema(rows, columns):
    schema = {}
    for key in columns:
        values = [row.get(key) for row in rows if row.get(key) is not None]
        types = sorted(
            {"boolean" if isinstance(v, bool) else "number" if isinstance(v, int | float) else "string" for v in values}
        )
        schema[key] = dict(types=types, nullable=any(row.get(key) is None for row in rows), unit=column_unit(key))
    return schema


def write_table(root, name, rows, columns=None):
    rows = clean(rows)
    columns = columns or sorted({k for row in rows for k in row}) or ["experiment_id", "platform"]
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = target.with_suffix(".json"), target.with_suffix(".csv")
    json_path.write_text(json.dumps(rows, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("true" if v is True else "false" if v is False else v) for k, v in row.items()})
    return dict(
        dataset=name,
        rows=len(rows),
        columns=columns,
        schema=column_schema(rows, columns),
        csv=csv_path.relative_to(root).as_posix(),
        json=json_path.relative_to(root).as_posix(),
        csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        json_sha256=hashlib.sha256(json_path.read_bytes()).hexdigest(),
    )


def export_plot_data(source_root: Path, output: Path, matrix: Path):
    source_label = source_root.as_posix()
    source_root, output = source_root.resolve(), output.resolve()
    if output == source_root or source_root in output.parents or output in source_root.parents:
        raise ValueError("Export directory must be separate from raw evidence")
    problems = verify_result_evidence(source_root)
    if problems:
        raise ValueError("Evidence verification failed: " + "; ".join(problems[:10]))
    raw, trials = {}, []
    for platform in ("h100", "l40s"):
        for path in sorted((source_root / platform).glob("E*/*.json")):
            record = json.loads(path.read_text())
            relative = path.relative_to(source_root).as_posix()
            raw[relative] = record
            trials.append(normalize(record, path, source_root))
    select_cohorts(trials)
    selected = [r for r in trials if r["analysis_eligible"] and r["role"] == "performance"]
    summaries = summarize(trials)
    training, effects = training_tables(selected, raw)
    inference = [r for r in summaries if r["mode"] == "inference"]
    recent = [r for r in inference if int(r["experiment_id"][1:]) >= 715]
    qwen = [r for r in recent if r["model"] == "Qwen/Qwen2.5-7B-Instruct"]
    datasets = {
        "trial_level/all_recorded_trials": trials,
        "trial_level/current_performance_trials": selected,
        "trial_level/performance_condition_summary": summaries,
        "training_scaling/deep_dive_condition_metrics": training,
        "training_scaling/scaling_efficiency_curves": [r for r in training if r["scaling_efficiency"] is not None],
        "training_parameter_effects/all_controlled_effects": effects,
        "inference_layout_comparisons/tinyllama_conditions": [r for r in recent if r["workload"] == "tinyllama"],
        "inference_layout_comparisons/qwen7b_conditions": qwen,
        "inference_concurrency/qwen7b_long_prompt_conditions": [r for r in qwen if r["prompt_tokens"] == 1280],
        "energy_latency_tradeoffs/inference_condition_metrics": inference,
        "energy_latency_tradeoffs/training_condition_metrics": [r for r in summaries if r["mode"] == "training"],
        "request_distributions/inference_samples_long": sample_rows(selected, raw),
        "matrix_coverage/platform_condition_coverage": coverage(trials, matrix),
        "provenance/source_records": [
            {
                k: r[k]
                for k in (
                    "source_record",
                    "source_sha256",
                    "run_id",
                    "experiment_id",
                    "platform",
                    "submission_cohort_id",
                    "selected_current_cohort",
                    "analysis_eligible",
                )
            }
            for r in trials
        ],
    }
    for study, start, end in (
        ("gradient_accumulation", 386, 433),
        ("batch_sequence_shapes", 434, 505),
        ("model_size", 506, 533),
        ("ddp_communication", 534, 578),
        ("distributed_strategies", 579, 607),
    ):
        datasets[f"training_parameter_studies/{study}_conditions"] = [
            r for r in training if start <= int(r["experiment_id"][1:]) <= end
        ]
    for lever in sorted({e["lever"] for e in effects}):
        datasets[f"training_parameter_effects/{lever}_effects"] = [e for e in effects if e["lever"] == lever]
    # An interrupted export must not retain an old success manifest.
    (output / "manifest.json").unlink(missing_ok=True)
    tables = [write_table(output, name, rows) for name, rows in datasets.items()]
    manifest = dict(
        schema_version="1.0",
        source_root=source_label,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        execution_matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        source_records=len(trials),
        current_performance_trials=len(selected),
        latest_completed_at=max((r["completed_at"] or "" for r in trials), default=None),
        cohort_policy=(
            "Latest submission cohort per experiment/platform, including failed latest attempts; "
            "no pooling across config/runtime identities."
        ),
        summary_policy=(
            "Unweighted independent-trial means; metric-specific n; approximate Student t 95% intervals; "
            "unavailable intervals are null."
        ),
        datasets=tables,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "data_dictionary.json").write_text(
        json.dumps({t["dataset"]: t["schema"] for t in tables}, indent=2) + "\n"
    )
    return manifest
