"""Observed-results follow-up: fixed effective batch, different microbatch splits."""

from hpc_ai_perf.config.scaling_deep_dive import Condition, Study, build_config, slug

CONFIG_ROOT = "configs/microbatch_accumulation"
STUDY_ID = "tinyllama_microbatch_accumulation_v1"
CANARY_WAVE = "microbatch-accumulation-canary"
PERFORMANCE_WAVE = "microbatch-accumulation-performance"
PAIRS = ((1, 16), (2, 8), (4, 4))
MOTIVATING_IDS = [f"E{i}" for i in range(418, 434)]


def design():
    study = Study(
        "MB",
        "microbatch-accumulation",
        "Fixed effective batch microbatch split",
        "Measure microbatch versus accumulation at matched effective batch per GPU",
        ("RQ1", "RQ4", "RQ5", "RQ7"),
    )
    specs = [(True, b, a, g) for b, a in PAIRS for g in (1, 8)]
    specs += [(False, b, a, g) for b, a in PAIRS for g in (1, 2, 4, 8)]
    rows = []
    for number, (canary, batch, accumulation, gpus) in enumerate(specs, 773):
        eid = f"E{number}"
        wave = CANARY_WAVE if canary else PERFORMANCE_WAVE
        c = Condition(
            "tinyllama",
            gpus,
            batch,
            1024,
            accumulation,
            platforms=("l40s",),
            role="canary" if canary else "performance",
            updates=20 if canary else 200,
        )
        data = build_config(eid, c, [study])
        data["experiment"] = f"{STUDY_ID}_{slug(c)}"
        data["workload"]["parameters"].update(
            scaling_study_id=STUDY_ID,
            scaling_type="weak",
            fixed_workload=False,
            comparison_scope=STUDY_ID,
        )
        data["runtime"]["environment"].update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        data["metadata"].update(
            phase="Microbatch accumulation follow-up",
            study_family=STUDY_ID,
            platform_wave=wave,
            comparison_scope=STUDY_ID,
            study_design_note="follow-up motivated by observed results",
            study_design="exploratory_sequential_follow_up",
            motivating_experiment_ids=MOTIVATING_IDS,
            follow_up_planned_on="2026-10-21",
            planned_after_results_commit="11bdd72",
            effective_batch_per_gpu=16,
            historical_control_ids=[f"E{i}" for i in range(426, 430)],
            historical_control_reuse="context only: historical runtime fingerprints differ",
        )
        rows.append((eid, wave, data))
    return rows
