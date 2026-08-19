from __future__ import annotations

from typing import Any

RESEARCH_QUESTIONS = {f"RQ{index}" for index in range(1, 8)}
ANALYSIS_TASKS = {f"A{index:02d}" for index in range(1, 9)}

# Phase-level mappings keep the 200 source configs concise while ensuring the
# normalized config embedded in every result has explicit research traceability.
PHASE_TRACEABILITY: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "Environment validation": (
        ("RQ1", "RQ2", "RQ3", "RQ4", "RQ5", "RQ6", "RQ7"),
        ("A01", "A02", "A03", "A04", "A05", "A06", "A07"),
    ),
    "Baseline profiling": (("RQ5", "RQ7"), ("A01", "A02", "A04", "A05", "A07")),
    "Workload characterisation": (("RQ4", "RQ5", "RQ7"), ("A01", "A02", "A03", "A04", "A05", "A07")),
    "Strong scaling": (("RQ1", "RQ5", "RQ7"), ("A01", "A03", "A04", "A05", "A07")),
    "Weak scaling": (("RQ1", "RQ5", "RQ7"), ("A01", "A03", "A04", "A05", "A07")),
    "Distributed training strategy": (
        ("RQ2", "RQ5", "RQ7"),
        ("A01", "A03", "A04", "A05", "A07"),
    ),
    "Memory optimisation": (("RQ2", "RQ5", "RQ7"), ("A01", "A03", "A04", "A05", "A07")),
    "Precision study": (("RQ3", "RQ5", "RQ7"), ("A01", "A02", "A04", "A05", "A07")),
    "Data movement": (("RQ4", "RQ5", "RQ7"), ("A03", "A04", "A05", "A07")),
    "Communication analysis": (("RQ4", "RQ5", "RQ7"), ("A03", "A04", "A05", "A07")),
    "Hardware comparison": (("RQ5", "RQ6", "RQ7"), ("A01", "A02", "A04", "A05", "A06", "A07")),
    "Inference framework": (("RQ5", "RQ7"), ("A02", "A03", "A04", "A05", "A07")),
    "Inference batching": (("RQ5", "RQ7"), ("A02", "A03", "A04", "A05", "A07")),
    "Inference concurrency": (("RQ5", "RQ7"), ("A02", "A03", "A04", "A05", "A07")),
    "Inference prompt length": (("RQ5", "RQ7"), ("A02", "A03", "A04", "A05", "A07")),
    "Inference multi-GPU": (("RQ1", "RQ5", "RQ7"), ("A02", "A03", "A04", "A05", "A07")),
    "Inference precision/quantization": (("RQ3", "RQ5", "RQ7"), ("A02", "A04", "A05", "A07")),
}


def enrich_matrix_metadata(experiment_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Add canonical phase-level RQ/task mappings to normalized matrix metadata."""
    enriched = dict(metadata)
    if not _is_matrix_experiment_id(experiment_id):
        return enriched
    phase = str(enriched.get("phase", ""))
    traceability = PHASE_TRACEABILITY.get(phase)
    if traceability is None:
        return enriched
    research_questions, analysis_tasks = traceability
    enriched.setdefault("research_questions", list(research_questions))
    enriched.setdefault("analysis_tasks", list(analysis_tasks))
    return enriched


def _is_matrix_experiment_id(experiment_id: str) -> bool:
    return len(experiment_id) == 4 and experiment_id.startswith("E") and experiment_id[1:].isdigit()
