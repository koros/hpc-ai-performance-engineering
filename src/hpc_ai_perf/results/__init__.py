"""Experiment result records and writers."""

from hpc_ai_perf.results.aggregate import (
    aggregate_results,
    flatten_result_record,
    load_result_records,
    write_results_csv,
)
from hpc_ai_perf.results.evidence import (
    EvidenceImportSummary,
    import_result_evidence,
    infer_evidence_platform,
    verify_result_evidence,
    write_evidence_manifest,
)
from hpc_ai_perf.results.record import ExperimentResult

__all__ = [
    "ExperimentResult",
    "EvidenceImportSummary",
    "aggregate_results",
    "flatten_result_record",
    "import_result_evidence",
    "infer_evidence_platform",
    "load_result_records",
    "verify_result_evidence",
    "write_evidence_manifest",
    "write_results_csv",
]
