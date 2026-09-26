import json

import pytest

from hpc_ai_perf.config.lowbit import ARTIFACTS, QUALITY_METHOD_IDS, digest, lowbit_issues, runtime_fingerprint
from hpc_ai_perf.validation.lowbit import matrix_fingerprint, resolved_lowbit_config, verify_acceptance


def write_quality_gate(root):
    for method, eid in QUALITY_METHOD_IDS.items():
        cfg = resolved_lowbit_config(eid, "h100")
        assert lowbit_issues(cfg) == []
        metrics = {
            "study_runtime_sha256": runtime_fingerprint(),
            "lowbit_matrix_sha256": matrix_fingerprint("lowbit-quality"),
            "study_parameters_sha256": digest(cfg.workload.parameters),
            "hf_model_revision": cfg.workload.model_revision,
            "precision_backend": ARTIFACTS[method][2] or "vllm_bf16",
            "quantization_method": method,
            "compute_dtype_observed": "torch.bfloat16",
            "runtime_packages": '{"vllm":"0.10.2"}',
            "quality_tokens": 128 * 511,
            "kv_cache_blocks_observed": 1024,
            "latency_sample_count": 16,
            "requests": 16,
            "throughput_tokens_sec": 20,
            "quality_perplexity": 10,
            "latency_p95_ms": 200,
            "time_to_first_token_p95_ms": 50,
            "memory_model_weights_gb": 5,
            "measured_seconds": 30,
            "energy_scope": "measured_region",
            "energy_measurement_status": "sufficient",
            "energy_joules_measured_region": 1000,
            "nvidia_smi.gpu_mapping_method": "cuda_uuid",
            "nvidia_smi.measurement_interior_samples": 100,
            "nvidia_smi.measurement_max_gap_seconds": 0.3,
        }
        (root / f"{method}.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "error": None,
                    "gpus": 1,
                    "environment": {"observed_gpu_model": "NVIDIA H100"},
                    "metrics": metrics,
                }
            )
        )


def test_quality_acceptance_requires_new_coverage_and_nf4_even_without_extra_flag(tmp_path):
    write_quality_gate(tmp_path)
    verify_acceptance(tmp_path, "h100", {"vllm": "0.10.2"}, wave="lowbit-quality")
    with pytest.raises(ValueError, match="fingerprint failed"):
        verify_acceptance(tmp_path, "h100", {"vllm": "0.10.2"})
    path = tmp_path / "bitsandbytes_nf4.json"
    record = json.loads(path.read_text())
    record["metrics"]["quality_tokens"] = 32 * 255
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="quality coverage failed"):
        verify_acceptance(tmp_path, "h100", {"vllm": "0.10.2"}, wave="lowbit-quality")
