from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from hpc_ai_perf.cli import _submit_platform_wave
from hpc_ai_perf.config import load_experiment_config
from hpc_ai_perf.config.inference_crossover import CANARY_WAVE, design, serving_runtime_sha256
from hpc_ai_perf.config.platform import load_platform_conditions
from hpc_ai_perf.planning import plan_serving_layout
from hpc_ai_perf.validation import capability_issues
from hpc_ai_perf.validation.inference_crossover import canary_issues
from hpc_ai_perf.workloads.vllm_router import build_gpu_groups, route_prompt_indices


def config(eid):
    return load_experiment_config(next(Path("configs/inference_crossover").glob(f"{eid}_*.json")))


def test_matrix_has_balanced_equal_budget_contrasts_and_no_duplicate_routes():
    rows = design()
    assert len(rows) == 35
    assert len({eid for eid, _, _, _ in rows}) == 35
    for platform, expected in (("l40s", 4), ("h100", 3)):
        budgets = 8 if platform == "l40s" else 4
        relevant = [(eid, wave, data) for eid, p, wave, data in rows if p == platform]
        assert sum(wave == CANARY_WAVE for _, wave, _ in relevant) == expected
        cells = Counter()
        for eid, wave, data in relevant:
            c = config(eid)
            assert capability_issues(c) == []
            assert plan_serving_layout(c)["valid"]
            assert c.hardware.gpus == budgets
            if wave != CANARY_WAVE:
                p = data["workload"]["parameters"]
                cells[(p["prompt_length"], p["concurrent_requests"])] += 1
        assert set(cells.values()) == {expected}
        conditions = [
            c
            for c in load_platform_conditions("platforms/execution_matrix.json", platform)
            if c.wave.startswith("inference-crossover-")
        ]
        assert len(conditions) == expected * 5
        assert len({c.matrix_id for c in conditions}) == len(conditions)


@pytest.mark.parametrize("tp,dp", [(1, 8), (2, 4), (4, 2), (1, 4), (2, 2)])
def test_replica_groups_are_disjoint_and_route_every_prompt_once(tp, dp):
    groups = build_gpu_groups(gpu_count=tp * dp, tensor_parallel_size=tp, replica_count=dp)
    assert len(groups) == dp
    assert len({gpu for group in groups for gpu in group}) == tp * dp
    routed = route_prompt_indices(8, dp)
    assert sorted(index for group in routed for index in group) == list(range(8))
    assert len({len(group) for group in routed}) == 1


def test_duplicate_visible_devices_cannot_launch_overlapping_replicas():
    with pytest.raises(ValueError, match="unique GPU"):
        build_gpu_groups(gpu_count=4, tensor_parallel_size=1, replica_count=4, visible_devices="0,0,1,2")


def test_canary_requires_current_runtime_fixed_work_and_balanced_routing():
    c = config("E715")
    record = {
        "status": "completed",
        "error": None,
        "platform_id": "l40s",
        "gpus": 8,
        "git_dirty": False,
        "environment": {"git_dirty": False},
        "metrics": {
            "serving_runtime_sha256": serving_runtime_sha256(),
            "serving_layout_observed": "tp1_dp8",
            "tensor_parallel_size_observed": 1,
            "replica_count_observed": 8,
            "requests": 16,
            "generated_tokens": 2048,
            "throughput_tokens_sec": 100,
            "time_to_first_token_p95_ms": 20,
            "time_per_output_token_p95_ms": 5,
            "memory_used_gb": 38,
            "replica_request_count_total": 16,
            "replica_request_count_imbalance_pct": 0,
        },
    }
    assert canary_issues(record, c, "l40s") == []
    record["metrics"]["serving_runtime_sha256"] = "stale"
    record["metrics"]["generated_tokens"] = 1000
    assert set(canary_issues(record, c, "l40s")) == {"runtime", "fixed output work"}


def test_performance_wave_cannot_create_a_bundle_before_acceptance(monkeypatch):
    from hpc_ai_perf.validation import inference_crossover

    def rejected(*args):
        raise ValueError("current canaries absent")

    monkeypatch.setattr(inference_crossover, "verify_inference_canaries", rejected)
    args = SimpleNamespace(
        wave="inference-crossover-short", platform_profile="platforms/l40s.json", venv_activate="/certified/activate"
    )
    with pytest.raises(ValueError, match="current canaries absent"):
        _submit_platform_wave(args)


def test_replica_worker_honours_fixed_output_sampling(monkeypatch):
    import sys

    from hpc_ai_perf.workloads.vllm_router import _vllm_replica_worker

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "0")
    observed = []

    class Engine:
        def __init__(self, **kwargs):
            self.llm_engine = None

        def get_tokenizer(self):
            return SimpleNamespace(vocab_size=100)

        def generate(self, prompts, sampling_params, **kwargs):
            observed.append(sampling_params)
            return []

    class Connection:
        def __init__(self):
            self.messages = iter(
                [
                    {"kind": "generate", "prompts": [], "max_new_tokens": 128, "ignore_eos": True},
                    {"kind": "shutdown"},
                ]
            )

        def recv(self):
            return next(self.messages)

        def send(self, message):
            pass

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Engine, SamplingParams=lambda **kwargs: kwargs))
    _vllm_replica_worker(Connection(), ("0",), "model", "revision", "bfloat16", 1, {})
    assert observed == [{"temperature": 0.0, "max_tokens": 128, "ignore_eos": True}]


def test_analysis_separates_acceptance_legacy_and_controlled_trials():
    from hpc_ai_perf.analysis.summaries import build_inference_tp_crossover_summary

    base = {
        "status": "completed",
        "mode": "inference",
        "phase": "Inference layout crossover",
        "platform_id": "l40s",
        "comparability": "exact",
        "workload": "tinyllama",
        "model": "model",
        "precision": "bf16",
        "framework": "vllm",
        "gpus": "8",
        "batch_size": "1",
        "parameter_tensor_parallel_size": "1",
        "parameter_serving_layout": "tp1_dp8",
        "parameter_replica_count": "8",
        "parameter_data_parallel_size": "8",
        "parameter_prompt_length": "128",
        "parameter_max_new_tokens": "128",
        "parameter_concurrent_requests": "8",
        "parameter_comparison_scope": "new-controlled-study",
        "condition_id": "E726@l40s",
    }
    rows = [{**base, "throughput_tokens_sec": str(value)} for value in (10, 11, 12)]
    rows.append({**base, "phase": "Inference layout acceptance", "throughput_tokens_sec": "10000"})
    rows.append({**base, "parameter_comparison_scope": "", "throughput_tokens_sec": "999"})
    summary = build_inference_tp_crossover_summary(rows)
    assert len(summary) == 2
    controlled = next(row for row in summary if row["comparison_scope"] == "new-controlled-study")
    assert controlled["trials"] == 3
    assert controlled["avg_throughput_tokens_sec"] == 11
    assert controlled["throughput_ci95_low"] < 11 < controlled["throughput_ci95_high"]


def collected_resolved_canaries(root, platform):
    """Build bundle-resolved records; schema/provenance validation is tested separately."""
    import hashlib
    import json

    from hpc_ai_perf.config import discover_experiment_configs
    from hpc_ai_perf.config.platform import load_platform_profile, resolve_platform_config
    from hpc_ai_perf.validation.qwen7b_training import canary_contract_sha256

    entries = {e.matrix_id: e for e in discover_experiment_configs("configs/inference_crossover")}
    profile = load_platform_profile(f"platforms/{platform}.json")
    paths = []
    sums = []
    for condition in load_platform_conditions("platforms/execution_matrix.json", platform, wave=CANARY_WAVE):
        entry = entries[condition.matrix_id]
        resolved = resolve_platform_config(entry, profile, condition)
        # This is the actual mismatch: the submitted resolved contract differs from the source config.
        assert canary_contract_sha256(resolved) != canary_contract_sha256(entry.config)
        payload = resolved.to_dict()
        payload["metadata"]["source_config_path"] = f"/frozen/bundle/repo/configs/{condition.matrix_id}.json"
        params = resolved.workload.parameters
        requests = resolved.runtime.max_steps * params["concurrent_requests"]
        record = {
            "status": "completed",
            "error": None,
            "config": payload,
            "platform_id": platform,
            "gpus": resolved.hardware.gpus,
            "git_dirty": False,
            "environment": {"git_dirty": False},
            "metrics": {
                "serving_runtime_sha256": serving_runtime_sha256(),
                "serving_layout_observed": params["serving_layout"],
                "tensor_parallel_size_observed": params["tensor_parallel_size"],
                "replica_count_observed": params["replica_count"],
                "requests": requests,
                "generated_tokens": requests * params["max_new_tokens"],
                "throughput_tokens_sec": 100,
                "time_to_first_token_p95_ms": 20,
                "time_per_output_token_p95_ms": 5,
                "memory_used_gb": 6,
                "replica_request_count_total": requests,
                "replica_request_count_imbalance_pct": 0,
            },
        }
        path = root / condition.matrix_id / "run.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record))
        paths.append(path)
        sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}\n")
    (root / "SHA256SUMS").write_text("".join(sums))
    return paths


@pytest.mark.parametrize("platform,count", [("h100", 3), ("l40s", 4)])
def test_collected_bundle_resolved_canaries_match_current_resolved_contract(tmp_path, monkeypatch, platform, count):
    from hpc_ai_perf.validation import inference_crossover as gate

    collected_resolved_canaries(tmp_path, platform)
    monkeypatch.setattr(gate, "validate_result_file", lambda path: [])
    assert len(gate.verify_inference_canaries(tmp_path, platform)) == count


@pytest.mark.parametrize("failure", ["checksum", "runtime", "contract", "TTFT", "evidence"])
def test_resolved_canary_fix_keeps_acceptance_checks_strict(tmp_path, monkeypatch, failure):
    import hashlib
    import json

    from hpc_ai_perf.validation import inference_crossover as gate

    paths = collected_resolved_canaries(tmp_path, "h100")
    monkeypatch.setattr(gate, "validate_result_file", lambda path: [])
    record = json.loads(paths[0].read_text())
    expected = failure
    if failure == "runtime":
        record["metrics"]["serving_runtime_sha256"] = "stale"
    elif failure == "contract":
        record["config"]["workload"]["parameters"]["prompt_length"] += 1
        expected = "resolved configuration contract"
    elif failure == "TTFT":
        record["metrics"]["time_to_first_token_p95_ms"] = None
    elif failure == "evidence":
        monkeypatch.setattr(gate, "validate_result_file", lambda path: ["invalid evidence"] if path == paths[0] else [])
    else:
        record["status"] = "failed"
    paths[0].write_text(json.dumps(record))
    if failure != "checksum":
        (tmp_path / "SHA256SUMS").write_text(
            "".join(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path).as_posix()}\n"
                for path in paths
            )
        )
    with pytest.raises(ValueError, match=expected):
        gate.verify_inference_canaries(tmp_path, "h100")
