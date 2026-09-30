import builtins
import csv
import hashlib
import json
import os
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
from scripts.verify_hardware_energy_revalidation import verify_results as verify_hardware_energy_results

from hpc_ai_perf.analysis import analyze_results, build_analysis_tables
from hpc_ai_perf.analysis.summaries import (
    build_experiment_rankings,
    build_phase_summary,
    build_scaling_summary,
    build_trial_summary,
)
from hpc_ai_perf.cli import _check_slurm_profile, _load_slurm_profile, _refresh_outputs, _render_slurm_bulk
from hpc_ai_perf.config import (
    ProfilingConfig,
    discover_experiment_configs,
    filter_config_entries,
    load_experiment_config,
)
from hpc_ai_perf.profiling import CompositeProfiler
from hpc_ai_perf.profiling.base import Profiler
from hpc_ai_perf.profiling.system import (
    NvidiaSmiProfiler,
    TorchMemoryProfiler,
    TorchProfiler,
    _NvidiaSmiSample,
    build_profiler,
    select_nvidia_smi_rows,
)
from hpc_ai_perf.readiness import write_readiness_report
from hpc_ai_perf.results import ExperimentResult, aggregate_results, audit_telemetry
from hpc_ai_perf.runners import LocalBenchmarkRunner
from hpc_ai_perf.runtime_checks import (
    disable_deepspeed_nvtx_annotations,
    prepare_deepspeed_import_environment,
    run_cluster_runtime_checks,
)
from hpc_ai_perf.slurm import (
    build_job_status_report,
    build_job_status_rows,
    build_submission_plan,
    record_job_submission,
    refresh_job_status_csv,
    render_slurm_script,
    submit_job_scripts,
    write_job_status_csv,
    write_job_status_report,
    write_slurm_script,
    write_submission_plan,
)
from hpc_ai_perf.tracking import track_experiment_progress
from hpc_ai_perf.validation import capability_issues, validate_experiment_configs, validate_result_records
from hpc_ai_perf.workloads import (
    EnvironmentValidationWorkload,
    GPT2SmallWorkload,
    TinyLlamaWorkload,
    TransformerWorkload,
    WorkloadRegistry,
    WorkloadRunOutput,
    default_registry,
)
from hpc_ai_perf.workloads import environment as environment_module
from hpc_ai_perf.workloads import transformer as transformer_module


def test_load_synthetic_config() -> None:
    config = load_experiment_config(Path("configs/local/synthetic_smoke.json"))

    assert config.experiment_id == "local_synthetic_smoke"
    assert config.workload.name == "synthetic"
    assert config.runtime.precision == "fp32"


def test_all_json_configs_load() -> None:
    config_paths = sorted(Path("configs").rglob("*.json"))

    assert config_paths
    for config_path in config_paths:
        config = load_experiment_config(config_path)

        assert config.experiment_id
        assert config.experiment
        assert config.workload.name


def test_discover_matrix_configs() -> None:
    entries = discover_experiment_configs()
    experiment_ids = {entry.config.experiment_id for entry in entries}
    source_ids = {f"E{index:03d}" for index in range(1, 201)}
    supplemental_ids = {f"E{index:03d}" for index in range(201, 305)}

    assert len(entries) == 304
    assert experiment_ids == source_ids | supplemental_ids


def test_filter_config_entries_by_phase_priority_and_workload() -> None:
    entries = discover_experiment_configs()
    filtered = filter_config_entries(
        entries,
        phase="Inference concurrency",
        priority="Must Have",
        workload="gpt2_small",
    )

    assert [entry.config.experiment_id for entry in filtered] == ["E138", "E140", "E141"]


def test_render_slurm_script_from_config() -> None:
    config_path = Path("configs/scaling_studies/E033_gpt2_small_strong_2gpu.json")
    config = load_experiment_config(config_path)

    script = render_slurm_script(config, config_path, partition="gpu", modules=["cuda/12.4"])

    assert "#SBATCH --job-name=hpc-ai-e033" in script
    assert "#SBATCH --gres=gpu:2" in script
    assert "#SBATCH --partition=gpu" in script
    assert "module load cuda/12.4" in script
    assert "python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2" in script
    assert "-m hpc_ai_perf.cli run --config configs/scaling_studies/E033_gpt2_small_strong_2gpu.json" in script


def test_render_slurm_script_with_setup_commands() -> None:
    config_path = Path("configs/environment_validation/E001_cuda_pytorch_1gpu.json")
    config = load_experiment_config(config_path)

    script = render_slurm_script(
        config,
        config_path,
        setup_commands=["module purge", "source .venv/bin/activate"],
        environment={"TOKENIZERS_PARALLELISM": "false"},
    )

    assert "set -euo pipefail\nmodule purge\nsource .venv/bin/activate" in script
    assert "export TOKENIZERS_PARALLELISM=false" in script


def test_render_slurm_script_includes_recorded_runtime_environment() -> None:
    config_path = Path("configs/environment_validation/E001_cuda_pytorch_1gpu.json")
    config = load_experiment_config(config_path)
    config = replace(
        config,
        runtime=replace(config.runtime, environment={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
    )

    script = render_slurm_script(config, config_path)

    assert "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in script


def test_render_slurm_repetition_array() -> None:
    config_path = Path("configs/baseline_profiling/E008_gpt2_small_training_baseline.json")
    config = load_experiment_config(config_path)

    script = render_slurm_script(config, config_path, repetitions=3)

    assert "#SBATCH --array=1-3" in script
    assert "#SBATCH --output=results/raw/%x-%A_%a.out" in script
    assert '--repetition "${SLURM_ARRAY_TASK_ID}"' in script


def test_render_slurm_can_stage_result_output() -> None:
    config_path = Path("configs/baseline_profiling/E008_gpt2_small_training_baseline.json")
    config = load_experiment_config(config_path)

    script = render_slurm_script(
        config,
        config_path,
        repetitions=3,
        run_output_dir="build/platform_results/l40s/wave_1",
        log_dir="slurm/generated/l40s/wave_1/logs",
    )

    assert "#SBATCH --output=slurm/generated/l40s/wave_1/logs/%x-%A_%a.out" in script
    assert "--output-dir build/platform_results/l40s/wave_1" in script
    assert '--repetition "${SLURM_ARRAY_TASK_ID}"' in script


def test_render_slurm_runs_bundle_verification_before_workload() -> None:
    config_path = Path("configs/baseline_profiling/E008_gpt2_small_training_baseline.json")
    config = load_experiment_config(config_path)

    script = render_slurm_script(
        config,
        config_path,
        pre_run_command=(
            "python -m hpc_ai_perf.cli verify-submission-bundle --manifest /bundle/SUBMISSION_MANIFEST.json"
        ),
    )

    assert script.index("verify-submission-bundle") < script.index("python -m hpc_ai_perf.cli run")


def test_write_slurm_script(tmp_path: Path) -> None:
    script_path = write_slurm_script("#!/bin/bash\n", tmp_path, "E001")

    assert script_path == tmp_path / "E001.sbatch"
    assert script_path.read_text(encoding="utf-8") == "#!/bin/bash\n"


def test_build_submission_plan(tmp_path: Path) -> None:
    write_slurm_script("#!/bin/bash\n", tmp_path, "E002")
    write_slurm_script("#!/bin/bash\n", tmp_path, "E001")

    plan = build_submission_plan(tmp_path, batch_name="Environment Validation")

    assert "# Slurm Submission Plan: Environment Validation" in plan
    assert "Total jobs: 2" in plan
    assert "sbatch" in plan
    assert plan.index("E001.sbatch") < plan.index("E002.sbatch")


def test_write_submission_plan(tmp_path: Path) -> None:
    write_slurm_script("#!/bin/bash\n", tmp_path, "E001")

    plan_path = write_submission_plan(tmp_path, tmp_path / "SUBMISSION_PLAN.md")

    assert plan_path == tmp_path / "SUBMISSION_PLAN.md"
    assert "`E001.sbatch`" in plan_path.read_text(encoding="utf-8")


def test_build_job_status_rows(tmp_path: Path) -> None:
    write_slurm_script("#!/bin/bash\n", tmp_path, "E002")
    write_slurm_script("#!/bin/bash\n", tmp_path, "E001")

    rows = build_job_status_rows(tmp_path, batch_name="Environment Validation")

    assert [row["experiment_id"] for row in rows] == ["E001", "E002"]
    assert rows[0]["batch_name"] == "Environment Validation"
    assert rows[0]["state"] == "PLANNED"
    assert rows[0]["job_id"] == ""


def test_record_job_submission_and_report(tmp_path: Path) -> None:
    write_slurm_script("#!/bin/bash\n", tmp_path, "E001")
    status_path = write_job_status_csv(tmp_path, tmp_path / "slurm_status.csv", batch_name="Environment Validation")

    record_job_submission(
        status_path,
        experiment_id="E001",
        sbatch_output="Submitted batch job 123456",
        notes="first smoke submission",
    )

    with status_path.open(encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))

    assert rows[0]["job_id"] == "123456"
    assert rows[0]["state"] == "SUBMITTED"
    assert rows[0]["submitted_at"]
    assert rows[0]["notes"] == "first smoke submission"

    report = build_job_status_report(status_path)

    assert "| SUBMITTED | 1 |" in report
    assert "| E001 | 123456 | SUBMITTED |" in report


def test_write_job_status_report(tmp_path: Path) -> None:
    write_slurm_script("#!/bin/bash\n", tmp_path, "E001")
    status_path = write_job_status_csv(tmp_path, tmp_path / "slurm_status.csv")

    report_path = write_job_status_report(status_path, tmp_path / "slurm_status.md")

    assert report_path == tmp_path / "slurm_status.md"
    assert "# Slurm Job Status" in report_path.read_text(encoding="utf-8")


def test_submit_job_scripts_dry_run(tmp_path: Path) -> None:
    write_slurm_script("#!/bin/bash\n", tmp_path, "E001")
    status_path = write_job_status_csv(tmp_path, tmp_path / "slurm_status.csv")

    submissions = submit_job_scripts(tmp_path, status_path, dry_run=True)

    assert submissions == [
        {
            "experiment_id": "E001",
            "script_path": str(tmp_path / "E001.sbatch"),
            "command": f"sbatch {tmp_path / 'E001.sbatch'}",
            "job_id": "",
            "state": "DRY_RUN",
            "output": "",
        }
    ]
    with status_path.open(encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    assert rows[0]["state"] == "PLANNED"


def test_submit_job_scripts_records_mocked_sbatch_output(tmp_path: Path) -> None:
    write_slurm_script("#!/bin/bash\n", tmp_path, "E001")
    status_path = write_job_status_csv(tmp_path, tmp_path / "slurm_status.csv")

    submissions = submit_job_scripts(
        tmp_path,
        status_path,
        runner=lambda script: "Submitted batch job 123456",
    )

    assert submissions[0]["job_id"] == "123456"
    assert submissions[0]["state"] == "SUBMITTED"
    with status_path.open(encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    assert rows[0]["job_id"] == "123456"
    assert rows[0]["state"] == "SUBMITTED"


def test_refresh_job_status_collapses_array_accounting_states(tmp_path: Path) -> None:
    write_slurm_script("#!/bin/bash\n", tmp_path, "E001")
    status_path = write_job_status_csv(tmp_path, tmp_path / "slurm_status.csv")
    record_job_submission(status_path, experiment_id="E001", job_id="123456")

    rows = refresh_job_status_csv(
        status_path,
        runner=lambda job_id: "\n".join(
            [
                f"{job_id}|COMPLETED",
                f"{job_id}_1|COMPLETED",
                f"{job_id}_2|FAILED",
                f"{job_id}_2.batch|FAILED",
            ]
        ),
    )

    assert rows[0]["state"] == "FAILED"


def test_render_slurm_bulk(tmp_path: Path) -> None:
    _render_slurm_bulk(
        config_root="configs",
        output_dir=str(tmp_path),
        phase="Environment validation",
        priority="Must Have",
        mode=None,
        workload=None,
        experiment_id=None,
        partition="gpu",
        account=None,
        time_limit=None,
        cpus_per_task=8,
        modules=[],
        env_values=[],
        profile_path=None,
    )

    generated = sorted(path.name for path in tmp_path.glob("*.sbatch"))

    assert generated == [
        "E001.sbatch",
        "E002.sbatch",
        "E003.sbatch",
        "E004.sbatch",
        "E006.sbatch",
        "E215.sbatch",
        "E230.sbatch",
    ]


def test_render_slurm_bulk_with_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        """{
  "partition": "gpu",
  "platform": "lovelace_h100",
  "account": "project",
  "cpus_per_task": 12,
  "setup_commands": ["module purge"],
  "modules": ["cuda/12.4"],
  "environment": {"TOKENIZERS_PARALLELISM": "false"}
}
""",
        encoding="utf-8",
    )

    _render_slurm_bulk(
        config_root="configs",
        output_dir=str(tmp_path),
        phase="Environment validation",
        priority="Must Have",
        mode=None,
        workload=None,
        experiment_id="E001",
        partition=None,
        account=None,
        time_limit=None,
        cpus_per_task=None,
        modules=[],
        env_values=[],
        profile_path=str(profile_path),
    )

    script = (tmp_path / "E001.sbatch").read_text(encoding="utf-8")

    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --account=project" in script
    assert "#SBATCH --cpus-per-task=12" in script
    assert "module purge\nmodule load cuda/12.4" in script
    assert "export TOKENIZERS_PARALLELISM=false" in script


def test_render_slurm_bulk_skips_deferred_scope_rows(tmp_path: Path) -> None:
    _render_slurm_bulk(
        config_root="configs",
        output_dir=str(tmp_path),
        phase="Inference framework",
        priority=None,
        mode=None,
        workload=None,
        experiment_id=None,
        partition="gpu",
        account=None,
        time_limit=None,
        cpus_per_task=8,
        modules=[],
        env_values=[],
        profile_path=None,
    )

    generated = sorted(path.stem for path in tmp_path.glob("*.sbatch"))

    assert generated == ["E129", "E130", "E153", "E177", "E178"]


def test_slurm_profile_example_loads() -> None:
    profile = _load_slurm_profile("slurm/profiles/example.json")

    assert profile["partition"] == "gpu"
    assert "cuda/12.4" in profile["modules"]


def test_check_slurm_profile_reports_placeholders() -> None:
    issues = _check_slurm_profile("slurm/profiles/example.json")

    assert any("account" in issue for issue in issues)
    assert any("placeholder" in issue for issue in issues)


def test_check_slurm_profile_accepts_ready_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        """{
  "partition": "gpu",
  "platform": "lovelace_h100",
  "account": "project123",
  "cpus_per_task": 8,
  "setup_commands": ["module purge", "source /path/to/venv/bin/activate"],
  "modules": ["cuda/12.4"],
  "environment": {"TOKENIZERS_PARALLELISM": "false"}
}
""",
        encoding="utf-8",
    )

    assert _check_slurm_profile(str(profile_path)) == []


def test_slurm_smoke_presets_exist() -> None:
    expected_scripts = {
        "distilbert_smoke.sbatch": "configs/local/distilbert_smoke.json",
        "gpt2_small_smoke.sbatch": "configs/local/gpt2_small_smoke.json",
        "tinyllama_inference_smoke.sbatch": "configs/local/tinyllama_inference_smoke.json",
    }

    for script_name, config_path in expected_scripts.items():
        script_path = Path("slurm/smoke") / script_name
        script = script_path.read_text(encoding="utf-8")

        assert "#SBATCH --gres=gpu:1" in script
        assert config_path in script


def test_local_runner_writes_standard_result(tmp_path: Path) -> None:
    config = load_experiment_config(Path("configs/local/synthetic_smoke.json"))

    result = LocalBenchmarkRunner().run(config)
    result_path = result.write_json(tmp_path)

    assert result.status == "completed"
    assert result.throughput_tokens_sec is not None
    assert result.throughput_unit == "synthetic_tokens_per_second"
    assert result_path.exists()


def test_local_runner_records_explicit_repetition_and_provenance() -> None:
    config = load_experiment_config(Path("configs/local/synthetic_smoke.json"))

    result = LocalBenchmarkRunner().run(config, repetition_number=3)

    assert result.repetition_number == 3
    assert "-r003-" in result.run_id
    assert result.measured_iterations == config.runtime.max_steps
    assert result.git_commit
    assert result.config_sha256
    assert result.command


def test_local_runner_exposes_workload_measurement_boundaries() -> None:
    config = load_experiment_config(Path("configs/local/synthetic_smoke.json"))
    events: list[str] = []

    class RecordingProfiler(Profiler):
        def start(self, config) -> None:
            events.append("start")

        def begin_measurement(self) -> None:
            events.append("begin_measurement")

        def end_measurement(self) -> None:
            events.append("end_measurement")

        def stop(self) -> dict[str, int | float | str | bool | None]:
            events.append("stop")
            return {}

    result = LocalBenchmarkRunner(profiler=RecordingProfiler()).run(config)

    assert result.status == "completed"
    assert events == ["start", "begin_measurement", "end_measurement", "stop"]


def test_runner_fails_when_enabled_workload_profiler_metrics_are_missing() -> None:
    config = load_experiment_config(Path("configs/local/synthetic_smoke.json"))
    config = replace(
        config,
        profiling=ProfilingConfig(enabled=True, profilers=["dataloader_timing"]),
    )

    class IncompleteWorkload:
        name = "synthetic"

        def prepare(self, config) -> None:
            return None

        def run(self, config) -> WorkloadRunOutput:
            return WorkloadRunOutput(metrics={"throughput_tokens_sec": 1.0})

    registry = WorkloadRegistry()
    registry.register(IncompleteWorkload())

    result = LocalBenchmarkRunner(registry=registry).run(config)

    assert result.status == "failed"
    assert result.throughput_unit == "synthetic_tokens_per_second"
    assert "dataloader_timing requires workload metric data_loading_seconds" in str(result.error)


def test_preserved_trials_get_unique_paths(tmp_path: Path) -> None:
    config = load_experiment_config(Path("configs/local/synthetic_smoke.json"))
    first = LocalBenchmarkRunner().run(config)
    second = LocalBenchmarkRunner().run(config)

    first_path = first.write_json(tmp_path, preserve_trial=True)
    second_path = second.write_json(tmp_path, preserve_trial=True)

    assert first_path != second_path
    assert first_path.parent == tmp_path / config.experiment_id
    assert len(list(first_path.parent.glob("*.json"))) == 2


def test_default_registry_includes_transformer_workload_skeletons() -> None:
    registry = default_registry()

    assert registry.names() == [
        "distilbert",
        "environment_validation",
        "gpt2_small",
        "qwen2_5_3b",
        "qwen2_5_7b",
        "qwen2_5_7b_bf16_serving",
        "qwen2_5_7b_long_context",
        "qwen2_5_7b_training",
        "synthetic",
        "tinyllama",
    ]


def test_transformer_workload_skeleton_fails_gracefully() -> None:
    config = load_experiment_config(Path("configs/scaling_studies/E033_gpt2_small_strong_2gpu.json"))
    registry = WorkloadRegistry()
    registry.register(
        TransformerWorkload(
            name="gpt2_small",
            display_name="GPT-2 Small",
            required_packages=("hpc_ai_perf_missing_dependency",),
        )
    )

    result = LocalBenchmarkRunner(registry=registry).run(config)

    assert result.status == "failed"
    assert result.error is not None
    assert "GPT-2 Small" in result.error
    assert "optional dependencies" in result.error


def test_torch_memory_profiler_sets_explicit_local_rank_device(monkeypatch) -> None:
    config = load_experiment_config(Path("configs/environment_validation/E004_fsdp_2gpu.json"))
    events: list[tuple[str, object]] = []

    class FakeDevice:
        def __init__(self, kind: str, index: int) -> None:
            self.kind = kind
            self.index = index

        def __eq__(self, other: object) -> bool:
            return isinstance(other, FakeDevice) and (self.kind, self.index) == (other.kind, other.index)

    expected_device = FakeDevice("cuda", 1)

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def set_device(device) -> None:
            assert device == expected_device
            events.append(("set_device", device))

        @staticmethod
        def reset_peak_memory_stats(device) -> None:
            assert device == expected_device
            events.append(("reset_peak_memory_stats", device))

        @staticmethod
        def synchronize(device) -> None:
            assert device == expected_device
            events.append(("synchronize", device))

        @staticmethod
        def max_memory_allocated(device) -> int:
            assert device == expected_device
            events.append(("max_memory_allocated", device))
            return 1024**3

        @staticmethod
        def max_memory_reserved(device) -> int:
            assert device == expected_device
            events.append(("max_memory_reserved", device))
            return 2 * 1024**3

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = FakeCuda()
    fake_torch.device = lambda kind, index: FakeDevice(kind, index)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("LOCAL_RANK", "1")

    profiler = TorchMemoryProfiler()
    profiler.start(config)
    metrics = profiler.stop()

    assert events[0] == ("set_device", expected_device)
    assert ("reset_peak_memory_stats", expected_device) in events
    assert metrics["memory_used_gb"] == 1
    assert metrics["memory_reserved_gb"] == 2


def test_torch_profiler_paths_are_unique_per_trial_and_rank(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RANK", "1")

    first = TorchProfiler(tmp_path, "E024", repetition_number=1)
    second = TorchProfiler(tmp_path, "E024", repetition_number=2)

    assert first._path == tmp_path / "profiler" / "E024" / "r001" / "rank1.json"
    assert second._path == tmp_path / "profiler" / "E024" / "r002" / "rank1.json"


def test_torch_profiler_uses_bundle_artifact_directory_when_provided(monkeypatch, tmp_path: Path) -> None:
    config = load_experiment_config(Path("configs/workload_characterisation/E024_distilbert_compute.json"))
    monkeypatch.setenv("HPC_AI_ARTIFACT_OUTPUT_DIR", str(tmp_path / "bundle" / "cache" / "runtime"))

    profiler = build_profiler(config, repetition_number=2)

    assert profiler._profilers[1]._path == (
        tmp_path / "bundle" / "cache" / "runtime" / "profiler" / "E024" / "r002" / "rank0.json"
    )


def test_nonzero_distributed_ranks_skip_node_wide_nvidia_smi_sampling(monkeypatch) -> None:
    config = load_experiment_config(Path("configs/communication_analysis/E115_gpt2_small_nccl_ddp_4gpu.json"))
    monkeypatch.setenv("RANK", "1")

    profiler = build_profiler(config, repetition_number=1)

    assert isinstance(profiler, CompositeProfiler)
    assert [item.name for item in profiler._profilers] == ["nccl", "torch_profiler"]


def test_topology_capture_records_matrix_and_pcie_inventory(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return types.SimpleNamespace(stdout="GPU-a, 00000000:01:00.0, 4, 4, 16, 16\n")

    monkeypatch.setattr(environment_module.subprocess, "run", fake_run)

    captured = environment_module._capture_gpu_topology()

    assert captured["gpu_topology_capture"] is True
    assert captured["gpu_topology_matrix"] == "GPU-a, 00000000:01:00.0, 4, 4, 16, 16"
    assert captured["gpu_pcie_link_inventory"] == "GPU-a, 00000000:01:00.0, 4, 4, 16, 16"
    assert calls == [
        ["nvidia-smi", "topo", "-m"],
        [
            "nvidia-smi",
            "--query-gpu=uuid,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,"
            "pcie.link.width.current,pcie.link.width.max",
            "--format=csv,noheader,nounits",
        ],
    ]


def test_deepspeed_import_environment_unsets_stale_cuda_home(monkeypatch, tmp_path: Path) -> None:
    stale_cuda_home = tmp_path / "cuda_without_nvcc"
    stale_cuda_home.mkdir()
    monkeypatch.setenv("CUDA_HOME", str(stale_cuda_home))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv("DS_IGNORE_CUDA_DETECTION", raising=False)
    monkeypatch.setattr("hpc_ai_perf.runtime_checks.shutil.which", lambda name: None)

    metadata = prepare_deepspeed_import_environment(None)

    assert "CUDA_HOME" not in os.environ
    assert metadata["cuda_home_before"] == str(stale_cuda_home)
    assert metadata["cuda_home_after"] is None
    assert metadata["nvcc_available"] is False
    assert metadata["deepspeed_cuda_detection_ignored"] is True
    assert os.environ["DS_IGNORE_CUDA_DETECTION"] == "1"


def test_deepspeed_nvtx_annotations_are_disabled_by_default(monkeypatch) -> None:
    deepspeed_nvtx = types.ModuleType("deepspeed.utils.nvtx")
    deepspeed_nvtx.enable_nvtx = True
    monkeypatch.setitem(sys.modules, "deepspeed.utils.nvtx", deepspeed_nvtx)

    assert disable_deepspeed_nvtx_annotations() is True
    assert deepspeed_nvtx.enable_nvtx is False


def test_deepspeed_nvtx_annotations_can_be_explicitly_enabled(monkeypatch) -> None:
    deepspeed_nvtx = types.ModuleType("deepspeed.utils.nvtx")
    deepspeed_nvtx.enable_nvtx = True
    monkeypatch.setitem(sys.modules, "deepspeed.utils.nvtx", deepspeed_nvtx)
    monkeypatch.setenv("HPC_AI_ENABLE_DEEPSPEED_NVTX", "1")

    assert disable_deepspeed_nvtx_annotations() is False
    assert deepspeed_nvtx.enable_nvtx is True


def test_deepspeed_managed_precision_does_not_nest_native_amp() -> None:
    context, scaler = transformer_module._training_precision(
        object(),
        "bf16",
        object(),
        managed_by_deepspeed=True,
    )

    assert context is transformer_module.nullcontext
    assert scaler is None
    assert transformer_module._precision_backend("bf16", managed_by_deepspeed=True) == "deepspeed_bf16"


def test_fp8_training_precision_requires_transformer_engine_context() -> None:
    device = types.SimpleNamespace(type="cuda")

    with pytest.raises(RuntimeError, match="configured Transformer Engine context"):
        transformer_module._training_precision(object(), "fp8", device)

    context_factory = lambda: transformer_module.nullcontext()  # noqa: E731
    context, scaler = transformer_module._training_precision(
        object(),
        "fp8",
        device,
        fp8_context=context_factory,
    )

    assert context is context_factory
    assert scaler is None
    assert transformer_module._precision_backend("fp8") == "transformer_engine_delayed_scaling_hybrid"


def test_transformer_engine_backend_selects_matched_module_conversion() -> None:
    config = load_experiment_config(Path("configs/precision_studies/E086_gpt2_small_training_fp8.json"))
    parameters = dict(config.workload.parameters)
    parameters["precision_backend"] = "transformer_engine"
    bf16 = replace(
        config,
        runtime=replace(config.runtime, precision="bf16"),
        workload=replace(config.workload, parameters=parameters),
    )

    assert transformer_module._uses_transformer_engine(config)
    assert transformer_module._uses_transformer_engine(bf16)


def test_transformer_engine_conversion_replaces_only_shape_compatible_projections() -> None:
    torch = pytest.importorskip("torch")

    class Conv1D(torch.nn.Module):
        __module__ = "transformers.pytorch_utils"

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(16 * 32, dtype=torch.float32).reshape(16, 32))
            self.bias = torch.nn.Parameter(torch.zeros(32))

    class FakeTELinear(torch.nn.Linear):
        def __init__(self, in_features, out_features, bias=True, params_dtype=None) -> None:
            super().__init__(in_features, out_features, bias=bias, dtype=params_dtype)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(16, 32)
            self.conv_projection = Conv1D()
            self.pre_classifier = torch.nn.Linear(32, 32)
            self.classifier = torch.nn.Linear(32, 3)

    model = Model()
    linear_weight = model.projection.weight.detach().clone()
    conv_weight = model.conv_projection.weight.detach().clone()
    fake_te = types.SimpleNamespace(Linear=FakeTELinear)

    converted, skipped = transformer_module._convert_projection_layers_to_transformer_engine(torch, fake_te, model)

    assert converted == ["projection", "conv_projection"]
    assert skipped == ["pre_classifier", "classifier"]
    assert isinstance(model.projection, FakeTELinear)
    assert isinstance(model.conv_projection, FakeTELinear)
    assert isinstance(model.pre_classifier, torch.nn.Linear)
    assert torch.allclose(model.projection.weight.float(), linear_weight, atol=2e-3)
    assert torch.allclose(model.conv_projection.weight.float(), conv_weight.transpose(0, 1), atol=1.0)
    assert isinstance(model.classifier, torch.nn.Linear)


def test_transformer_engine_conversion_skips_without_optional_torch(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(pytest.skip.Exception, match="torch"):
        test_transformer_engine_conversion_replaces_only_shape_compatible_projections()


def test_fsdp_activation_checkpointing_prefers_non_reentrant_kwargs() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.kwargs = None

        def gradient_checkpointing_enable(self, **kwargs) -> None:
            self.kwargs = kwargs

    model = FakeModel()

    backend = transformer_module._enable_activation_checkpointing(model, prefer_non_reentrant=True)

    assert backend == "hf_gradient_checkpointing_non_reentrant"
    assert model.kwargs == {"gradient_checkpointing_kwargs": {"use_reentrant": False}}


def test_activation_checkpointing_falls_back_when_kwargs_are_unavailable() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls = []

        def gradient_checkpointing_enable(self, **kwargs) -> None:
            if kwargs:
                raise TypeError("unexpected keyword")
            self.calls.append(kwargs)

    model = FakeModel()

    backend = transformer_module._enable_activation_checkpointing(model, prefer_non_reentrant=True)

    assert backend == "hf_gradient_checkpointing_default_kwargs_unavailable"
    assert model.calls == [{}]


def test_runtime_check_rejects_old_deepspeed_without_nvcc(monkeypatch, tmp_path: Path) -> None:
    stale_cuda_home = tmp_path / "cuda_without_nvcc"
    stale_cuda_home.mkdir()

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def get_device_name(index: int) -> str:
            return "NVIDIA L40S"

    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.8.0"
    fake_torch.version = types.SimpleNamespace(cuda="12.4")
    fake_torch.cuda = FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("CUDA_HOME", str(stale_cuda_home))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv("DS_IGNORE_CUDA_DETECTION", raising=False)
    monkeypatch.setattr("hpc_ai_perf.runtime_checks.shutil.which", lambda name: None)
    monkeypatch.setattr("hpc_ai_perf.runtime_checks.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr("hpc_ai_perf.runtime_checks.metadata.version", lambda package: "0.17.6")

    checks = run_cluster_runtime_checks(
        expected_gpu_model="L40S",
        expected_gpus=2,
        require_deepspeed=True,
        require_clean_git=False,
    )

    failing_checks = {check.name: check for check in checks if check.failed}
    assert "deepspeed_version" in failing_checks
    assert "0.17.6" in failing_checks["deepspeed_version"].detail
    assert "deepspeed>=0.18.9" in str(failing_checks["deepspeed_version"].remediation)


def test_runtime_check_requires_gloo_for_vllm_serving(monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_name(index: int) -> str:
            return "NVIDIA L40S"

    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.8.0"
    fake_torch.version = types.SimpleNamespace(cuda="12.4")
    fake_torch.cuda = FakeCuda()
    fake_torch.distributed = types.SimpleNamespace(is_gloo_available=lambda: False)
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = "0.10.2+cu124"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr("hpc_ai_perf.runtime_checks.importlib.util.find_spec", lambda name: object())

    checks = run_cluster_runtime_checks(
        expected_gpu_model="L40S",
        expected_gpus=1,
        require_vllm=True,
        require_clean_git=False,
    )

    failing_checks = {check.name: check for check in checks if check.failed}
    assert "vllm_gloo" in failing_checks
    assert "Gloo-enabled PyTorch" in str(failing_checks["vllm_gloo"].remediation)


def test_fsdp_environment_validation_uses_explicit_cuda_device(monkeypatch) -> None:
    config = load_experiment_config(Path("configs/environment_validation/E004_fsdp_2gpu.json"))
    devices_seen: list[object] = []

    class FakeDevice:
        def __init__(self, kind: str, index: int) -> None:
            self.kind = kind
            self.index = index

        def __eq__(self, other: object) -> bool:
            return isinstance(other, FakeDevice) and (self.kind, self.index) == (other.kind, other.index)

    expected_device = FakeDevice("cuda", 1)

    class FakeTensor:
        def backward(self) -> None:
            return None

        def detach(self):
            return self

        def cpu(self):
            return self

        def __float__(self) -> float:
            return 0.5

    class FakeModel:
        def __init__(self) -> None:
            self.device = None

        def to(self, device):
            self.device = device
            devices_seen.append(device)
            return self

        def parameters(self):
            return []

        def __call__(self, inputs):
            return FakeTensor()

    class FakeFSDP(FakeModel):
        def __init__(self, model, *, device_id, use_orig_params) -> None:
            super().__init__()
            assert model.device == expected_device
            assert device_id == expected_device
            assert use_orig_params is True
            self.model = model

    class FakeCuda:
        @staticmethod
        def reset_peak_memory_stats(device) -> None:
            assert device == expected_device
            devices_seen.append(device)

        @staticmethod
        def synchronize(device) -> None:
            assert device == expected_device
            devices_seen.append(device)

        @staticmethod
        def max_memory_allocated(device) -> int:
            assert device == expected_device
            devices_seen.append(device)
            return 1024**3

    class FakeTorch:
        cuda = FakeCuda()
        nn = types.SimpleNamespace(
            Sequential=lambda *layers: FakeModel(),
            Linear=lambda *args: object(),
            ReLU=lambda: object(),
            functional=types.SimpleNamespace(cross_entropy=lambda output, labels: FakeTensor()),
        )
        optim = types.SimpleNamespace(
            AdamW=lambda parameters, lr: types.SimpleNamespace(
                zero_grad=lambda set_to_none: None,
                step=lambda: None,
            )
        )

        @staticmethod
        def device(kind: str, index: int) -> FakeDevice:
            return FakeDevice(kind, index)

        @staticmethod
        def randn(*shape, device):
            assert device == expected_device
            devices_seen.append(device)
            return FakeTensor()

        @staticmethod
        def randint(low: int, high: int, shape, *, device):
            assert device == expected_device
            devices_seen.append(device)
            return FakeTensor()

    fake_fsdp_module = types.ModuleType("torch.distributed.fsdp")
    fake_fsdp_module.FullyShardedDataParallel = FakeFSDP
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", fake_fsdp_module)
    monkeypatch.setattr(environment_module, "_torch_with_cuda", lambda _: FakeTorch())
    monkeypatch.setattr(environment_module, "_init_process_group", lambda torch, config: (object(), 1, 1, 2))

    output = EnvironmentValidationWorkload().run(config)

    assert output.metrics["fsdp_init"] is True
    assert output.metrics["single_train_step"] is True
    assert devices_seen
    assert all(device == expected_device for device in devices_seen)


def test_deepspeed_environment_validation_uses_explicit_cuda_device(monkeypatch) -> None:
    config = load_experiment_config(Path("configs/environment_validation/E005_deepspeed_zero_2gpu.json"))
    devices_seen: list[object] = []

    class FakeDevice:
        def __init__(self, kind: str, index: int) -> None:
            self.kind = kind
            self.index = index

        def __eq__(self, other: object) -> bool:
            return isinstance(other, FakeDevice) and (self.kind, self.index) == (other.kind, other.index)

    expected_device = FakeDevice("cuda", 1)

    class FakeTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def __float__(self) -> float:
            return 0.5

    class FakeParameter:
        dtype = "bfloat16"

    class FakeModel:
        def __init__(self) -> None:
            self.device = None

        def to(self, device):
            self.device = device
            devices_seen.append(device)
            return self

        def parameters(self):
            return []

        def __call__(self, inputs):
            return FakeTensor()

    class FakeEngine:
        @staticmethod
        def parameters():
            return iter([FakeParameter()])

        def __call__(self, inputs):
            return FakeTensor()

        @staticmethod
        def backward(loss) -> None:
            return None

        @staticmethod
        def step() -> None:
            return None

    class FakeCuda:
        @staticmethod
        def set_device(device) -> None:
            assert device == expected_device
            devices_seen.append(device)

        @staticmethod
        def synchronize(device) -> None:
            assert device == expected_device

    class FakeTorch:
        cuda = FakeCuda()
        nn = types.SimpleNamespace(
            Linear=lambda *args: FakeModel(),
            functional=types.SimpleNamespace(cross_entropy=lambda output, labels: FakeTensor()),
        )
        optim = types.SimpleNamespace(AdamW=lambda parameters, lr: "torch-adamw")

        @staticmethod
        def device(kind: str, index: int) -> FakeDevice:
            return FakeDevice(kind, index)

        @staticmethod
        def randn(*shape, device, dtype):
            assert device == expected_device
            assert dtype == "bfloat16"
            devices_seen.append(device)
            return FakeTensor()

        @staticmethod
        def randint(low: int, high: int, shape, *, device):
            assert device == expected_device
            devices_seen.append(device)
            return FakeTensor()

    def fake_deepspeed_initialize(*, model, model_parameters, optimizer, config):
        assert model.device == expected_device
        assert optimizer == "torch-adamw"
        assert config["gradient_accumulation_steps"] == 1
        assert config["zero_optimization"]["stage"] == 2
        return FakeEngine(), None, None, None

    fake_deepspeed = types.ModuleType("deepspeed")
    fake_deepspeed.initialize = fake_deepspeed_initialize
    monkeypatch.setitem(sys.modules, "deepspeed", fake_deepspeed)
    monkeypatch.setattr(environment_module, "_torch_with_cuda", lambda _: FakeTorch())
    monkeypatch.setattr(environment_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")

    output = EnvironmentValidationWorkload().run(config)

    assert output.metrics["zero_init"] is True
    assert output.metrics["single_train_step"] is True
    assert output.metrics["torch_optimizer"] == "AdamW"
    assert devices_seen
    assert all(device == expected_device for device in devices_seen)


def test_vllm_environment_validation_serves_a_real_request(monkeypatch) -> None:
    config = load_experiment_config(Path("configs/environment_validation/E006_vllm_1gpu.json"))

    class FakeCuda:
        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def synchronize() -> None:
            return None

    class FakeTorch:
        cuda = FakeCuda()

    class FakeCandidate:
        token_ids = [1, 2]

    class FakeOutput:
        outputs = [FakeCandidate()]

    class FakeLlm:
        def __init__(self, **kwargs) -> None:
            assert kwargs["model"] == "gpt2"
            assert kwargs["revision"] == config.workload.parameters["hf_model_revision"]

        def generate(self, prompts, **kwargs):
            assert prompts == [{"prompt_token_ids": [15496]}]
            assert "prompt_token_ids" not in kwargs
            return [FakeOutput()]

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = "test"
    fake_vllm.LLM = FakeLlm
    fake_vllm.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(environment_module, "_torch_with_cuda", lambda _: FakeTorch())
    monkeypatch.setattr(environment_module.importlib.util, "find_spec", lambda _: object())

    output = EnvironmentValidationWorkload().run(config)

    assert output.metrics["engine_start"] is True
    assert output.metrics["request_served"] is True
    assert output.metrics["generated_tokens"] == 2
    assert output.metrics["throughput_tokens_sec"] > 0


def test_vllm_environment_validation_supports_eight_gpu_tensor_parallel_canary(monkeypatch) -> None:
    config = load_experiment_config(Path("configs/environment_validation/E230_l40s_vllm_tinyllama_tp8_canary.json"))
    assert capability_issues(config) == []
    llm_kwargs: dict[str, object] = {}

    class FakeCuda:
        @staticmethod
        def device_count() -> int:
            return 8

        @staticmethod
        def synchronize() -> None:
            return None

    class FakeTorch:
        cuda = FakeCuda()

    class FakeCandidate:
        token_ids = [1, 2]

    class FakeOutput:
        outputs = [FakeCandidate()]

    class FakeLlm:
        def __init__(self, **kwargs) -> None:
            llm_kwargs.update(kwargs)

        def generate(self, prompts, **kwargs):
            assert prompts == [{"prompt_token_ids": [1]}]
            return [FakeOutput()]

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = "test"
    fake_vllm.LLM = FakeLlm
    fake_vllm.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(environment_module, "_torch_with_cuda", lambda _: FakeTorch())
    monkeypatch.setattr(environment_module.importlib.util, "find_spec", lambda _: object())

    output = EnvironmentValidationWorkload().run(config)

    assert llm_kwargs["tensor_parallel_size"] == 8
    assert llm_kwargs["compilation_config"] == {"level": 0, "use_inductor": False}
    assert output.metrics["tensor_parallel_size_observed"] == 8


def test_vllm_environment_validation_reports_cuda_wheel_mismatch(monkeypatch) -> None:
    config = load_experiment_config(Path("configs/environment_validation/E006_vllm_1gpu.json"))
    real_import = builtins.__import__

    class FakeTorch:
        version = types.SimpleNamespace(cuda="12.4")

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm":
            raise ImportError("libcudart.so.13: cannot open shared object file")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(environment_module, "_torch_with_cuda", lambda _: FakeTorch())
    monkeypatch.setattr(environment_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as exc_info:
        EnvironmentValidationWorkload().run(config)

    message = str(exc_info.value)
    assert "vLLM import failed before engine startup" in message
    assert "torch CUDA 12.4" in message
    assert "libcudart.so.13" in message


def test_tensorrt_llm_environment_validation_builds_loads_and_serves(tmp_path: Path, monkeypatch) -> None:
    config = load_experiment_config(Path("configs/environment_validation/E007_tensorrt_llm_1gpu.json"))
    model_path = tmp_path / "model"
    engine_root = tmp_path / "engines"
    model_path.mkdir()
    (model_path / "hpc_ai_perf_model_revision.txt").write_text(
        str(config.workload.parameters["hf_model_revision"]), encoding="utf-8"
    )
    monkeypatch.setenv("TRTLLM_MODEL_PATH", str(model_path))
    monkeypatch.setenv("TRTLLM_ENGINE_ROOT", str(engine_root))
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    class FakeCuda:
        @staticmethod
        def synchronize() -> None:
            return None

        @staticmethod
        def empty_cache() -> None:
            return None

        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "NVIDIA H100 SXM 80GB"

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_capability(index: int) -> tuple[int, int]:
            assert index == 0
            return 9, 0

    class FakeTorch:
        __version__ = "test-torch"
        version = types.SimpleNamespace(cuda="12.4")
        cuda = FakeCuda()

    class FakeBuildConfig:
        max_batch_size = 0
        max_num_tokens = 0

    class FakeCandidate:
        token_ids = [1, 2]

    class FakeOutput:
        outputs = [FakeCandidate()]

    created_models: list[str] = []

    class FakeLlm:
        def __init__(self, **kwargs) -> None:
            created_models.append(kwargs["model"])
            if kwargs["model"] == str(model_path):
                assert kwargs["tensor_parallel_size"] == 1
                assert kwargs["build_config"].max_batch_size == 1
                assert kwargs["build_config"].max_num_tokens == 72
            else:
                assert kwargs == {"model": str(engine_root / "E007-12345")}

        def save(self, engine_path: str) -> None:
            output_path = Path(engine_path)
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "rank0.engine").write_bytes(b"test-engine")

        def generate(self, prompts, sampling_params=None, use_tqdm=False):
            assert prompts == ["The capital of France is"]
            assert sampling_params["max_tokens"] == 8
            assert use_tqdm is False
            return [FakeOutput()]

        def shutdown(self) -> None:
            return None

    fake_tensorrt_llm = types.ModuleType("tensorrt_llm")
    fake_tensorrt_llm.__version__ = "0.19.0"
    fake_tensorrt_llm.LLM = FakeLlm
    fake_tensorrt_llm.SamplingParams = lambda **kwargs: kwargs
    fake_tensorrt_llm.llmapi = types.SimpleNamespace(BuildConfig=FakeBuildConfig)
    monkeypatch.setitem(sys.modules, "tensorrt_llm", fake_tensorrt_llm)
    monkeypatch.setattr(environment_module, "_torch_with_cuda", lambda _: FakeTorch())
    monkeypatch.setattr(environment_module.importlib.util, "find_spec", lambda _: object())

    output = EnvironmentValidationWorkload().run(config)

    engine_path = engine_root / "E007-12345"
    manifest_path = engine_path / "hpc_ai_perf_engine_manifest.json"
    assert created_models == [str(model_path), str(engine_path)]
    assert output.metrics["tensorrt_llm_import"] is True
    assert output.metrics["engine_build"] is True
    assert output.metrics["engine_loaded"] is True
    assert output.metrics["request_served"] is True
    assert output.metrics["served_requests"] == 3
    assert output.metrics["generated_tokens"] == 6
    assert len(str(output.metrics["engine_manifest_sha256"])) == 64
    assert manifest_path.is_file()


def test_tensorrt_llm_environment_validation_rejects_runtime_without_engine_save(tmp_path: Path, monkeypatch) -> None:
    config = load_experiment_config(Path("configs/environment_validation/E007_tensorrt_llm_1gpu.json"))
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "hpc_ai_perf_model_revision.txt").write_text(
        str(config.workload.parameters["hf_model_revision"]), encoding="utf-8"
    )
    monkeypatch.setenv("TRTLLM_MODEL_PATH", str(model_path))
    monkeypatch.setenv("TRTLLM_ENGINE_ROOT", str(tmp_path / "engines"))
    monkeypatch.setenv("SLURM_JOB_ID", "new-api")

    class FakeCuda:
        @staticmethod
        def empty_cache() -> None:
            return None

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "NVIDIA H100 SXM 80GB"

    class FakeTorch:
        cuda = FakeCuda()

    class FakeLlm:
        def __init__(self, **kwargs) -> None:
            return None

    fake_tensorrt_llm = types.ModuleType("tensorrt_llm")
    fake_tensorrt_llm.__version__ = "0.19.0"
    fake_tensorrt_llm.LLM = FakeLlm
    monkeypatch.setitem(sys.modules, "tensorrt_llm", fake_tensorrt_llm)
    monkeypatch.setattr(environment_module, "_torch_with_cuda", lambda _: FakeTorch())
    monkeypatch.setattr(environment_module.importlib.util, "find_spec", lambda _: object())

    try:
        EnvironmentValidationWorkload().run(config)
    except RuntimeError as exc:
        assert "does not expose LLM.save()" in str(exc)
    else:
        raise AssertionError("E007 accepted a TensorRT-LLM runtime without saved-engine support")


def test_tensorrt_llm_environment_validation_rejects_unresolved_artifact_paths(monkeypatch) -> None:
    config = load_experiment_config(Path("configs/environment_validation/E007_tensorrt_llm_1gpu.json"))
    monkeypatch.delenv("TRTLLM_MODEL_PATH", raising=False)
    monkeypatch.delenv("TRTLLM_ENGINE_ROOT", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    fake_tensorrt_llm = types.ModuleType("tensorrt_llm")
    fake_tensorrt_llm.__version__ = "0.19.0"
    monkeypatch.setitem(sys.modules, "tensorrt_llm", fake_tensorrt_llm)
    monkeypatch.setattr(environment_module, "_torch_with_cuda", lambda _: object())
    monkeypatch.setattr(environment_module.importlib.util, "find_spec", lambda _: object())

    try:
        EnvironmentValidationWorkload().run(config)
    except RuntimeError as exc:
        assert "unresolved environment variable" in str(exc)
    else:
        raise AssertionError("E007 accepted unresolved artifact paths")


def test_tensorrt_llm_environment_validation_rejects_wrong_gpu(tmp_path: Path, monkeypatch) -> None:
    config = load_experiment_config(Path("configs/environment_validation/E007_tensorrt_llm_1gpu.json"))
    monkeypatch.setenv("TRTLLM_MODEL_PATH", str(tmp_path / "model"))
    monkeypatch.setenv("TRTLLM_ENGINE_ROOT", str(tmp_path / "engines"))
    monkeypatch.setenv("SLURM_JOB_ID", "wrong-gpu")

    class FakeCuda:
        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "NVIDIA L40S"

    class FakeTorch:
        cuda = FakeCuda()

    fake_tensorrt_llm = types.ModuleType("tensorrt_llm")
    fake_tensorrt_llm.__version__ = "0.19.0"
    monkeypatch.setitem(sys.modules, "tensorrt_llm", fake_tensorrt_llm)
    monkeypatch.setattr(environment_module, "_torch_with_cuda", lambda _: FakeTorch())
    monkeypatch.setattr(environment_module.importlib.util, "find_spec", lambda _: object())

    try:
        EnvironmentValidationWorkload().run(config)
    except RuntimeError as exc:
        assert "requested NVIDIA H100 SXM 80GB, observed NVIDIA L40S" in str(exc)
    else:
        raise AssertionError("E007 accepted an L40S for an H100-labelled result")


def test_gpt2_inference_dispatches_to_causal_lm_helper(monkeypatch) -> None:
    config = load_experiment_config(Path("configs/local/gpt2_small_smoke.json"))
    config = config.__class__(
        experiment_id=config.experiment_id,
        experiment=config.experiment,
        mode="inference",
        workload=config.workload,
        hardware=config.hardware,
        distributed=config.distributed,
        runtime=config.runtime,
        profiling=config.profiling,
        metadata=config.metadata,
    )
    calls = []

    def fake_inference(dispatched_config):
        calls.append(dispatched_config.experiment_id)
        return transformer_module.WorkloadRunOutput(metrics={"throughput_tokens_sec": 1.0})

    monkeypatch.setattr(transformer_module, "_run_causal_lm_inference", fake_inference)

    output = GPT2SmallWorkload().run(config)

    assert calls == ["local_gpt2_small_smoke"]
    assert output.metrics["throughput_tokens_sec"] == 1.0


@pytest.mark.parametrize(
    (
        "config_path",
        "expected_tensor_parallel_size",
        "expected_compilation_config",
        "expected_execution_mode",
        "expected_quantization",
    ),
    [
        ("configs/inference/E130_gpt2_small_framework_vllm.json", 1, None, "eager", None),
        ("configs/inference/E151_gpt2_small_precision_fp8.json", 1, None, "eager", "fp8"),
        (
            "configs/inference/E147_gpt2_small_multi_gpu_2gpu.json",
            2,
            {"level": 0, "use_inductor": False},
            "eager_no_inductor",
            None,
        ),
    ],
)
def test_vllm_inference_uses_certified_eager_execution(
    monkeypatch,
    config_path: str,
    expected_tensor_parallel_size: int,
    expected_compilation_config: dict[str, object] | None,
    expected_execution_mode: str,
    expected_quantization: str | None,
) -> None:
    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising=False)
    config = load_experiment_config(Path(config_path))
    config = replace(config, runtime=replace(config.runtime, max_steps=1))
    llm_kwargs: dict[str, object] = {}

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def synchronize() -> None:
            return None

        @staticmethod
        def get_device_capability(index: int) -> tuple[int, int]:
            assert index == 0
            return (9, 0)

    class FakeGenerator:
        def manual_seed(self, seed: int):
            assert seed == 1337
            return self

    class FakeTokenIds:
        @staticmethod
        def tolist() -> list[list[int]]:
            return [[1] * 128]

    class FakeTorch:
        cuda = FakeCuda()
        Generator = FakeGenerator

        @staticmethod
        def randint(*args, **kwargs) -> FakeTokenIds:
            assert args == (0, 50257, (1, 128))
            assert isinstance(kwargs["generator"], FakeGenerator)
            return FakeTokenIds()

    class FakeTokenizer:
        vocab_size = 50257

    class FakeCandidate:
        token_ids = [1, 2]

    class FakeOutput:
        outputs = [FakeCandidate()]
        metrics = None

    class FakeLlm:
        def __init__(self, **kwargs) -> None:
            if expected_tensor_parallel_size > 1:
                assert os.environ["TORCHDYNAMO_DISABLE"] == "1"
            else:
                assert "TORCHDYNAMO_DISABLE" not in os.environ
            llm_kwargs.update(kwargs)
            self.llm_engine = types.SimpleNamespace(
                model_config=types.SimpleNamespace(
                    quantization=kwargs.get("quantization"),
                    dtype=kwargs["dtype"],
                )
            )

        @staticmethod
        def get_tokenizer() -> FakeTokenizer:
            return FakeTokenizer()

        @staticmethod
        def generate(*args, **kwargs) -> list[FakeOutput]:
            return [FakeOutput()]

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLlm
    fake_vllm.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(transformer_module.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(transformer_module, "_prepare_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(transformer_module, "_start_cuda_timer", lambda *args: None)

    output = transformer_module._run_vllm_inference(config)

    assert llm_kwargs["tensor_parallel_size"] == expected_tensor_parallel_size
    assert llm_kwargs["enforce_eager"] is True
    assert llm_kwargs.get("compilation_config") == expected_compilation_config
    assert llm_kwargs.get("quantization") == expected_quantization
    assert output.metrics["vllm_execution_mode"] == expected_execution_mode
    assert output.metrics["vllm_inductor_disabled"] == (expected_compilation_config is not None)
    assert output.metrics["vllm_torchdynamo_disabled"] == (expected_tensor_parallel_size > 1)
    if expected_quantization == "fp8":
        assert output.metrics["precision_backend"] == "vllm_fp8_online"
        assert output.metrics["quantization_method"] == "fp8_w8a8_dynamic"
        assert output.metrics["vllm_quantization"] == "fp8"
        assert output.metrics["quantized_linear_compute"] == "fp8_w8a8"
        assert output.metrics["comparison_control_id"] == "E130"
        assert output.metrics["quality_request_success_rate"] == 1.0


def test_vllm_fp8_rejects_an_engine_that_did_not_activate_quantization(monkeypatch) -> None:
    config = load_experiment_config(Path("configs/inference/E151_gpt2_small_precision_fp8.json"))
    config = replace(config, runtime=replace(config.runtime, max_steps=1))

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_capability(index: int) -> tuple[int, int]:
            return (9, 0)

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = FakeCuda()
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = object

    class FakeLlm:
        def __init__(self, **kwargs) -> None:
            self.llm_engine = types.SimpleNamespace(
                model_config=types.SimpleNamespace(quantization=None, dtype="bfloat16")
            )

    fake_vllm.LLM = FakeLlm
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(transformer_module.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(transformer_module, "_prepare_runtime", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="did not activate the requested quantization backend"):
        transformer_module._run_vllm_inference(config)


def test_tinyllama_training_dispatches_to_causal_training(monkeypatch) -> None:
    config = load_experiment_config(Path("configs/local/tinyllama_inference_smoke.json"))
    training_config = config.__class__(
        experiment_id=config.experiment_id,
        experiment=config.experiment,
        mode="training",
        workload=config.workload,
        hardware=config.hardware,
        distributed=config.distributed,
        runtime=config.runtime,
        profiling=config.profiling,
        metadata=config.metadata,
    )

    calls = []

    def fake_training(dispatched_config):
        calls.append(dispatched_config.experiment_id)
        return transformer_module.WorkloadRunOutput(metrics={"loss": 1.0})

    monkeypatch.setattr(transformer_module, "_run_causal_training", fake_training)

    output = TinyLlamaWorkload().run(training_config)

    assert calls == ["local_tinyllama_inference_smoke"]
    assert output.metrics["loss"] == 1.0


def test_transformer_device_auto_prefers_mps_when_cuda_is_unavailable() -> None:
    class FakeMpsBackend:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()

        class backends:
            mps = FakeMpsBackend()

        @staticmethod
        def device(name: str) -> str:
            return name

    assert transformer_module._device(FakeTorch, {"device": "auto"}) == "mps"


def test_cuda_event_timer_reports_seconds() -> None:
    class FakeEvent:
        def __init__(self, **kwargs) -> None:
            assert kwargs == {"enable_timing": True}

        def record(self) -> None:
            return None

        def synchronize(self) -> None:
            return None

        def elapsed_time(self, completed) -> float:
            return 125.0

    class FakeCuda:
        Event = FakeEvent

        @staticmethod
        def is_available() -> bool:
            return True

    class FakeTorch:
        cuda = FakeCuda()

    timer = transformer_module._start_cuda_timer(FakeTorch)

    assert transformer_module._stop_cuda_timer(FakeTorch, timer) == 0.125


def test_nvidia_smi_rows_follow_cuda_uuid_order() -> None:
    rows = [
        "GPU-b, 80, 2048, 120, 50, 2400",
        "GPU-a, 40, 1024, 100, 45, 2200",
        "GPU-c, 10, 512, 80, 40, 1800",
    ]

    selected, method = select_nvidia_smi_rows(
        rows,
        target_gpu_uuids=("GPU-a", "GPU-b"),
        expected_gpu_count=2,
    )

    assert method == "cuda_uuid"
    assert selected == [rows[1], rows[0]]


def test_nvidia_smi_rows_reject_ambiguous_positional_mapping() -> None:
    with pytest.raises(ValueError, match="cannot prove nvidia-smi device ownership"):
        select_nvidia_smi_rows(
            ["GPU-a, 40", "GPU-b, 50", "GPU-c, 60"],
            target_gpu_uuids=(),
            expected_gpu_count=2,
        )


def test_nvidia_smi_profiler_records_uuid_mapping_and_metric_scopes(monkeypatch) -> None:
    profiler = NvidiaSmiProfiler()
    profiler._gpu_count = 2
    profiler._target_gpu_uuids = ("GPU-a", "GPU-b")
    profiler._started_at = 1.0
    monkeypatch.setattr(
        "hpc_ai_perf.profiling.system.query_nvidia_smi_rows",
        lambda query, timeout=5: [
            "GPU-b, 80, 2048, 120, 50, 2400",
            "GPU-a, 40, 1024, 100, 45, 2200",
        ],
    )
    monkeypatch.setattr("hpc_ai_perf.profiling.system.time.perf_counter", lambda: 3.0)

    profiler._sample_once()
    metrics = profiler.stop()

    assert metrics["gpu_utilization"] == 60.0
    assert metrics["memory_used_gb"] == 2.0
    assert metrics["power_draw_watts"] == 220.0
    assert metrics["profiled_gpu_uuids"] == "GPU-a,GPU-b"
    assert metrics["gpu_mapping_method"] == "cuda_uuid"
    assert metrics["telemetry_schema_version"] == "3"
    assert metrics["memory_metric_scope"] == "max_per_gpu"


def test_nvidia_smi_profiler_integrates_only_the_measured_region() -> None:
    profiler = NvidiaSmiProfiler()
    profiler._started_at = 0.0
    profiler._collection_stopped_at = 4.0
    profiler._measurement_started_at = 1.0
    profiler._measurement_stopped_at = 3.0
    profiler._samples = [
        _NvidiaSmiSample(0.0, 0.0, 1024.0, 50.0, 30.0, 1000.0),
        _NvidiaSmiSample(1.0, 50.0, 2048.0, 100.0, 40.0, 2000.0),
        _NvidiaSmiSample(2.0, 90.0, 3072.0, 200.0, 50.0, 2500.0),
        _NvidiaSmiSample(3.0, 50.0, 2048.0, 100.0, 40.0, 2000.0),
        _NvidiaSmiSample(4.0, 0.0, 1024.0, 50.0, 30.0, 1000.0),
    ]

    metrics = profiler.stop()

    assert metrics["energy_scope"] == "measured_region"
    assert metrics["measurement_status"] == "sufficient"
    assert metrics["measurement_duration_seconds"] == 2.0
    assert metrics["energy_joules_measured_region"] == 300.0
    assert metrics["power_draw_watts_measured_region"] == 150.0
    assert metrics["energy_joules_full_window"] == 450.0


def test_nvidia_smi_profiler_marks_short_measurement_as_insufficient() -> None:
    profiler = NvidiaSmiProfiler()
    profiler._started_at = 0.0
    profiler._collection_stopped_at = 1.0
    profiler._measurement_started_at = 0.25
    profiler._measurement_stopped_at = 0.75
    profiler._samples = [_NvidiaSmiSample(0.5, 50.0, 1024.0, 100.0, 40.0, 2000.0)]

    metrics = profiler.stop()

    assert metrics["energy_scope"] == "full_window"
    assert metrics["measurement_status"] == "insufficient_samples"
    assert "energy_joules_measured_region" not in metrics


def test_composite_freezes_measurement_before_profiler_finalization() -> None:
    events: list[str] = []

    class RecordingProfiler(Profiler):
        def __init__(self, name: str, priority: int) -> None:
            self.name = name
            self.measurement_stop_priority = priority

        def begin_measurement(self) -> None:
            events.append(f"begin:{self.name}")

        def end_measurement(self) -> None:
            events.append(f"end:{self.name}")

        def stop(self) -> dict[str, int | float | str | bool | None]:
            events.append(f"stop:{self.name}")
            return {}

    profiler = CompositeProfiler([RecordingProfiler("nvidia_smi", 100), RecordingProfiler("torch_profiler", 0)])
    profiler.begin_measurement()
    profiler.end_measurement()
    profiler.stop()

    assert events == [
        "begin:nvidia_smi",
        "begin:torch_profiler",
        "end:nvidia_smi",
        "end:torch_profiler",
        "stop:torch_profiler",
        "stop:nvidia_smi",
    ]


def test_aggregate_results_writes_processed_csv(tmp_path: Path) -> None:
    config = load_experiment_config(Path("configs/scaling_studies/E033_gpt2_small_strong_2gpu.json"))
    raw_dir = tmp_path / "raw"
    output_path = tmp_path / "processed" / "experiments.csv"

    result = ExperimentResult.from_config(
        config,
        status="completed",
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:10:00Z",
        runtime_seconds=600.0,
        metrics={"throughput_tokens_sec": 1234.5, "throughput_unit": "training_tokens_per_second"},
    )
    result.write_json(raw_dir)

    csv_path = aggregate_results(raw_dir, output_path)

    with csv_path.open(encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))

    assert csv_path == output_path
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "E033"
    assert rows[0]["condition_id"] == "E033@h100"
    assert rows[0]["platform_id"] == "h100"
    assert rows[0]["phase"] == "Strong scaling"
    assert rows[0]["priority"] == "Must Have"
    assert rows[0]["strategy"] == "ddp"
    assert rows[0]["throughput_unit"] == "training_tokens_per_second"
    assert rows[0]["metric_throughput_tokens_sec"]
    assert rows[0]["gpu_telemetry_mapping"] == "not_collected"


def test_aggregate_results_labels_legacy_and_uuid_verified_telemetry(tmp_path: Path) -> None:
    config = load_experiment_config(Path("configs/baseline_profiling/E008_gpt2_small_training_baseline.json"))
    raw_dir = tmp_path / "raw"
    output_path = tmp_path / "processed.csv"
    legacy = ExperimentResult.from_config(
        config,
        status="completed",
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:00:01Z",
        runtime_seconds=1.0,
        repetition_number=1,
        metrics={"nvidia_smi.samples": 5},
    )
    verified = ExperimentResult.from_config(
        config,
        status="completed",
        started_at="2026-01-01T00:00:02Z",
        completed_at="2026-01-01T00:00:03Z",
        runtime_seconds=1.0,
        repetition_number=2,
        metrics={
            "nvidia_smi.samples": 5,
            "nvidia_smi.gpu_mapping_method": "cuda_uuid",
            "nvidia_smi.telemetry_schema_version": "2",
            "nvidia_smi.profiled_gpu_uuids": "GPU-a",
        },
    )
    legacy.write_json(raw_dir, preserve_trial=True)
    verified.write_json(raw_dir, preserve_trial=True)

    aggregate_results(raw_dir, output_path)
    with output_path.open(encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))

    assert [row["gpu_telemetry_mapping"] for row in rows] == ["legacy_positional", "uuid_verified"]
    assert rows[1]["gpu_telemetry_schema_version"] == "2"
    assert rows[1]["profiled_gpu_uuids"] == "GPU-a"


def test_audit_telemetry_preserves_raw_results_and_flags_repeated_trial_outlier(tmp_path: Path) -> None:
    config = load_experiment_config(Path("configs/baseline_profiling/E008_gpt2_small_training_baseline.json"))
    raw_dir = tmp_path / "raw"
    original_contents: dict[Path, str] = {}
    for repetition, memory in enumerate((10.0, 10.5, 40.0), start=1):
        result = ExperimentResult.from_config(
            config,
            status="completed",
            started_at=f"2026-01-01T00:00:0{repetition}Z",
            completed_at=f"2026-01-01T00:00:1{repetition}Z",
            runtime_seconds=10.0,
            repetition_number=repetition,
            metrics={
                "nvidia_smi.samples": 20,
                "nvidia_smi.gpu_utilization": 80.0,
                "nvidia_smi.memory_used_gb": memory,
                "nvidia_smi.power_draw_watts": 200.0,
            },
        )
        path = result.write_json(raw_dir, preserve_trial=True)
        original_contents[path] = path.read_text(encoding="utf-8")

    csv_path, markdown_path = audit_telemetry(
        raw_dir,
        tmp_path / "analysis" / "telemetry_audit.csv",
        tmp_path / "analysis" / "telemetry_audit.md",
    )

    with csv_path.open(encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    assert len(rows) == 1
    assert rows[0]["metric"] == "memory_used_gb"
    assert rows[0]["repetition_number"] == "3"
    assert rows[0]["issue"] == "repeated_trial_outlier"
    assert "| legacy_positional | 3 |" in markdown_path.read_text(encoding="utf-8")
    assert {path: path.read_text(encoding="utf-8") for path in original_contents} == original_contents


def test_analyze_results_writes_summary_tables(tmp_path: Path) -> None:
    config = load_experiment_config(Path("configs/scaling_studies/E033_gpt2_small_strong_2gpu.json"))
    raw_dir = tmp_path / "raw"
    processed_path = tmp_path / "processed" / "experiments.csv"
    analysis_dir = tmp_path / "analysis"

    result = ExperimentResult.from_config(
        config,
        status="completed",
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:10:00Z",
        runtime_seconds=600.0,
        metrics={
            "throughput_tokens_sec": 1200.0,
            "throughput_samples_sec": 30.0,
            "throughput_unit": "training_tokens_per_second",
            "memory_used_gb": 42.0,
        },
    )
    result.write_json(raw_dir)
    aggregate_results(raw_dir, processed_path)

    written_paths = analyze_results(processed_path, analysis_dir)

    assert sorted(path.name for path in written_paths) == [
        "communication_summary.csv",
        "data_movement_summary.csv",
        "experiment_rankings.csv",
        "hardware_summary.csv",
        "inference_summary.csv",
        "memory_summary.csv",
        "phase_summary.csv",
        "precision_summary.csv",
        "research_finding_summary.csv",
        "research_question_summary.csv",
        "scaling_summary.csv",
        "strategy_summary.csv",
        "trial_summary.csv",
    ]

    with (analysis_dir / "phase_summary.csv").open(encoding="utf-8", newline="") as file_obj:
        phase_rows = list(csv.DictReader(file_obj))
    with (analysis_dir / "experiment_rankings.csv").open(encoding="utf-8", newline="") as file_obj:
        ranking_rows = list(csv.DictReader(file_obj))
    with (analysis_dir / "research_finding_summary.csv").open(encoding="utf-8", newline="") as file_obj:
        finding_fields = csv.DictReader(file_obj).fieldnames
    with (analysis_dir / "strategy_summary.csv").open(encoding="utf-8", newline="") as file_obj:
        strategy_fields = csv.DictReader(file_obj).fieldnames

    assert phase_rows[0]["phase"] == "Strong scaling"
    assert phase_rows[0]["completed"] == "1"
    assert phase_rows[0]["throughput_unit"] == "training_tokens_per_second"
    assert phase_rows[0]["avg_throughput_tokens_sec"] == "1200.0"
    assert ranking_rows[0]["experiment_id"] == "E033"
    assert ranking_rows[0]["throughput_tokens_per_gpu"] == "600.0"
    assert "scaling_type" in finding_fields
    assert "scaling_type" not in strategy_fields


def test_energy_efficiency_requires_measured_region_scope() -> None:
    common = {
        "status": "completed",
        "condition_id": "E214@l40s",
        "experiment_id": "E214",
        "platform_id": "l40s",
        "comparability": "platform_specific",
        "gpus": "2",
        "metric_total_tokens": "1000",
        "throughput_tokens_sec": "100",
    }
    measured = {**common, "energy_scope": "measured_region", "energy_joules": "200"}
    legacy = {
        **common,
        "condition_id": "E213@l40s",
        "experiment_id": "E213",
        "energy_scope": "legacy_full_window",
        "energy_joules": "800",
        "metric_energy_joules_estimate": "800",
    }

    rows = build_experiment_rankings([measured, legacy])

    by_id = {row["experiment_id"]: row for row in rows}
    assert by_id["E214"]["energy_joules"] == 200.0
    assert by_id["E214"]["tokens_per_kwh"] == 18_000_000.0
    assert by_id["E213"]["energy_joules"] is None
    assert by_id["E213"]["tokens_per_kwh"] is None


def test_hardware_energy_reference_metadata_is_flattened(tmp_path: Path) -> None:
    config = load_experiment_config(Path("configs/hardware_energy_revalidation/E235_l40s_gpt2_training_energy.json"))
    result = ExperimentResult.from_config(
        config,
        status="completed",
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:00:10Z",
        runtime_seconds=10.0,
        metrics={
            "energy_scope": "measured_region",
            "energy_measurement_status": "sufficient",
            "energy_joules_measured_region": 1200.0,
        },
    )
    raw_dir = tmp_path / "raw"
    processed_path = tmp_path / "experiments.csv"
    result.write_json(raw_dir)

    aggregate_results(raw_dir, processed_path)

    with processed_path.open(encoding="utf-8", newline="") as file_obj:
        row = next(csv.DictReader(file_obj))
    assert row["energy_reference_experiment_id"] == "E123"
    assert row["energy_joules"] == "1200.0"


def test_hardware_summary_uses_only_protocol_matched_revalidation_energy() -> None:
    source = {
        "condition_id": "E123@l40s",
        "experiment_id": "E123",
        "platform_id": "l40s",
        "comparability": "exact",
        "phase": "Hardware comparison",
        "workload": "gpt2_small",
        "model": "gpt2",
        "mode": "training",
        "gpus": "1",
        "precision": "bf16",
        "strategy": "none",
        "batch_size": "16",
        "parameter_sequence_length": "1024",
        "parameter_stack": "pytorch_single_gpu",
        "model_revision": "revision-a",
        "dataset_revision": "synthetic_tokens_v1",
        "measured_iterations": "100",
        "throughput_unit": "training_tokens_per_second",
        "throughput_tokens_sec": "45000",
        "status": "completed",
    }
    matched = {
        **source,
        "condition_id": "E235@l40s",
        "experiment_id": "E235",
        "comparability": "platform_specific",
        "phase": "Hardware energy revalidation",
        "energy_reference_experiment_id": "E123",
        "throughput_tokens_sec": "99999",
        "energy_joules": "12000",
    }
    mismatched = {
        **matched,
        "condition_id": "E999@l40s",
        "experiment_id": "E999",
        "batch_size": "8",
        "energy_joules": "1",
    }

    summary = build_analysis_tables([source, matched, mismatched])["hardware_summary"]

    assert len(summary) == 1
    assert summary[0]["condition_id"] == "E123@l40s"
    assert summary[0]["avg_throughput_tokens_sec"] == 45000.0
    assert summary[0]["avg_energy_joules"] == 12000.0
    assert summary[0]["energy_trials"] == 1
    assert summary[0]["energy_evidence_source"] == "E235@l40s"


def test_hardware_energy_revalidation_gate_requires_three_sufficient_trials(tmp_path: Path) -> None:
    result_root = tmp_path / "raw" / "l40s"
    references = {"E235": "E123", "E236": "E124", "E237": "E127", "E238": "E128"}
    for experiment_id, reference_id in references.items():
        experiment_dir = result_root / experiment_id
        experiment_dir.mkdir(parents=True)
        for repetition in range(1, 4):
            record = {
                "status": "completed",
                "repetition_number": repetition,
                "config": {"metadata": {"energy_reference_experiment_id": reference_id}},
                "metrics": {
                    "energy_scope": "measured_region",
                    "energy_measurement_status": "sufficient",
                    "energy_joules_measured_region": 100.0 + repetition,
                    "power_draw_watts": 200.0,
                    "nvidia_smi.measurement_samples": 4,
                    "nvidia_smi.gpu_mapping_method": "cuda_uuid",
                    "nvidia_smi.telemetry_schema_version": "3",
                },
                "environment": {"observed_gpu_model": "NVIDIA L40S"},
            }
            (experiment_dir / f"trial-{repetition}.json").write_text(json.dumps(record), encoding="utf-8")

    issues, accepted = verify_hardware_energy_results(result_root)

    assert not issues
    assert set(accepted) == set(references)
    assert all(result["trials"] == 3 for result in accepted.values())


def test_hardware_energy_revalidation_gate_rejects_insufficient_samples(tmp_path: Path) -> None:
    result_root = tmp_path / "raw" / "l40s"
    experiment_dir = result_root / "E235"
    experiment_dir.mkdir(parents=True)
    record = {
        "status": "completed",
        "repetition_number": 1,
        "config": {"metadata": {"energy_reference_experiment_id": "E123"}},
        "metrics": {
            "energy_scope": "full_window",
            "energy_measurement_status": "insufficient_samples",
            "nvidia_smi.measurement_samples": 1,
            "nvidia_smi.gpu_mapping_method": "cuda_uuid",
            "nvidia_smi.telemetry_schema_version": "3",
        },
        "environment": {"observed_gpu_model": "NVIDIA L40S"},
    }
    (experiment_dir / "trial-1.json").write_text(json.dumps(record), encoding="utf-8")

    issues, _ = verify_hardware_energy_results(result_root)

    assert any("insufficient samples" in issue for issue in issues)


def test_dashboard_static_assets_exist() -> None:
    dashboard_files = [
        Path("dashboard/index.html"),
        Path("dashboard/analysis.html"),
        Path("dashboard/experiments-data.json"),
        Path("dashboard/experiment-details.json"),
    ]

    for dashboard_file in dashboard_files:
        assert dashboard_file.exists()

    html = Path("dashboard/index.html").read_text(encoding="utf-8")
    analysis = Path("dashboard/analysis.html").read_text(encoding="utf-8")

    assert 'href="analysis.html"' in html
    assert 'href="index.html"' in analysis
    assert 'id="themeTree"' in html
    assert 'id="drawer"' in html
    assert 'id="nerds"' in analysis
    assert 'id="snapshotLine"' in html
    assert 'id="snapshotLine"' in analysis


def test_hpc_smoke_runbook_exists() -> None:
    runbook = Path("docs/hpc_smoke_runbook.md").read_text(encoding="utf-8")

    assert "sbatch slurm/smoke/gpt2_small_smoke.sbatch" in runbook
    assert "python -m hpc_ai_perf.cli refresh" in runbook
    assert "make serve-dashboard" in runbook


def test_current_status_snapshot_exists() -> None:
    status = Path("docs/current_status.md").read_text(encoding="utf-8")

    assert "Source matrix configs | 200" in status
    assert "L40S supplemental capacity probes | 9" in status
    assert "Cross-platform telemetry revalidation anchors | 5" in status
    assert "Smoke-Ready Workloads" in status
    assert "python -m hpc_ai_perf.cli refresh" in status


def test_hpc_submission_batches_doc_exists() -> None:
    batches = Path("docs/hpc_submission_batches.md").read_text(encoding="utf-8")

    assert "make render-slurm-env-validation" in batches
    assert "make render-slurm-baseline-must-have" in batches
    assert "make plan-slurm-first-batches" in batches
    assert "make init-slurm-first-status" in batches
    assert "record-slurm-job" in batches
    assert "E008-E011" in batches


def test_readme_documents_hpc_submission_workflow() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "## HPC Submission Workflow" in readme
    assert "slurm/profiles/cluster_l40s.json" in readme
    assert "make submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=0" in readme
    assert "make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=0" in readme
    assert "make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=0" in readme
    assert "make stage-platform-wave PLATFORM=l40s PLATFORM_WAVE=0" in readme


def test_slurm_status_make_targets_do_not_rerender_scripts() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "init-slurm-env-status:\n\t$(PYTHON) -m hpc_ai_perf.cli init-slurm-status" in makefile
    assert "init-slurm-baseline-status:\n\t$(PYTHON) -m hpc_ai_perf.cli init-slurm-status" in makefile


def test_readiness_report_writes_submission_summary(tmp_path: Path) -> None:
    report_path = write_readiness_report(tmp_path / "readiness_report.md")
    report = report_path.read_text(encoding="utf-8")

    assert "Source matrix configs discovered: 200" in report
    assert "Supplemental platform probes discovered: 99" in report
    assert "Cross-platform telemetry anchors discovered: 5" in report
    assert "Total E-number configs discovered: 304" in report
    assert "`gpt2_small`" in report
    assert "slurm/smoke/gpt2_small_smoke.sbatch" in report
    assert "Submission Gate" in report
    assert "Research-readiness blockers: 7" in report
    assert "Deferred scope rows: 11" in report


def test_track_experiment_progress_writes_reports(tmp_path: Path) -> None:
    config = load_experiment_config(Path("configs/scaling_studies/E033_gpt2_small_strong_2gpu.json"))
    raw_dir = tmp_path / "raw"
    output_csv = tmp_path / "processed" / "experiment_tracker.csv"
    output_markdown = tmp_path / "processed" / "experiment_tracker.md"

    result = ExperimentResult.from_config(
        config,
        status="completed",
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:10:00Z",
        runtime_seconds=600.0,
        metrics={
            "throughput_tokens_sec": 1200.0,
            "throughput_unit": "training_tokens_per_second",
            "measured_seconds": 590.0,
            "gpu_hours": 1180.0 / 3600,
            "total_tokens": 708000,
        },
        environment={
            "config_sha256": "a" * 64,
            "git_dirty": False,
            "observed_gpu_model": "NVIDIA H100 SXM 80GB",
        },
    )
    result.write_json(raw_dir)

    csv_path, markdown_path = track_experiment_progress("configs", raw_dir, output_csv, output_markdown)

    with csv_path.open(encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    e033 = next(row for row in rows if row["experiment_id"] == "E033")
    e034 = next(row for row in rows if row["experiment_id"] == "E034")
    e086 = next(row for row in rows if row["experiment_id"] == "E086")
    e201 = next(row for row in rows if row["experiment_id"] == "E201")

    assert csv_path == output_csv
    assert markdown_path == output_markdown
    assert len(rows) == 304
    assert e033["status"] == "Completed"
    assert e033["runtime_seconds"] == "600.0"
    assert e034["status"] == "Configured"
    assert e086["status"] == "Configured"
    assert e201["status"] == "Configured"
    assert "| Completed | 1 |" in markdown_path.read_text(encoding="utf-8")


def test_track_dev_experiment_progress_writes_separate_reports(tmp_path: Path) -> None:
    config = load_experiment_config(Path("configs/dev/dev_synthetic_tokens_8192.json"))
    raw_dir = tmp_path / "raw"
    output_csv = tmp_path / "processed" / "dev_experiment_tracker.csv"
    output_markdown = tmp_path / "processed" / "dev_experiment_tracker.md"

    result = ExperimentResult.from_config(
        config,
        status="completed",
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:00:01Z",
        runtime_seconds=1.0,
        metrics={"throughput_tokens_sec": 8192.0},
    )
    result.write_json(raw_dir)

    csv_path, markdown_path = track_experiment_progress(
        "configs",
        raw_dir,
        output_csv,
        output_markdown,
        profile="dev",
    )

    with csv_path.open(encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    completed = next(row for row in rows if row["experiment_id"] == "dev_synthetic_tokens_8192")
    configured = next(row for row in rows if row["experiment_id"] == "dev_synthetic_tokens_32768")

    assert csv_path == output_csv
    assert markdown_path == output_markdown
    assert len(rows) == 15
    assert completed["status"] == "Completed"
    assert completed["runtime_seconds"] == "1.0"
    assert configured["status"] == "Configured"
    assert "| Completed | 1 |" in markdown_path.read_text(encoding="utf-8")


def test_experiment_configs_validate() -> None:
    assert validate_experiment_configs("configs") == []


def test_config_validation_rejects_unknown_workload_parameters(tmp_path: Path) -> None:
    config = load_experiment_config(Path("configs/local/synthetic_smoke.json")).to_dict()
    config["workload"]["parameters"]["token_per_step_typo"] = 10
    config_path = tmp_path / "bad_parameter.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    issues = validate_experiment_configs(tmp_path)

    assert any("unknown workload parameter(s): token_per_step_typo" in issue.message for issue in issues)


def test_research_preflight_rejects_unbuilt_backend_artifacts() -> None:
    config = load_experiment_config(Path("configs/inference/E131_gpt2_small_framework_tensorrt_llm.json"))

    assert "TensorRT-LLM config requires a built engine_path" in capability_issues(config)


def test_vllm_fp8_requires_an_explicit_reproducible_contract() -> None:
    config = load_experiment_config(Path("configs/inference/E151_gpt2_small_precision_fp8.json"))
    parameters = dict(config.workload.parameters)
    parameters.pop("quantization_method")
    incomplete = replace(config, workload=replace(config.workload, parameters=parameters))

    assert "vLLM FP8 inference requires quantization_method=fp8_w8a8_dynamic" in capability_issues(incomplete)


@pytest.mark.parametrize(
    ("fp8_path", "control_path"),
    [
        (
            "configs/precision_studies/E086_gpt2_small_training_fp8.json",
            "configs/precision_studies/E231_l40s_gpt2_transformer_engine_bf16_control.json",
        ),
        (
            "configs/precision_studies/E093_distilbert_training_fp8.json",
            "configs/precision_studies/E232_l40s_distilbert_transformer_engine_bf16_control.json",
        ),
    ],
)
def test_transformer_engine_fp8_training_conditions_match_their_bf16_controls(
    fp8_path: str,
    control_path: str,
) -> None:
    fp8 = load_experiment_config(Path(fp8_path))
    control = load_experiment_config(Path(control_path))

    assert capability_issues(fp8) == []
    assert capability_issues(control) == []
    assert fp8.workload.parameters["precision_backend"] == "transformer_engine"
    assert fp8.workload.parameters["quantization_method"] == "fp8_delayed_scaling_hybrid"
    assert fp8.workload.parameters["base_compute_dtype"] == "bf16"
    assert fp8.workload.parameters["comparison_control_id"] == control.experiment_id
    assert control.workload.parameters["precision_backend"] == "transformer_engine"
    assert control.workload.parameters["comparison_treatment_id"] == fp8.experiment_id
    assert control.runtime.precision == "bf16"
    assert fp8.workload.name == control.workload.name
    assert fp8.workload.model == control.workload.model
    assert fp8.workload.model_revision == control.workload.model_revision
    assert fp8.workload.parameters["batch_size"] == control.workload.parameters["batch_size"]
    assert fp8.workload.parameters["sequence_length"] == control.workload.parameters["sequence_length"]
    assert fp8.runtime.max_steps == control.runtime.max_steps


def test_transformer_engine_fp8_training_rejects_an_implicit_backend() -> None:
    config = load_experiment_config(Path("configs/precision_studies/E086_gpt2_small_training_fp8.json"))
    parameters = dict(config.workload.parameters)
    parameters.pop("precision_backend")
    incomplete = replace(config, workload=replace(config.workload, parameters=parameters))

    assert "FP8 training requires precision_backend=transformer_engine" in capability_issues(incomplete)


@pytest.mark.parametrize(
    ("fp8_path", "control_path"),
    [
        (
            "configs/inference/E151_gpt2_small_precision_fp8.json",
            "configs/inference/E130_gpt2_small_framework_vllm.json",
        ),
        (
            "configs/inference/E199_tinyllama_precision_fp8.json",
            "configs/inference/E178_tinyllama_framework_vllm.json",
        ),
    ],
)
def test_vllm_fp8_conditions_match_their_bf16_controls(fp8_path: str, control_path: str) -> None:
    fp8 = load_experiment_config(Path(fp8_path))
    control = load_experiment_config(Path(control_path))
    controlled_parameters = ("batch_size", "concurrent_requests", "prompt_length", "max_new_tokens")

    assert fp8.workload.parameters["comparison_control_id"] == control.experiment_id
    assert fp8.workload.name == control.workload.name
    assert fp8.workload.model == control.workload.model
    assert fp8.workload.model_revision == control.workload.model_revision
    assert fp8.hardware.gpus == control.hardware.gpus
    assert fp8.runtime.max_steps == control.runtime.max_steps
    for parameter in controlled_parameters:
        assert fp8.workload.parameters[parameter] == control.workload.parameters[parameter]


def test_research_preflight_has_only_documented_execution_blockers() -> None:
    blocked = {entry.config.experiment_id for entry in discover_experiment_configs() if capability_issues(entry.config)}

    assert blocked == {
        "E089",
        "E096",
        "E131",
        "E152",
        "E154",
        "E155",
        "E175",
        "E176",
        "E179",
        "E200",
        "E294",
        "E295",
        "E296",
        "E297",
        "E298",
        "E299",
        "E300",
    }


def test_research_preflight_excludes_explicitly_deferred_scope() -> None:
    issues = validate_experiment_configs("configs", research_ready=True)
    assert {issue.path.stem[:4] for issue in issues} == {f"E{index}" for index in range(294, 301)}
    assert all(issue.message == "matrix transformer config must pin workload.model_revision" for issue in issues)


def test_matrix_transformer_models_are_pinned() -> None:
    awaiting_revision = {f"E{index}" for index in range(294, 301)}
    for entry in discover_experiment_configs():
        if entry.config.workload.name == "environment_validation":
            continue
        if entry.config.experiment_id in awaiting_revision:
            assert entry.config.workload.model_revision is None
            continue
        assert entry.config.workload.model_revision is not None
        assert len(entry.config.workload.model_revision) == 40


def test_matrix_configs_have_normalized_research_traceability() -> None:
    for entry in discover_experiment_configs():
        research_questions = entry.config.metadata.get("research_questions")
        analysis_tasks = entry.config.metadata.get("analysis_tasks")

        assert research_questions
        assert analysis_tasks
        assert all(identifier.startswith("RQ") for identifier in research_questions)
        assert all(identifier.startswith("A") for identifier in analysis_tasks)


def test_weak_scaling_efficiency_uses_constant_runtime_ideal() -> None:
    rows = [
        {
            "status": "completed",
            "phase": "Weak scaling",
            "workload": "gpt2_small",
            "model": "gpt2",
            "precision": "bf16",
            "strategy": "ddp",
            "scaling_type": "weak",
            "gpus": "1",
            "metric_measured_seconds": "10",
            "throughput_tokens_sec": "100",
        },
        {
            "status": "completed",
            "phase": "Weak scaling",
            "workload": "gpt2_small",
            "model": "gpt2",
            "precision": "bf16",
            "strategy": "ddp",
            "scaling_type": "weak",
            "gpus": "4",
            "metric_measured_seconds": "10",
            "throughput_tokens_sec": "400",
        },
    ]

    summary = build_scaling_summary(rows)

    assert summary[1]["speedup"] == 4.0
    assert summary[1]["scaling_efficiency"] == 1.0


def test_trial_summary_requires_three_trials_and_reports_confidence_interval() -> None:
    base = {
        "experiment_id": "E008",
        "phase": "Baseline profiling",
        "workload": "gpt2_small",
        "mode": "training",
        "status": "completed",
    }
    rows = [
        {**base, "runtime_seconds": "10", "throughput_tokens_sec": "100"},
        {**base, "runtime_seconds": "12", "throughput_tokens_sec": "110"},
        {**base, "runtime_seconds": "14", "throughput_tokens_sec": "120"},
    ]

    summary = build_trial_summary(rows)[0]

    assert summary["evidence_status"] == "sufficient"
    assert summary["required_trials"] == 3
    assert summary["runtime_median_seconds"] == 12.0
    assert summary["throughput_tokens_median"] == 110.0
    assert summary["throughput_tokens_ci95_low"] < 110.0
    assert summary["throughput_tokens_ci95_high"] > 110.0

    summary = build_trial_summary(rows[:2])[0]
    assert summary["evidence_status"] == "insufficient_data"


def test_analysis_never_pools_l40s_and_h100_conditions() -> None:
    base = {
        "experiment_id": "E008",
        "phase": "Baseline profiling",
        "priority": "Must Have",
        "workload": "gpt2_small",
        "model": "gpt2",
        "mode": "training",
        "precision": "bf16",
        "strategy": "none",
        "gpus": "1",
        "status": "completed",
        "throughput_unit": "training_tokens_per_second",
        "comparability": "exact",
        "required_trials": "1",
    }
    rows = [
        {
            **base,
            "condition_id": "E008@l40s",
            "platform_id": "l40s",
            "runtime_seconds": "20",
            "throughput_tokens_sec": "100",
        },
        {
            **base,
            "condition_id": "E008@h100",
            "platform_id": "h100",
            "runtime_seconds": "10",
            "throughput_tokens_sec": "200",
        },
    ]

    phase_rows = build_phase_summary(rows)
    trial_rows = build_trial_summary(rows)

    assert {row["platform_id"] for row in phase_rows} == {"l40s", "h100"}
    assert {row["condition_id"] for row in trial_rows} == {"E008@l40s", "E008@h100"}
    assert all(row["completed_trials"] == 1 for row in trial_rows)
    assert all(row["evidence_status"] == "sufficient" for row in trial_rows)


def test_every_analysis_table_keeps_platform_groups_separate() -> None:
    def result_row(
        experiment_id: str,
        platform_id: str,
        phase: str,
        *,
        gpus: str = "1",
        mode: str = "training",
    ) -> dict[str, str]:
        is_h100 = platform_id == "h100"
        return {
            "condition_id": f"{experiment_id}@{platform_id}",
            "experiment_id": experiment_id,
            "platform_id": platform_id,
            "comparability": "exact",
            "phase": phase,
            "priority": "Must Have",
            "workload": "gpt2_small",
            "model": "gpt2",
            "mode": mode,
            "precision": "bf16",
            "strategy": "ddp",
            "gpus": gpus,
            "status": "completed",
            "required_trials": "1",
            "throughput_unit": "training_tokens_per_second",
            "runtime_seconds": "10" if is_h100 else "20",
            "metric_measured_seconds": "9" if is_h100 else "18",
            "throughput_tokens_sec": "200" if is_h100 else "100",
            "throughput_samples_sec": "20" if is_h100 else "10",
            "memory_used_gb": "20" if is_h100 else "30",
            "platform": "lovelace_h100" if is_h100 else "lovelace_l40s",
            "gpu_model": "NVIDIA H100 SXM 80GB" if is_h100 else "NVIDIA L40S 48GB",
            "framework": "pytorch",
            "batch_size": "1",
            "metric_allreduce_time_seconds": "0.1",
            "metric_nccl_bandwidth_gbps": "100",
            "metric_data_loading_seconds": "1",
            "metric_cpu_to_gpu_transfer_seconds": "0.5",
            "metric_memory_reserved_gb": "32",
            "metric_step_time_mean_ms": "5",
            "metric_loss": "1",
            "parameter_num_workers": "4",
            "parameter_pinned_memory": "true",
            "parameter_prefetch_factor": "2",
            "parameter_data_movement_study_type": "num_workers",
            "parameter_memory_study_type": "micro_batch",
            "parameter_micro_batch_size": "1",
            "parameter_gradient_accumulation_steps": "1",
            "global_batch_size": "1",
        }

    specifications = [
        ("E008", "Baseline profiling", "1", "training"),
        ("E032", "Strong scaling", "1", "training"),
        ("E034", "Strong scaling", "4", "training"),
        ("E047", "Distributed training strategy", "2", "training"),
        ("E063", "Memory optimisation", "1", "training"),
        ("E097", "Data movement", "1", "training"),
        ("E113", "Communication analysis", "2", "training"),
        ("E121", "Hardware comparison", "1", "training"),
        ("E129", "Inference framework", "1", "inference"),
    ]
    rows = [
        result_row(experiment_id, platform_id, phase, gpus=gpus, mode=mode)
        for experiment_id, phase, gpus, mode in specifications
        for platform_id in ["l40s", "h100"]
    ]

    tables = build_analysis_tables(rows)

    assert set(tables) == {
        "phase_summary",
        "experiment_rankings",
        "scaling_summary",
        "precision_summary",
        "trial_summary",
        "strategy_summary",
        "communication_summary",
        "data_movement_summary",
        "memory_summary",
        "inference_summary",
        "hardware_summary",
        "research_question_summary",
        "research_finding_summary",
    }
    for table_name, table_rows in tables.items():
        assert table_rows
        assert {row["platform_id"] for row in table_rows} == {"l40s", "h100"}
        if table_name != "research_question_summary":
            assert {row["comparability"] for row in table_rows} == {"exact"}

    assert len(tables["phase_summary"]) == 16
    assert len(tables["scaling_summary"]) == 4
    assert len(tables["precision_summary"]) == 4
    assert len(tables["strategy_summary"]) == 2
    assert len(tables["communication_summary"]) == 2
    assert len(tables["data_movement_summary"]) == 2
    assert {row["study_type"] for row in tables["data_movement_summary"]} == {"num_workers"}
    assert len(tables["memory_summary"]) == 2
    assert len(tables["inference_summary"]) == 2
    assert len(tables["hardware_summary"]) == 2
    assert len(tables["trial_summary"]) == len(rows)


def test_research_question_tables_report_coverage_and_matched_strategy_tradeoffs() -> None:
    def row(experiment_id: str, strategy: str, throughput: str, memory: str) -> dict[str, str]:
        return {
            "condition_id": f"{experiment_id}@l40s",
            "experiment_id": experiment_id,
            "platform_id": "l40s",
            "comparability": "exact",
            "phase": "Distributed training strategy",
            "workload": "gpt2_small",
            "model": "gpt2",
            "mode": "training",
            "precision": "bf16",
            "strategy": strategy,
            "gpus": "2",
            "status": "completed",
            "required_trials": "1",
            "throughput_unit": "training_tokens_per_second",
            "global_batch_size": "32",
            "per_gpu_batch_size": "16",
            "parameter_sequence_length": "1024",
            "parameter_gradient_accumulation_steps": "1",
            "throughput_tokens_sec": throughput,
            "memory_used_gb": memory,
        }

    tables = build_analysis_tables(
        [
            row("E047", "ddp", "100", "30"),
            row("E052", "zero_stage_1", "120", "24"),
        ]
    )

    coverage = next(
        entry
        for entry in tables["research_question_summary"]
        if entry["research_question"] == "RQ2" and entry["platform_id"] == "l40s"
    )
    assert coverage["evidence_status"] == "sufficient"
    assert coverage["direct_conditions"] == 2
    findings = tables["research_finding_summary"]
    zero = next(entry for entry in findings if entry["strategy"] == "zero_stage_1")
    ddp = next(entry for entry in findings if entry["strategy"] == "ddp")
    assert "Pareto frontier" in zero["finding"]
    assert "Dominated" in ddp["finding"]
    assert zero["source_condition_ids"] == "E052@l40s"


def test_rq1_findings_preserve_strong_and_weak_scaling_series() -> None:
    def row(experiment_id: str, phase: str, scaling_type: str, gpus: str, runtime: str) -> dict[str, str]:
        return {
            "condition_id": f"{experiment_id}@h100",
            "experiment_id": experiment_id,
            "platform_id": "h100",
            "comparability": "exact",
            "phase": phase,
            "workload": "distilbert",
            "model": "distilbert",
            "mode": "training",
            "precision": "bf16",
            "strategy": "ddp",
            "scaling_type": scaling_type,
            "gpus": gpus,
            "status": "completed",
            "required_trials": "1",
            "throughput_unit": "training_tokens_per_second",
            "runtime_seconds": runtime,
            "throughput_tokens_sec": str(float(gpus) * 100),
        }

    rows = [
        row("E038", "Strong scaling", "strong", "1", "10"),
        row("E039", "Strong scaling", "strong", "2", "6"),
        row("E041", "Weak scaling", "weak", "1", "10"),
        row("E042", "Weak scaling", "weak", "2", "10"),
    ]

    findings = build_analysis_tables(rows)["research_finding_summary"]
    rq1 = [finding for finding in findings if finding["research_question"] == "RQ1"]

    assert {finding["scaling_type"] for finding in rq1} == {"strong", "weak"}
    assert next(finding for finding in rq1 if finding["scaling_type"] == "strong")["source_condition_ids"] == (
        "E038@h100; E039@h100"
    )
    assert next(finding for finding in rq1 if finding["scaling_type"] == "weak")["source_condition_ids"] == (
        "E041@h100; E042@h100"
    )


def test_research_question_summary_marks_incomplete_energy_as_partial() -> None:
    row = {
        "condition_id": "E008@l40s",
        "experiment_id": "E008",
        "platform_id": "l40s",
        "comparability": "exact",
        "phase": "Baseline profiling",
        "status": "completed",
        "required_trials": "1",
    }

    summary = next(
        entry
        for entry in build_analysis_tables([row])["research_question_summary"]
        if entry["research_question"] == "RQ5" and entry["platform_id"] == "l40s"
    )

    assert summary["energy_trials"] == 0
    assert summary["evidence_status"] == "partial_energy_evidence"


def test_rq6_matches_hardware_protocols_with_different_experiment_ids() -> None:
    def row(experiment_id: str, platform_id: str) -> dict[str, str]:
        return {
            "condition_id": f"{experiment_id}@{platform_id}",
            "experiment_id": experiment_id,
            "platform_id": platform_id,
            "comparability": "exact",
            "phase": "Hardware comparison",
            "workload": "gpt2_small",
            "model": "gpt2",
            "mode": "training",
            "precision": "bf16",
            "gpus": "1",
            "batch_size": "16",
            "global_batch_size": "16",
            "per_gpu_batch_size": "16",
            "parameter_sequence_length": "1024",
            "throughput_unit": "training_tokens_per_second",
            "status": "completed",
            "required_trials": "1",
        }

    summaries = build_analysis_tables([row("E121", "h100"), row("E123", "l40s")])["research_question_summary"]
    rq6 = [entry for entry in summaries if entry["research_question"] == "RQ6"]

    assert {entry["platform_id"] for entry in rq6} == {"h100", "l40s"}
    assert {entry["evidence_status"] for entry in rq6} == {"sufficient"}
    assert all("matched protocol signatures" in entry["scope_note"] for entry in rq6)


def test_analysis_rejects_ambiguous_processed_matrix_rows() -> None:
    row = {
        "experiment_id": "E008",
        "condition_id": "",
        "platform_id": "",
        "comparability": "",
        "status": "completed",
    }

    with pytest.raises(ValueError, match="invalid platform_id"):
        build_analysis_tables([row])


def test_result_validation_reports_bad_record(tmp_path: Path) -> None:
    bad_result = tmp_path / "bad.json"
    bad_result.write_text('{"experiment_id": "E999", "status": "done", "runtime_seconds": -1}\n', encoding="utf-8")

    issues = validate_result_records(tmp_path)

    assert issues
    assert any("missing required result field" in issue.message for issue in issues)
    assert any("unsupported status" in issue.message for issue in issues)
    assert any("runtime_seconds" in issue.message for issue in issues)


def test_refresh_outputs_rebuilds_derived_files(tmp_path: Path) -> None:
    config = load_experiment_config(Path("configs/scaling_studies/E033_gpt2_small_strong_2gpu.json"))
    raw_dir = tmp_path / "raw"
    processed_results = tmp_path / "processed" / "experiments.csv"
    analysis_dir = tmp_path / "analysis"
    tracker_csv = tmp_path / "processed" / "experiment_tracker.csv"
    tracker_markdown = tmp_path / "processed" / "experiment_tracker.md"
    readiness_report = tmp_path / "processed" / "readiness_report.md"

    config_sha256 = hashlib.sha256(
        json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = ExperimentResult.from_config(
        config,
        status="completed",
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:10:00Z",
        runtime_seconds=600.0,
        metrics={
            "throughput_tokens_sec": 1200.0,
            "throughput_unit": "training_tokens_per_second",
            "measured_seconds": 590.0,
            "gpu_hours": 1180.0 / 3600,
            "total_tokens": 708000,
        },
        environment={
            "config_sha256": config_sha256,
            "git_dirty": False,
            "git_commit": "deadbeef",
            "command": "python -m hpc_ai_perf.cli run --config E033.json",
            "observed_gpu_model": "NVIDIA H100 SXM 80GB",
        },
    )
    result.write_json(raw_dir)

    _refresh_outputs(
        config_root="configs",
        result_dir=str(raw_dir),
        processed_results=str(processed_results),
        analysis_dir=str(analysis_dir),
        tracker_csv=str(tracker_csv),
        tracker_markdown=str(tracker_markdown),
        readiness_report=str(readiness_report),
        skip_validation=False,
    )

    assert processed_results.exists()
    assert (analysis_dir / "phase_summary.csv").exists()
    assert tracker_csv.exists()
    assert tracker_markdown.exists()
    assert readiness_report.exists()
