import csv
from pathlib import Path

from hpc_ai_perf.analysis import analyze_results
from hpc_ai_perf.cli import _load_slurm_profile, _refresh_outputs, _render_slurm_bulk
from hpc_ai_perf.config import discover_experiment_configs, filter_config_entries, load_experiment_config
from hpc_ai_perf.readiness import write_readiness_report
from hpc_ai_perf.results import ExperimentResult, aggregate_results
from hpc_ai_perf.runners import LocalBenchmarkRunner
from hpc_ai_perf.slurm import (
    build_job_status_report,
    build_job_status_rows,
    build_submission_plan,
    record_job_submission,
    render_slurm_script,
    write_job_status_csv,
    write_job_status_report,
    write_slurm_script,
    write_submission_plan,
)
from hpc_ai_perf.tracking import track_experiment_progress
from hpc_ai_perf.validation import validate_experiment_configs, validate_result_records
from hpc_ai_perf.workloads import (
    GPT2SmallWorkload,
    TinyLlamaWorkload,
    TransformerWorkload,
    WorkloadRegistry,
    default_registry,
)
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

    assert len(entries) == 200
    assert experiment_ids == {f"E{index:03d}" for index in range(1, 201)}


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
    assert "python -m hpc_ai_perf.cli run --config configs/scaling_studies/E033_gpt2_small_strong_2gpu.json" in script


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

    assert generated == ["E001.sbatch", "E002.sbatch", "E003.sbatch", "E004.sbatch", "E006.sbatch"]


def test_render_slurm_bulk_with_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        """{
  "partition": "gpu",
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


def test_slurm_profile_example_loads() -> None:
    profile = _load_slurm_profile("slurm/profiles/example.json")

    assert profile["partition"] == "gpu"
    assert "cuda/12.4" in profile["modules"]


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
    assert result_path.exists()


def test_default_registry_includes_transformer_workload_skeletons() -> None:
    registry = default_registry()

    assert registry.names() == ["distilbert", "gpt2_small", "synthetic", "tinyllama"]


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


def test_tinyllama_training_is_explicitly_unsupported() -> None:
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

    try:
        TinyLlamaWorkload().run(training_config)
    except NotImplementedError as exc:
        assert "TinyLlama training is not implemented" in str(exc)
    else:
        raise AssertionError("Expected TinyLlama training to be unsupported")


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
        metrics={"throughput_tokens_sec": 1234.5},
    )
    result.write_json(raw_dir)

    csv_path = aggregate_results(raw_dir, output_path)

    with csv_path.open(encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))

    assert csv_path == output_path
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "E033"
    assert rows[0]["phase"] == "Strong scaling"
    assert rows[0]["priority"] == "Must Have"
    assert rows[0]["strategy"] == "ddp"
    assert rows[0]["metric_throughput_tokens_sec"]


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
        metrics={"throughput_tokens_sec": 1200.0, "throughput_samples_sec": 30.0, "memory_used_gb": 42.0},
    )
    result.write_json(raw_dir)
    aggregate_results(raw_dir, processed_path)

    written_paths = analyze_results(processed_path, analysis_dir)

    assert sorted(path.name for path in written_paths) == [
        "experiment_rankings.csv",
        "phase_summary.csv",
        "precision_summary.csv",
        "scaling_summary.csv",
    ]

    with (analysis_dir / "phase_summary.csv").open(encoding="utf-8", newline="") as file_obj:
        phase_rows = list(csv.DictReader(file_obj))
    with (analysis_dir / "experiment_rankings.csv").open(encoding="utf-8", newline="") as file_obj:
        ranking_rows = list(csv.DictReader(file_obj))

    assert phase_rows[0]["phase"] == "Strong scaling"
    assert phase_rows[0]["completed"] == "1"
    assert phase_rows[0]["avg_throughput_tokens_sec"] == "1200.0"
    assert ranking_rows[0]["experiment_id"] == "E033"
    assert ranking_rows[0]["throughput_tokens_per_gpu"] == "600.0"


def test_dashboard_static_assets_exist() -> None:
    dashboard_files = [
        Path("dashboard/index.html"),
        Path("dashboard/styles.css"),
        Path("dashboard/app.js"),
    ]

    for dashboard_file in dashboard_files:
        assert dashboard_file.exists()

    html = Path("dashboard/index.html").read_text(encoding="utf-8")
    script = Path("dashboard/app.js").read_text(encoding="utf-8")

    assert 'id="rankingTable"' in html
    assert "../results/processed/experiments.csv" in script
    assert "../results/analysis/phase_summary.csv" in script
    assert "../results/processed/readiness_report.md" in script
    assert 'id="readinessSummary"' in html


def test_hpc_smoke_runbook_exists() -> None:
    runbook = Path("docs/hpc_smoke_runbook.md").read_text(encoding="utf-8")

    assert "sbatch slurm/smoke/gpt2_small_smoke.sbatch" in runbook
    assert "python -m hpc_ai_perf.cli refresh" in runbook
    assert "make serve-dashboard" in runbook


def test_current_status_snapshot_exists() -> None:
    status = Path("docs/current_status.md").read_text(encoding="utf-8")

    assert "Total matrix configs | 200" in status
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
    assert "SLURM_PROFILE=slurm/profiles/cluster.local.json make plan-slurm-first-batches" in readme
    assert "make init-slurm-first-status" in readme
    assert "record-slurm-job" in readme


def test_readiness_report_writes_submission_summary(tmp_path: Path) -> None:
    report_path = write_readiness_report(tmp_path / "readiness_report.md")
    report = report_path.read_text(encoding="utf-8")

    assert "Matrix configs discovered: 200" in report
    assert "`gpt2_small`" in report
    assert "slurm/smoke/gpt2_small_smoke.sbatch" in report
    assert "Submission Gate" in report


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
        metrics={"throughput_tokens_sec": 1200.0},
    )
    result.write_json(raw_dir)

    csv_path, markdown_path = track_experiment_progress("configs", raw_dir, output_csv, output_markdown)

    with csv_path.open(encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    e033 = next(row for row in rows if row["experiment_id"] == "E033")
    e034 = next(row for row in rows if row["experiment_id"] == "E034")

    assert csv_path == output_csv
    assert markdown_path == output_markdown
    assert len(rows) == 200
    assert e033["status"] == "Completed"
    assert e033["runtime_seconds"] == "600.0"
    assert e034["status"] == "Configured"
    assert "| Completed | 1 |" in markdown_path.read_text(encoding="utf-8")


def test_experiment_configs_validate() -> None:
    assert validate_experiment_configs("configs") == []


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

    result = ExperimentResult.from_config(
        config,
        status="completed",
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:10:00Z",
        runtime_seconds=600.0,
        metrics={"throughput_tokens_sec": 1200.0},
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
