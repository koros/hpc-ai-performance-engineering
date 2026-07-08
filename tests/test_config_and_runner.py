import csv
from pathlib import Path

from hpc_ai_perf.analysis import analyze_results
from hpc_ai_perf.cli import _render_slurm_bulk
from hpc_ai_perf.config import discover_experiment_configs, filter_config_entries, load_experiment_config
from hpc_ai_perf.results import ExperimentResult, aggregate_results
from hpc_ai_perf.runners import LocalBenchmarkRunner
from hpc_ai_perf.slurm import render_slurm_script, write_slurm_script
from hpc_ai_perf.tracking import track_experiment_progress
from hpc_ai_perf.validation import validate_experiment_configs, validate_result_records


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


def test_write_slurm_script(tmp_path: Path) -> None:
    script_path = write_slurm_script("#!/bin/bash\n", tmp_path, "E001")

    assert script_path == tmp_path / "E001.sbatch"
    assert script_path.read_text(encoding="utf-8") == "#!/bin/bash\n"


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
    )

    generated = sorted(path.name for path in tmp_path.glob("*.sbatch"))

    assert generated == ["E001.sbatch", "E002.sbatch", "E003.sbatch", "E004.sbatch", "E006.sbatch"]


def test_local_runner_writes_standard_result(tmp_path: Path) -> None:
    config = load_experiment_config(Path("configs/local/synthetic_smoke.json"))

    result = LocalBenchmarkRunner().run(config)
    result_path = result.write_json(tmp_path)

    assert result.status == "completed"
    assert result.throughput_tokens_sec is not None
    assert result_path.exists()


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
