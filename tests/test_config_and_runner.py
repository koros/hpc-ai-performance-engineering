from pathlib import Path

from hpc_ai_perf.config import discover_experiment_configs, filter_config_entries, load_experiment_config
from hpc_ai_perf.runners import LocalBenchmarkRunner
from hpc_ai_perf.slurm import render_slurm_script, write_slurm_script


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


def test_local_runner_writes_standard_result(tmp_path: Path) -> None:
    config = load_experiment_config(Path("configs/local/synthetic_smoke.json"))

    result = LocalBenchmarkRunner().run(config)
    result_path = result.write_json(tmp_path)

    assert result.status == "completed"
    assert result.throughput_tokens_sec is not None
    assert result_path.exists()
