from pathlib import Path

from hpc_ai_perf.config import load_experiment_config
from hpc_ai_perf.runners import LocalBenchmarkRunner


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


def test_local_runner_writes_standard_result(tmp_path: Path) -> None:
    config = load_experiment_config(Path("configs/local/synthetic_smoke.json"))

    result = LocalBenchmarkRunner().run(config)
    result_path = result.write_json(tmp_path)

    assert result.status == "completed"
    assert result.throughput_tokens_sec is not None
    assert result_path.exists()
