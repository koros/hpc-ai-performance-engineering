.PHONY: install-dev lint format test check hooks list-configs render-slurm-example render-slurm-must-have aggregate-results analyze-results track-progress serve-dashboard

install-dev:
	python -m pip install -e ".[dev,yaml]"

lint:
	ruff check .

format:
	ruff format .
	ruff check . --fix

test:
	pytest

check: lint test

hooks:
	pre-commit install
	pre-commit run --all-files

list-configs:
	python -m hpc_ai_perf.cli list-configs

render-slurm-example:
	python -m hpc_ai_perf.cli render-slurm --config configs/environment_validation/E001_cuda_pytorch_1gpu.json

render-slurm-must-have:
	python -m hpc_ai_perf.cli render-slurm-bulk --priority "Must Have"

aggregate-results:
	python -m hpc_ai_perf.cli aggregate-results

analyze-results:
	python -m hpc_ai_perf.cli analyze-results

track-progress:
	python -m hpc_ai_perf.cli track-progress

serve-dashboard:
	python -m http.server 8000
