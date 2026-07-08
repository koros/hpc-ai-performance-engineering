.PHONY: install-dev lint format test check hooks list-configs render-slurm-example

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
