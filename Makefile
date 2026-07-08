SLURM_PROFILE ?=
SLURM_PROFILE_ARG := $(if $(SLURM_PROFILE),--slurm-profile $(SLURM_PROFILE),)

.PHONY: install-dev lint format test check hooks list-configs render-slurm-example render-slurm-must-have render-slurm-env-validation render-slurm-baseline-must-have render-slurm-first-batches plan-slurm-env-validation plan-slurm-baseline-must-have plan-slurm-first-batches list-slurm-smoke aggregate-results analyze-results track-progress validate readiness-report refresh serve-dashboard

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
	python -m hpc_ai_perf.cli render-slurm --config configs/environment_validation/E001_cuda_pytorch_1gpu.json $(SLURM_PROFILE_ARG)

render-slurm-must-have:
	python -m hpc_ai_perf.cli render-slurm-bulk --priority "Must Have" $(SLURM_PROFILE_ARG)

render-slurm-env-validation:
	python -m hpc_ai_perf.cli render-slurm-bulk --phase "Environment validation" --priority "Must Have" --output-dir slurm/generated/environment_validation_must_have $(SLURM_PROFILE_ARG)

render-slurm-baseline-must-have:
	python -m hpc_ai_perf.cli render-slurm-bulk --phase "Baseline profiling" --priority "Must Have" --output-dir slurm/generated/baseline_profiling_must_have $(SLURM_PROFILE_ARG)

render-slurm-first-batches: render-slurm-env-validation render-slurm-baseline-must-have

plan-slurm-env-validation: render-slurm-env-validation
	python -m hpc_ai_perf.cli plan-slurm-submission --script-dir slurm/generated/environment_validation_must_have --output slurm/generated/environment_validation_must_have/SUBMISSION_PLAN.md --batch-name "Environment Validation Must Have"

plan-slurm-baseline-must-have: render-slurm-baseline-must-have
	python -m hpc_ai_perf.cli plan-slurm-submission --script-dir slurm/generated/baseline_profiling_must_have --output slurm/generated/baseline_profiling_must_have/SUBMISSION_PLAN.md --batch-name "Baseline Profiling Must Have"

plan-slurm-first-batches: plan-slurm-env-validation plan-slurm-baseline-must-have

list-slurm-smoke:
	ls slurm/smoke/*.sbatch

aggregate-results:
	python -m hpc_ai_perf.cli aggregate-results

analyze-results:
	python -m hpc_ai_perf.cli analyze-results

track-progress:
	python -m hpc_ai_perf.cli track-progress

validate:
	python -m hpc_ai_perf.cli validate

readiness-report:
	python -m hpc_ai_perf.cli readiness-report

refresh:
	python -m hpc_ai_perf.cli refresh

serve-dashboard:
	python -m http.server 8000
