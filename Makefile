SLURM_PROFILE ?=
SLURM_PROFILE_ARG := $(if $(SLURM_PROFILE),--slurm-profile $(SLURM_PROFILE),)

DEV_CONFIGS := $(sort $(wildcard configs/dev/*.json))

.PHONY: install-dev lint format test check hooks list-configs run-dev-baselines run-dev-refresh check-slurm-profile render-slurm-example render-slurm-must-have render-slurm-env-validation render-slurm-baseline-must-have render-slurm-first-batches plan-slurm-env-validation plan-slurm-baseline-must-have plan-slurm-first-batches init-slurm-env-status init-slurm-baseline-status init-slurm-first-status dry-run-submit-slurm-env dry-run-submit-slurm-baseline dry-run-submit-slurm-first report-slurm-env-status report-slurm-baseline-status report-slurm-first-status list-slurm-smoke aggregate-results analyze-results track-progress validate readiness-report refresh serve-dashboard

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

run-dev-baselines:
	@for config in $(DEV_CONFIGS); do \
		echo "Running $$config"; \
		python -m hpc_ai_perf.cli run --config "$$config"; \
	done

run-dev-refresh: run-dev-baselines refresh

check-slurm-profile:
	test -n "$(SLURM_PROFILE)" || (echo "Set SLURM_PROFILE=slurm/profiles/cluster.local.json" && exit 1)
	python -m hpc_ai_perf.cli check-slurm-profile --slurm-profile $(SLURM_PROFILE)

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

init-slurm-env-status:
	python -m hpc_ai_perf.cli init-slurm-status --script-dir slurm/generated/environment_validation_must_have --output results/processed/slurm_environment_validation_status.csv --batch-name "Environment Validation Must Have"

init-slurm-baseline-status:
	python -m hpc_ai_perf.cli init-slurm-status --script-dir slurm/generated/baseline_profiling_must_have --output results/processed/slurm_baseline_profiling_status.csv --batch-name "Baseline Profiling Must Have"

init-slurm-first-status: init-slurm-env-status init-slurm-baseline-status

dry-run-submit-slurm-env:
	python -m hpc_ai_perf.cli submit-slurm-batch --script-dir slurm/generated/environment_validation_must_have --status results/processed/slurm_environment_validation_status.csv --dry-run

dry-run-submit-slurm-baseline:
	python -m hpc_ai_perf.cli submit-slurm-batch --script-dir slurm/generated/baseline_profiling_must_have --status results/processed/slurm_baseline_profiling_status.csv --dry-run

dry-run-submit-slurm-first: dry-run-submit-slurm-env dry-run-submit-slurm-baseline

report-slurm-env-status:
	python -m hpc_ai_perf.cli report-slurm-status --status results/processed/slurm_environment_validation_status.csv --output results/processed/slurm_environment_validation_status.md

report-slurm-baseline-status:
	python -m hpc_ai_perf.cli report-slurm-status --status results/processed/slurm_baseline_profiling_status.csv --output results/processed/slurm_baseline_profiling_status.md

report-slurm-first-status: report-slurm-env-status report-slurm-baseline-status

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
