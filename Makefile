PYTHON ?= python3
SLURM_PROFILE ?=
SLURM_PROFILE_ARG := $(if $(SLURM_PROFILE),--slurm-profile $(SLURM_PROFILE),)
PLATFORM ?= l40s
PLATFORM_WAVE ?= 0
PLATFORM_PROFILE ?= platforms/$(PLATFORM).json
PLATFORM_CONFIG_DIR ?= build/platform_configs/$(PLATFORM)
PLATFORM_SLURM_DIR ?= slurm/generated/$(PLATFORM)/wave_$(PLATFORM_WAVE)
PLATFORM_RUN_OUTPUT_DIR ?= build/platform_results/$(PLATFORM)/wave_$(PLATFORM_WAVE)
PLATFORM_STATUS ?= results/processed/slurm_$(PLATFORM)_wave_$(PLATFORM_WAVE)_status.csv
RUNTIME_EXPECTED_GPU_MODEL ?= $(if $(filter h100,$(PLATFORM)),H100,L40S)
EXPECTED_TORCH_VERSION ?= 2.8.0
EXPECTED_TORCH_CUDA ?= 12.4
EXPECTED_GPUS ?=
EXPECTED_GPUS_ARG := $(if $(EXPECTED_GPUS),--expected-gpus $(EXPECTED_GPUS),)
REQUIRE_DEEPSPEED ?=
REQUIRE_DEEPSPEED_ARG := $(if $(REQUIRE_DEEPSPEED),--require-deepspeed,)
REQUIRE_VLLM ?=
REQUIRE_VLLM_ARG := $(if $(REQUIRE_VLLM),--require-vllm,)

DEV_CONFIGS := $(sort $(wildcard configs/dev/*.json))
DEV_BASELINE_CONFIGS := $(sort $(wildcard configs/dev/*baseline.json))

# Dev targets run local Mac/workstation checks; matrix tracking stays separate.
.PHONY: install-dev lint format test test-dashboard check hooks list-configs run-dev-baselines run-dev-experiments run-dev-refresh check-slurm-profile check-platform-runtime check-l40s-runtime check-h100-runtime materialize-platform-wave render-platform-wave render-l40s-wave render-h100-wave plan-platform-wave init-platform-wave-status dry-run-submit-platform-wave submit-platform-wave track-platform-progress track-l40s-progress track-h100-progress import-result-evidence render-slurm-example render-slurm-must-have render-slurm-env-validation render-slurm-tensorrt-llm render-slurm-baseline-must-have render-slurm-first-batches plan-slurm-env-validation plan-slurm-baseline-must-have plan-slurm-first-batches init-slurm-env-status init-slurm-baseline-status init-slurm-first-status dry-run-submit-slurm-env dry-run-submit-slurm-baseline dry-run-submit-slurm-first report-slurm-env-status report-slurm-baseline-status report-slurm-first-status list-slurm-smoke aggregate-results analyze-results track-progress track-dev-progress validate validate-result-evidence readiness-report refresh serve-dashboard

install-dev:
	$(PYTHON) -m pip install -e ".[dev,yaml]"

lint:
	ruff check .

format:
	ruff format .
	ruff check . --fix

test:
	$(PYTHON) -m pytest

test-dashboard:
	node dashboard/app.test.cjs

check: lint test test-dashboard

hooks:
	pre-commit install
	pre-commit run --all-files

list-configs:
	$(PYTHON) -m hpc_ai_perf.cli list-configs

run-dev-baselines:
	@for config in $(DEV_BASELINE_CONFIGS); do \
		echo "Running $$config"; \
		$(PYTHON) -m hpc_ai_perf.cli run --config "$$config"; \
	done

run-dev-experiments:
	@for config in $(DEV_CONFIGS); do \
		echo "Running $$config"; \
		$(PYTHON) -m hpc_ai_perf.cli run --config "$$config"; \
	done

run-dev-refresh: run-dev-experiments refresh track-dev-progress

check-slurm-profile:
	test -n "$(SLURM_PROFILE)" || (echo "Set SLURM_PROFILE=slurm/profiles/cluster.local.json" && exit 1)
	$(PYTHON) -m hpc_ai_perf.cli check-slurm-profile --slurm-profile $(SLURM_PROFILE)

check-platform-runtime:
	$(PYTHON) -m hpc_ai_perf.cli check-runtime --expected-gpu-model $(RUNTIME_EXPECTED_GPU_MODEL) --expected-torch-version $(EXPECTED_TORCH_VERSION) --expected-torch-cuda $(EXPECTED_TORCH_CUDA) $(EXPECTED_GPUS_ARG) $(REQUIRE_DEEPSPEED_ARG) $(REQUIRE_VLLM_ARG)

check-l40s-runtime:
	$(MAKE) check-platform-runtime PLATFORM=l40s

check-h100-runtime:
	$(MAKE) check-platform-runtime PLATFORM=h100

# L40S is the active platform; the same targets accept PLATFORM=h100 when that node returns.
materialize-platform-wave:
	$(PYTHON) -m hpc_ai_perf.cli materialize-platform-configs --platform-profile $(PLATFORM_PROFILE) --wave $(PLATFORM_WAVE) --output-dir $(PLATFORM_CONFIG_DIR)

render-platform-wave:
	$(PYTHON) -m hpc_ai_perf.cli render-platform-slurm --platform-profile $(PLATFORM_PROFILE) --wave $(PLATFORM_WAVE) --config-output-dir $(PLATFORM_CONFIG_DIR) --output-dir $(PLATFORM_SLURM_DIR) --run-output-dir $(PLATFORM_RUN_OUTPUT_DIR) $(SLURM_PROFILE_ARG)

render-l40s-wave:
	$(MAKE) render-platform-wave PLATFORM=l40s

render-h100-wave:
	$(MAKE) render-platform-wave PLATFORM=h100

plan-platform-wave: render-platform-wave
	$(PYTHON) -m hpc_ai_perf.cli plan-slurm-submission --script-dir $(PLATFORM_SLURM_DIR) --output $(PLATFORM_SLURM_DIR)/SUBMISSION_PLAN.md --batch-name "$(PLATFORM) wave $(PLATFORM_WAVE)"

init-platform-wave-status:
	$(PYTHON) -m hpc_ai_perf.cli init-slurm-status --script-dir $(PLATFORM_SLURM_DIR) --output $(PLATFORM_STATUS) --batch-name "$(PLATFORM) wave $(PLATFORM_WAVE)"

dry-run-submit-platform-wave:
	$(PYTHON) -m hpc_ai_perf.cli submit-slurm-batch --script-dir $(PLATFORM_SLURM_DIR) --status $(PLATFORM_STATUS) --dry-run

submit-platform-wave:
	$(PYTHON) -m hpc_ai_perf.cli submit-slurm-batch --script-dir $(PLATFORM_SLURM_DIR) --status $(PLATFORM_STATUS)

track-platform-progress:
	$(PYTHON) -m hpc_ai_perf.cli track-platform-progress --platform-profile $(PLATFORM_PROFILE)

track-l40s-progress:
	$(MAKE) track-platform-progress PLATFORM=l40s

track-h100-progress:
	$(MAKE) track-platform-progress PLATFORM=h100

render-slurm-example:
	$(PYTHON) -m hpc_ai_perf.cli render-slurm --config configs/environment_validation/E001_cuda_pytorch_1gpu.json $(SLURM_PROFILE_ARG)

render-slurm-must-have:
	$(PYTHON) -m hpc_ai_perf.cli render-slurm-bulk --priority "Must Have" $(SLURM_PROFILE_ARG)

render-slurm-env-validation:
	$(MAKE) render-platform-wave PLATFORM=$(PLATFORM) PLATFORM_WAVE=0 PLATFORM_SLURM_DIR=slurm/generated/environment_validation_must_have

render-slurm-tensorrt-llm:
	$(PYTHON) -m hpc_ai_perf.cli render-slurm --config configs/environment_validation/E007_tensorrt_llm_1gpu.json --output-dir slurm/generated/tensorrt_llm $(SLURM_PROFILE_ARG)

render-slurm-baseline-must-have:
	$(PYTHON) -m hpc_ai_perf.cli render-slurm-bulk --phase "Baseline profiling" --priority "Must Have" --repetitions 3 --output-dir slurm/generated/baseline_profiling_must_have $(SLURM_PROFILE_ARG)

render-slurm-first-batches: render-slurm-env-validation render-slurm-baseline-must-have

plan-slurm-env-validation: render-slurm-env-validation
	$(PYTHON) -m hpc_ai_perf.cli plan-slurm-submission --script-dir slurm/generated/environment_validation_must_have --output slurm/generated/environment_validation_must_have/SUBMISSION_PLAN.md --batch-name "Environment Validation Must Have"

plan-slurm-baseline-must-have: render-slurm-baseline-must-have
	$(PYTHON) -m hpc_ai_perf.cli plan-slurm-submission --script-dir slurm/generated/baseline_profiling_must_have --output slurm/generated/baseline_profiling_must_have/SUBMISSION_PLAN.md --batch-name "Baseline Profiling Must Have"

plan-slurm-first-batches: plan-slurm-env-validation plan-slurm-baseline-must-have

init-slurm-env-status:
	$(PYTHON) -m hpc_ai_perf.cli init-slurm-status --script-dir slurm/generated/environment_validation_must_have --output results/processed/slurm_environment_validation_status.csv --batch-name "Environment Validation Must Have"

init-slurm-baseline-status:
	$(PYTHON) -m hpc_ai_perf.cli init-slurm-status --script-dir slurm/generated/baseline_profiling_must_have --output results/processed/slurm_baseline_profiling_status.csv --batch-name "Baseline Profiling Must Have"

init-slurm-first-status: init-slurm-env-status init-slurm-baseline-status

dry-run-submit-slurm-env:
	$(PYTHON) -m hpc_ai_perf.cli submit-slurm-batch --script-dir slurm/generated/environment_validation_must_have --status results/processed/slurm_environment_validation_status.csv --dry-run

dry-run-submit-slurm-baseline:
	$(PYTHON) -m hpc_ai_perf.cli submit-slurm-batch --script-dir slurm/generated/baseline_profiling_must_have --status results/processed/slurm_baseline_profiling_status.csv --dry-run

dry-run-submit-slurm-first: dry-run-submit-slurm-env dry-run-submit-slurm-baseline

report-slurm-env-status:
	$(PYTHON) -m hpc_ai_perf.cli report-slurm-status --status results/processed/slurm_environment_validation_status.csv --output results/processed/slurm_environment_validation_status.md

report-slurm-baseline-status:
	$(PYTHON) -m hpc_ai_perf.cli report-slurm-status --status results/processed/slurm_baseline_profiling_status.csv --output results/processed/slurm_baseline_profiling_status.md

report-slurm-first-status: report-slurm-env-status report-slurm-baseline-status

list-slurm-smoke:
	ls slurm/smoke/*.sbatch

aggregate-results:
	$(PYTHON) -m hpc_ai_perf.cli aggregate-results

analyze-results:
	$(PYTHON) -m hpc_ai_perf.cli analyze-results

track-progress:
	$(PYTHON) -m hpc_ai_perf.cli track-progress

track-dev-progress:
	$(PYTHON) -m hpc_ai_perf.cli track-progress --profile dev --output-csv results/processed/dev_experiment_tracker.csv --output-markdown results/processed/dev_experiment_tracker.md

validate:
	$(PYTHON) -m hpc_ai_perf.cli validate

validate-result-evidence:
	$(PYTHON) -m hpc_ai_perf.cli verify-result-evidence --result-root results/raw

import-result-evidence:
	test -n "$(SOURCE)" || (echo "Set SOURCE to a copied HPC result file or directory" && exit 1)
	$(PYTHON) -m hpc_ai_perf.cli import-result-evidence --source $(SOURCE) --result-root results/raw

readiness-report:
	$(PYTHON) -m hpc_ai_perf.cli readiness-report

refresh:
	$(PYTHON) -m hpc_ai_perf.cli refresh

serve-dashboard:
	$(PYTHON) -m http.server 8000
