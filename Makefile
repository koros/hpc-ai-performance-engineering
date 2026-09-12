PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
SLURM_PROFILE ?=
SLURM_PROFILE_ARG := $(if $(SLURM_PROFILE),--slurm-profile $(SLURM_PROFILE),)
PLATFORM ?= l40s
PLATFORM_WAVE ?= 0
PLATFORM_PROFILE ?= platforms/$(PLATFORM).json
PLATFORM_CONFIG_DIR ?= build/platform_configs/$(PLATFORM)
PLATFORM_SLURM_DIR ?= slurm/generated/$(PLATFORM)/wave_$(PLATFORM_WAVE)
PLATFORM_RUN_OUTPUT_DIR ?= build/platform_results/$(PLATFORM)/wave_$(PLATFORM_WAVE)
PLATFORM_STATUS ?= results/processed/slurm_$(PLATFORM)_wave_$(PLATFORM_WAVE)_status.csv
SUBMISSION_BUNDLE_ROOT ?= ../hpc-ai-submissions
SUBMISSION_BUNDLE_ID ?=
SUBMISSION_BUNDLE_ID_ARG := $(if $(SUBMISSION_BUNDLE_ID),--bundle-id $(SUBMISSION_BUNDLE_ID),)
VENV_ACTIVATE ?=
VENV_ACTIVATE_ARG := $(if $(VENV_ACTIVATE),--venv-activate $(VENV_ACTIVATE),)
SUBMISSION_BUNDLE_DIR ?=
PLATFORM_WAVE_STATE ?= build/platform_waves/$(PLATFORM)/wave_$(PLATFORM_WAVE).json
RUNTIME_EXPECTED_GPU_MODEL ?= $(if $(filter h100,$(PLATFORM)),H100,L40S)
EXPECTED_TORCH_VERSION ?= 2.8.0
EXPECTED_TORCH_CUDA ?= 12.4
EXPECTED_GPUS ?=
EXPECTED_GPUS_ARG := $(if $(EXPECTED_GPUS),--expected-gpus $(EXPECTED_GPUS),)
REQUIRE_DEEPSPEED ?=
REQUIRE_DEEPSPEED_ARG := $(if $(REQUIRE_DEEPSPEED),--require-deepspeed,)
REQUIRE_VLLM ?=
REQUIRE_VLLM_ARG := $(if $(REQUIRE_VLLM),--require-vllm,)
VLLM_VERSION ?= 0.10.2
VLLM_BASE ?= $(HOME)/.local/share/hpc-ai-performance-engineering
TORCH_VERSION ?= 2.8.0
TORCH_GLOO_BUILD_ENV ?= $(VLLM_BASE)/pytorch-$(TORCH_VERSION)-cu124-gloo-build
TORCH_GLOO_SOURCE ?= $(VLLM_BASE)/src/pytorch-v$(TORCH_VERSION)
TORCH_GLOO_WHEEL_DIR ?= $(VLLM_BASE)/wheels/pytorch-$(TORCH_VERSION)-cu124-gloo
VLLM_ENV ?= $(VLLM_BASE)/vllm-$(VLLM_VERSION)-gloo
VLLM_SOURCE ?= $(VLLM_BASE)/src/vllm-v$(VLLM_VERSION)-gloo
TRANSFORMER_ENGINE_VERSION ?= 2.8.0
TRANSFORMER_ENGINE_CUDNN_VERSION ?= 9.3.0.75
TRANSFORMER_ENGINE_CUBLAS_VERSION ?= 12.4.5.8
TRANSFORMER_ENGINE_NVRTC_VERSION ?= 12.4.127
TRANSFORMER_ENGINE_TRITON_VERSION ?= 3.4.0
FP8_ENV ?= $(VLLM_BASE)/fp8-training-te$(TRANSFORMER_ENGINE_VERSION)
FP8_ACCEPTANCE_ROOT ?= build/fp8_acceptance/$(PLATFORM)
NCCL_HOME ?=

DEV_CONFIGS := $(sort $(wildcard configs/dev/*.json))
DEV_BASELINE_CONFIGS := $(sort $(wildcard configs/dev/*baseline.json))

# Dev targets run local Mac/workstation checks; matrix tracking stays separate.
.PHONY: check-cuda-toolchain
.PHONY: setup-fp8-training-environment submit-fp8-training-build submit-fp8-training-acceptance verify-fp8-training-acceptance submit-fp8-training-wave retry-fp8-training-wave
.PHONY: install-dev lint format test test-dashboard check hooks list-configs run-dev-baselines run-dev-experiments run-dev-refresh check-slurm-profile check-platform-runtime check-l40s-runtime check-h100-runtime check-cuda-toolchain setup-pytorch-gloo-environment submit-pytorch-gloo-build setup-vllm-environment submit-vllm-build submit-vllm-acceptance submit-vllm-pilot submit-vllm-wave retry-vllm-wave materialize-platform-wave render-platform-wave render-l40s-wave render-h100-wave plan-platform-wave init-platform-wave-status dry-run-submit-rendered-platform-wave submit-rendered-platform-wave submit-platform-wave status-platform-wave collect-platform-wave retry-platform-wave stage-platform-wave create-platform-wave-bundle plan-platform-wave-bundle init-platform-wave-bundle-status dry-run-submit-platform-wave-bundle submit-platform-wave-bundle import-platform-wave-bundle track-platform-progress track-l40s-progress track-h100-progress import-result-evidence render-slurm-example render-slurm-must-have render-slurm-env-validation render-slurm-tensorrt-llm render-slurm-baseline-must-have render-slurm-first-batches plan-slurm-env-validation plan-slurm-baseline-must-have plan-slurm-first-batches init-slurm-env-status init-slurm-baseline-status init-slurm-first-status dry-run-submit-slurm-env dry-run-submit-slurm-baseline dry-run-submit-slurm-first report-slurm-env-status report-slurm-baseline-status report-slurm-first-status list-slurm-smoke aggregate-results analyze-results audit-telemetry track-progress track-dev-progress validate validate-result-evidence readiness-report refresh build-dashboard-site serve-dashboard

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

check-cuda-toolchain:
	$(PYTHON) scripts/check_cuda_toolchain.py --expected-gpu-model $(RUNTIME_EXPECTED_GPU_MODEL) $(EXPECTED_GPUS_ARG) --require-deepspeed

# The site Torch lacks Gloo, which vLLM requires for its CPU coordination
# group. Build a reusable private Torch wheel first; neither build mutates the
# project .venv or relies on --system-site-packages.
setup-pytorch-gloo-environment:
	HPC_AI_TORCH_VERSION="$(TORCH_VERSION)" bash slurm/setup_pytorch_gloo_environment.sh --project-root "$(CURDIR)" --env-dir "$(TORCH_GLOO_BUILD_ENV)" --source-dir "$(TORCH_GLOO_SOURCE)" --wheel-dir "$(TORCH_GLOO_WHEEL_DIR)"

submit-pytorch-gloo-build:
	sbatch --export=ALL,HPC_AI_TORCH_VERSION="$(TORCH_VERSION)",HPC_AI_TORCH_ENV="$(TORCH_GLOO_BUILD_ENV)",HPC_AI_TORCH_SOURCE="$(TORCH_GLOO_SOURCE)",HPC_AI_TORCH_WHEEL_DIR="$(TORCH_GLOO_WHEEL_DIR)" slurm/smoke/pytorch_gloo_build_l40s.sbatch

setup-vllm-environment:
	HPC_AI_VLLM_VERSION="$(VLLM_VERSION)" bash slurm/setup_vllm_environment.sh --project-root "$(CURDIR)" --env-dir "$(VLLM_ENV)" --source-dir "$(VLLM_SOURCE)" --torch-wheel-dir "$(TORCH_GLOO_WHEEL_DIR)"

submit-vllm-build:
	test -r "$(TORCH_GLOO_WHEEL_DIR)/BUILD_MANIFEST.json" || (echo "Run make submit-pytorch-gloo-build and wait for it to succeed first" && exit 1)
	sbatch --export=ALL,HPC_AI_VLLM_VERSION="$(VLLM_VERSION)",HPC_AI_VLLM_ENV="$(VLLM_ENV)",HPC_AI_VLLM_SOURCE="$(VLLM_SOURCE)",HPC_AI_TORCH_WHEEL_DIR="$(TORCH_GLOO_WHEEL_DIR)" slurm/smoke/vllm_build_l40s.sbatch

submit-vllm-acceptance:
	test -r "$(VLLM_ENV)/bin/activate" || (echo "Run make submit-vllm-build and wait for it to succeed first" && exit 1)
	sbatch --partition=$(if $(filter h100,$(PLATFORM)),gpu_h100,gpu_l40s) --export=ALL,HPC_AI_EXPECTED_GPU_MODEL="$(RUNTIME_EXPECTED_GPU_MODEL)",HPC_AI_VLLM_VERSION="$(VLLM_VERSION)",HPC_AI_VLLM_ENV="$(VLLM_ENV)" slurm/smoke/vllm_acceptance.sbatch

submit-vllm-pilot:
	$(MAKE) submit-vllm-wave PLATFORM=$(PLATFORM) PLATFORM_WAVE=vllm-pilot

submit-vllm-wave:
	test -r "$(VLLM_ENV)/bin/activate" || (echo "Run make submit-vllm-build and make submit-vllm-acceptance first" && exit 1)
	@case "$(PLATFORM_WAVE)" in vllm-*) ;; *) echo "Set PLATFORM_WAVE to a dedicated vllm-* wave"; exit 1;; esac
	$(MAKE) submit-platform-wave PLATFORM=$(PLATFORM) PLATFORM_WAVE=$(PLATFORM_WAVE) VENV_ACTIVATE="$(VLLM_ENV)/bin/activate"

retry-vllm-wave:
	test -r "$(VLLM_ENV)/bin/activate" || (echo "Run make submit-vllm-build and make submit-vllm-acceptance first" && exit 1)
	@case "$(PLATFORM_WAVE)" in vllm-*) ;; *) echo "Set PLATFORM_WAVE to a dedicated vllm-* wave"; exit 1;; esac
	$(MAKE) retry-platform-wave PLATFORM=$(PLATFORM) PLATFORM_WAVE=$(PLATFORM_WAVE) VENV_ACTIVATE="$(VLLM_ENV)/bin/activate"

# Transformer Engine is compiled against the same private Torch ABI as vLLM,
# but lives in a separate environment so neither experimental stack mutates the
# other. Canonical FP8 evidence is gated by two real-model acceptance runs.
setup-fp8-training-environment:
	test -n "$(NCCL_HOME)" || (echo "Set NCCL_HOME to a matching NCCL development root" && exit 1)
	test -r "$(NCCL_HOME)/include/nccl.h" || (echo "NCCL_HOME must contain include/nccl.h" && exit 1)
	HPC_AI_TE_VERSION="$(TRANSFORMER_ENGINE_VERSION)" HPC_AI_TE_CUDNN_VERSION="$(TRANSFORMER_ENGINE_CUDNN_VERSION)" HPC_AI_TE_CUBLAS_VERSION="$(TRANSFORMER_ENGINE_CUBLAS_VERSION)" HPC_AI_TE_NVRTC_VERSION="$(TRANSFORMER_ENGINE_NVRTC_VERSION)" HPC_AI_TE_TRITON_VERSION="$(TRANSFORMER_ENGINE_TRITON_VERSION)" HPC_AI_NCCL_HOME="$(NCCL_HOME)" bash slurm/setup_fp8_training_environment.sh --project-root "$(CURDIR)" --env-dir "$(FP8_ENV)" --torch-wheel-dir "$(TORCH_GLOO_WHEEL_DIR)" --nccl-home "$(NCCL_HOME)"

submit-fp8-training-build:
	test -r "$(TORCH_GLOO_WHEEL_DIR)/BUILD_MANIFEST.json" || (echo "Run make submit-pytorch-gloo-build and wait for it to succeed first" && exit 1)
	test -n "$(NCCL_HOME)" || (echo "Set NCCL_HOME to a matching NCCL development root" && exit 1)
	test -r "$(NCCL_HOME)/include/nccl.h" || (echo "NCCL_HOME must contain include/nccl.h" && exit 1)
	sbatch --export=ALL,HPC_AI_TE_VERSION="$(TRANSFORMER_ENGINE_VERSION)",HPC_AI_TE_CUDNN_VERSION="$(TRANSFORMER_ENGINE_CUDNN_VERSION)",HPC_AI_TE_CUBLAS_VERSION="$(TRANSFORMER_ENGINE_CUBLAS_VERSION)",HPC_AI_TE_NVRTC_VERSION="$(TRANSFORMER_ENGINE_NVRTC_VERSION)",HPC_AI_TE_TRITON_VERSION="$(TRANSFORMER_ENGINE_TRITON_VERSION)",HPC_AI_FP8_ENV="$(FP8_ENV)",HPC_AI_TORCH_WHEEL_DIR="$(TORCH_GLOO_WHEEL_DIR)",HPC_AI_NCCL_HOME="$(NCCL_HOME)" slurm/smoke/fp8_training_build_l40s.sbatch

submit-fp8-training-acceptance:
	@case "$(PLATFORM)" in l40s|h100) ;; *) echo "FP8 training acceptance supports PLATFORM=l40s or PLATFORM=h100"; exit 1;; esac
	test -r "$(FP8_ENV)/BUILD_MANIFEST.json" || (echo "Run make submit-fp8-training-build and wait for it to succeed first" && exit 1)
	sbatch --partition=$(if $(filter h100,$(PLATFORM)),gpu_h100,gpu_l40s) --export=ALL,HPC_AI_EXPECTED_GPU_MODEL="$(RUNTIME_EXPECTED_GPU_MODEL)",HPC_AI_FP8_ENV="$(FP8_ENV)" slurm/smoke/fp8_training_acceptance.sbatch

verify-fp8-training-acceptance:
	$(PYTHON) scripts/verify_fp8_training_acceptance.py --result-root "$(FP8_ACCEPTANCE_ROOT)" --expected-gpu-model "$(RUNTIME_EXPECTED_GPU_MODEL)"

submit-fp8-training-wave: verify-fp8-training-acceptance
	@case "$(PLATFORM)" in l40s|h100) ;; *) echo "FP8 training waves support PLATFORM=l40s or PLATFORM=h100"; exit 1;; esac
	test -r "$(FP8_ENV)/bin/activate_hpc_ai_fp8" || (echo "Run make submit-fp8-training-build and wait for it to succeed first" && exit 1)
	$(MAKE) submit-platform-wave PLATFORM=$(PLATFORM) PLATFORM_WAVE=training-fp8 VENV_ACTIVATE="$(FP8_ENV)/bin/activate_hpc_ai_fp8"

retry-fp8-training-wave: verify-fp8-training-acceptance
	@case "$(PLATFORM)" in l40s|h100) ;; *) echo "FP8 training waves support PLATFORM=l40s or PLATFORM=h100"; exit 1;; esac
	test -r "$(FP8_ENV)/bin/activate_hpc_ai_fp8" || (echo "Run make submit-fp8-training-build and wait for it to succeed first" && exit 1)
	$(MAKE) retry-platform-wave PLATFORM=$(PLATFORM) PLATFORM_WAVE=training-fp8 VENV_ACTIVATE="$(FP8_ENV)/bin/activate_hpc_ai_fp8"

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

dry-run-submit-rendered-platform-wave:
	$(PYTHON) -m hpc_ai_perf.cli submit-slurm-batch --script-dir $(PLATFORM_SLURM_DIR) --status $(PLATFORM_STATUS) --dry-run

submit-rendered-platform-wave:
	$(PYTHON) -m hpc_ai_perf.cli submit-slurm-batch --script-dir $(PLATFORM_SLURM_DIR) --status $(PLATFORM_STATUS)

# One-command platform lifecycle. It records the generated bundle path in an
# ignored build state file, so monitoring, collection, and staging need no
# copied path or shell environment variable.
submit-platform-wave:
	$(PYTHON) -m hpc_ai_perf.cli submit-platform-wave --platform-profile $(PLATFORM_PROFILE) --wave $(PLATFORM_WAVE) --bundle-root $(SUBMISSION_BUNDLE_ROOT) $(SUBMISSION_BUNDLE_ID_ARG) --state-file $(PLATFORM_WAVE_STATE) $(SLURM_PROFILE_ARG) $(VENV_ACTIVATE_ARG)

status-platform-wave:
	$(PYTHON) -m hpc_ai_perf.cli status-platform-wave --platform-profile $(PLATFORM_PROFILE) --wave $(PLATFORM_WAVE) --state-file $(PLATFORM_WAVE_STATE)

collect-platform-wave:
	$(PYTHON) -m hpc_ai_perf.cli collect-platform-wave --platform-profile $(PLATFORM_PROFILE) --wave $(PLATFORM_WAVE) --state-file $(PLATFORM_WAVE_STATE) --result-root results/raw

retry-platform-wave:
	$(PYTHON) -m hpc_ai_perf.cli retry-platform-wave --platform-profile $(PLATFORM_PROFILE) --wave $(PLATFORM_WAVE) --bundle-root $(SUBMISSION_BUNDLE_ROOT) $(SUBMISSION_BUNDLE_ID_ARG) --state-file $(PLATFORM_WAVE_STATE) $(SLURM_PROFILE_ARG) $(VENV_ACTIVATE_ARG)

stage-platform-wave:
	$(PYTHON) -m hpc_ai_perf.cli stage-platform-wave --platform-profile $(PLATFORM_PROFILE) --wave $(PLATFORM_WAVE) --state-file $(PLATFORM_WAVE_STATE)

# Legacy bundle primitives remain available for inspection and one-off
# recovery, but the targets above are the standard L40S/H100 workflow.
create-platform-wave-bundle:
	test -n "$(SUBMISSION_BUNDLE_ID)" || (echo "Set SUBMISSION_BUNDLE_ID to a stable wave identifier" && exit 1)
	$(PYTHON) -m hpc_ai_perf.cli create-platform-submission-bundle --platform-profile $(PLATFORM_PROFILE) --wave $(PLATFORM_WAVE) --bundle-root $(SUBMISSION_BUNDLE_ROOT) --bundle-id $(SUBMISSION_BUNDLE_ID) $(SLURM_PROFILE_ARG) $(VENV_ACTIVATE_ARG)

plan-platform-wave-bundle:
	test -n "$(SUBMISSION_BUNDLE_DIR)" || (echo "Set SUBMISSION_BUNDLE_DIR to the created bundle path" && exit 1)
	$(PYTHON) -m hpc_ai_perf.cli plan-slurm-submission --script-dir $(SUBMISSION_BUNDLE_DIR)/scripts --output $(SUBMISSION_BUNDLE_DIR)/SUBMISSION_PLAN.md --batch-name "$(PLATFORM) wave $(PLATFORM_WAVE) bundle"

init-platform-wave-bundle-status:
	test -n "$(SUBMISSION_BUNDLE_DIR)" || (echo "Set SUBMISSION_BUNDLE_DIR to the created bundle path" && exit 1)
	$(PYTHON) -m hpc_ai_perf.cli init-slurm-status --script-dir $(SUBMISSION_BUNDLE_DIR)/scripts --output $(SUBMISSION_BUNDLE_DIR)/SLURM_STATUS.csv --batch-name "$(PLATFORM) wave $(PLATFORM_WAVE) bundle"

dry-run-submit-platform-wave-bundle:
	test -n "$(SUBMISSION_BUNDLE_DIR)" || (echo "Set SUBMISSION_BUNDLE_DIR to the created bundle path" && exit 1)
	$(PYTHON) -m hpc_ai_perf.cli submit-slurm-batch --script-dir $(SUBMISSION_BUNDLE_DIR)/scripts --status $(SUBMISSION_BUNDLE_DIR)/SLURM_STATUS.csv --dry-run

submit-platform-wave-bundle:
	test -n "$(SUBMISSION_BUNDLE_DIR)" || (echo "Set SUBMISSION_BUNDLE_DIR to the created bundle path" && exit 1)
	$(PYTHON) -m hpc_ai_perf.cli submit-slurm-batch --script-dir $(SUBMISSION_BUNDLE_DIR)/scripts --status $(SUBMISSION_BUNDLE_DIR)/SLURM_STATUS.csv

import-platform-wave-bundle:
	test -n "$(SUBMISSION_BUNDLE_DIR)" || (echo "Set SUBMISSION_BUNDLE_DIR to the created bundle path" && exit 1)
	$(PYTHON) -m hpc_ai_perf.cli import-result-evidence --source $(SUBMISSION_BUNDLE_DIR)/results --result-root results/raw

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

audit-telemetry:
	$(PYTHON) -m hpc_ai_perf.cli audit-telemetry

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

build-dashboard-site: refresh test-dashboard
	$(PYTHON) scripts/build_dashboard_site.py --output-dir build/pages

serve-dashboard:
	$(PYTHON) -m http.server 8000
