#!/bin/bash
#SBATCH --job-name=hpc-ai-benchmark
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=results/raw/%x-%j.out
#SBATCH --error=results/raw/%x-%j.err

set -euo pipefail

# Pass one concrete config path, for example configs/baseline_profiling/E008_gpt2_small_train_1gpu.json.
CONFIG_PATH="${1:?Usage: sbatch run_config.sbatch <config-path>}"

python -m hpc_ai_perf.cli run --config "${CONFIG_PATH}"
