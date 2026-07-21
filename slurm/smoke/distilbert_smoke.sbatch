#!/bin/bash
#SBATCH --job-name=hpc-ai-distilbert-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_h100
#SBATCH --output=results/raw/%x-%j.out
#SBATCH --error=results/raw/%x-%j.err

set -euo pipefail

source slurm/modules_for_hpc_ai
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false

python -m hpc_ai_perf.cli run --config configs/local/distilbert_smoke.json
