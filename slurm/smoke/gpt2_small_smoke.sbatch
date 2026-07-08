#!/bin/bash
#SBATCH --job-name=hpc-ai-gpt2-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=results/raw/%x-%j.out
#SBATCH --error=results/raw/%x-%j.err

set -euo pipefail

# Adjust these lines for the target HPC environment.
# module load cuda/12.4
# source .venv/bin/activate

python -m hpc_ai_perf.cli run --config configs/local/gpt2_small_smoke.json
