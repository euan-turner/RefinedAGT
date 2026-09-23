#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=agt-octmnist-ladder-sharded
#SBATCH --gpus=4
#SBATCH --output=%x_%A_%a.out
# One OCT-MNIST ladder run of at least 2^16 leaves per array task, sharded across the 4 GPUs.
# --array and --time are given at submission, per cost class (ladders.sh).
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/isambard/env.sh"
cd "$AGT_REPO/scripts/poisoning_paper"
MANIFEST=$1
args=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
echo "run: $args"
torchrun --standalone --nproc-per-node=4 octmnist_pca_ladder_run.py $args
