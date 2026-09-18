#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --gpus=4
# E3 pre-training-radius control, bisect-all run at d=$1; --job-name and --time are given at submission
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/isambard/env.sh"
cd "$AGT_REPO/scripts/poisoning_paper"
torchrun --standalone --nproc-per-node=4 cifar_pca_pt_epsilon_control.py \
    --pca-dims "$1" --pt-epsilon 0.01 --split-dims "$1"
