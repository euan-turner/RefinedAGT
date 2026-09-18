#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --job-name=agt-octmnist-threat
#SBATCH --gpus=4
#SBATCH --time=03:00:00
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/isambard/env.sh"
cd "$AGT_REPO/scripts/poisoning_paper"
torchrun --standalone --nproc-per-node=4 octmnist_pca_threat_sweep.py
