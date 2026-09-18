#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --job-name=agt-octmnist-single
#SBATCH --gpus=1
#SBATCH --time=08:00:00
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/isambard/env.sh"
cd "$AGT_REPO/scripts/poisoning_paper"
failed=0
for script in octmnist_pca_refinement_sweep.py octmnist_pca_epsilon_sweep.py octmnist_pca_leaf_sweep.py octmnist_pca_plots.py; do
    echo "== $script"
    python "$script" || { echo "FAILED: $script"; failed=1; }
done
exit "$failed"
