#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --job-name=agt-aggregate
#SBATCH --gpus=1
#SBATCH --time=02:00:00
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/isambard/env.sh"
cd "$AGT_REPO/scripts/poisoning_paper"
failed=0
for cmd in "cifar_pca_threat_sweep.py" "cifar_pca_leaf_sweep.py" "cifar_pca_dims_sweep.py" "uci_refinement_sweep.py --rungs 2 3 4"; do
    echo "== $cmd"
    python $cmd || { echo "FAILED: $cmd"; failed=1; }
done
exit "$failed"
