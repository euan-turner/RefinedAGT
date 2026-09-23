#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --job-name=agt-octmnist-ladder-cheap
#SBATCH --gpus=1
#SBATCH --time=01:00:00
# OCT-MNIST ladder runs under 2^16 leaves, in sequence. Argument: a run list from
# `python octmnist_pca_ladder.py --gpus 1 --missing`.
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/isambard/env.sh"
cd "$AGT_REPO/scripts/poisoning_paper"
MANIFEST=$1
failed=0
while read -r args; do
    echo "run: $args"
    python octmnist_pca_ladder_run.py $args < /dev/null || { echo "FAILED: $args"; failed=1; }   # /dev/null: python must not eat the manifest
done < "$MANIFEST"
exit "$failed"
