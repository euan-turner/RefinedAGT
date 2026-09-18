#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --job-name=agt-cifar-cheap
#SBATCH --gpus=1
#SBATCH --time=02:00:00
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/isambard/env.sh"
cd "$AGT_REPO/scripts/poisoning_paper"
MANIFEST=$1
failed=0
while read -r args; do
    echo "run: $args"
    python cifar_pca_run.py $args < /dev/null || { echo "FAILED: $args"; failed=1; }   # /dev/null: python must not eat the manifest
done < "$MANIFEST"
exit "$failed"
