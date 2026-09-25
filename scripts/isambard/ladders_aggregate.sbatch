#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --job-name=agt-ladders-aggregate
#SBATCH --gpus=1
#SBATCH --time=01:00:00
# Once every ladders.sh and magic_ladder.sh job has finished: tables and results files for the four
# refinement ladders.
# Each script reads cached runs only and exits listing any that are missing.
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/isambard/env.sh"
cd "$AGT_REPO/scripts/poisoning_paper"
failed=0
for cmd in "cifar_pca_leaf_sweep.py --cell e4" "octmnist_pca_ladder_sweep.py" \
           "halfmoons_refinement_sweep.py --rungs 2 3 4 5 6 8 10 12 16" "magic_refinement_sweep.py"; do
    echo "== $cmd"
    python $cmd || { echo "FAILED: $cmd"; failed=1; }
done
exit "$failed"
