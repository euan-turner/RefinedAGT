#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --job-name=agt-pt-control-cheap
#SBATCH --gpus=1
#SBATCH --time=00:30:00
# E3 pre-training-radius control, runs below 2^16 leaves: unrefined, top-12 and eps/2 at d=22 and 24
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/isambard/env.sh"
cd "$AGT_REPO/scripts/poisoning_paper"
failed=0
for d in 22 24; do
    for args in "" "--split-dims 12" "--eps 0.025"; do
        echo "run: --pca-dims $d --pt-epsilon 0.01 $args"
        python cifar_pca_pt_epsilon_control.py --pca-dims "$d" --pt-epsilon 0.01 $args \
            || { echo "FAILED: d=$d $args"; failed=1; }
    done
done
exit "$failed"
