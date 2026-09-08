#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --gpus=1
# Runs under 2^16 leaves, in sequence on one GPU. Argument: a run list from `run.py <task> --list
# --gpus 1`. --time and --job-name are given at submission (submit.sh). Finished runs are cached, so a
# job that times out keeps the runs it completed.
set -euo pipefail
: "${AGT_REPO:?source scripts/refinement_paper/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/refinement_paper/isambard/env.sh"
cd "$AGT_REPO/scripts/refinement_paper"
MANIFEST=$1
failed=0
while read -r line; do
    args=${line%%#*}
    echo "run: $args"
    python run.py $args < /dev/null || { echo "FAILED: $args"; failed=1; }   # /dev/null: python must not eat the run list
done < "$MANIFEST"
exit "$failed"
