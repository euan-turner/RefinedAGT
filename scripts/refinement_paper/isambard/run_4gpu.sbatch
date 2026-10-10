#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --gpus=4
# One run of at least 2^16 leaves, its leaves sharded across the 4 GPUs of a node. Arguments: the
# run.py arguments of the run. --time and --job-name are given at submission (submit.sh).
set -euo pipefail
: "${AGT_REPO:?source scripts/refinement_paper/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/refinement_paper/isambard/env.sh"
cd "$AGT_REPO/scripts/refinement_paper"
echo "run: $*"
torchrun --standalone --nproc-per-node=4 run.py "$@"
