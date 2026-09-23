#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --job-name=agt-halfmoons-sharded
#SBATCH --gpus=4
# Half-moons rungs of at least 2^16 leaves, sharded across the 4 GPUs, in sequence.
# Usage: sbatch --time=<hh:mm:ss> halfmoons_sharded.sbatch <n_splits> [<n_splits> ...]
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/isambard/env.sh"
cd "$AGT_REPO/scripts/poisoning_paper"
torchrun --standalone --nproc-per-node=4 halfmoons_run.py --n-splits "$@"
