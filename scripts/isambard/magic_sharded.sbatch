#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --job-name=agt-magic-sharded
#SBATCH --gpus=4
# MAGIC ladder rungs of at least 2^16 leaves, sharded across the 4 GPUs, in sequence. Each finished rung
# is cached, so a job that times out keeps the rungs it completed.
# Usage: sbatch --time=<hh:mm:ss> magic_sharded.sbatch <rung> [<rung> ...]   (rungs as N^D or N^DxM^E)
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/isambard/env.sh"
cd "$AGT_REPO/scripts/poisoning_paper"
torchrun --standalone --nproc-per-node=4 magic_run.py --rungs "$@"
