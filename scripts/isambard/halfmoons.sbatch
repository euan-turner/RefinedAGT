#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --job-name=agt-halfmoons
#SBATCH --gpus=1
#SBATCH --time=01:00:00
# The unrefined run and the single-GPU rungs 2^6..6^6 (under 2^16 leaves). Measured locally: 4^6 in 16 s
# on one RTX 5090 in float64. halfmoons_sharded.sbatch runs the larger rungs.
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/isambard/env.sh"
cd "$AGT_REPO/scripts/poisoning_paper"
python halfmoons_run.py --baseline --n-splits 2 3 4 5 6
