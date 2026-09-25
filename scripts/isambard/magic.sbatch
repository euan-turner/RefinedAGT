#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --job-name=agt-magic
#SBATCH --gpus=1
#SBATCH --time=01:00:00
# The unrefined MAGIC run and the single-GPU ladder rungs 2^5, 2^10, 3^5x2^5, 3^10 (under 2^16 leaves).
# Measured locally: ~20 ms/leaf on one RTX 5090 in float64; scaled by the half-moons 5090:GH200 ratio
# (3.9 vs 1.42 ms/leaf) that is ~7 ms/leaf here, so 3^10 ~7 min. magic_sharded.sbatch runs the rest.
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/isambard/env.sh"
cd "$AGT_REPO/scripts/poisoning_paper"
python magic_run.py --baseline --rungs 2^5 2^10 3^5x2^5 3^10
