#!/bin/bash
# Submit the MAGIC refinement ladder (README section 6). Run on a login node after sourcing env.sh.
# Safe to re-run: cached rungs return at once.
#
# --time values are estimates: ~20 ms/leaf measured on one RTX 5090, scaled to ~7 ms/leaf on one
# GH200 by the half-moons ratio, over 4 GPUs. 4^5x3^5 ~8 min, 4^10 ~30 min, 5^5x4^5 ~1.6 h, 5^10 ~5 h,
# 6^5x5^5 ~12 h. 6^10 (~31 h) is past the 24 h limit.
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh first}"
S="$AGT_REPO/scripts/isambard"

# compute nodes may have no network: fetch MAGIC from OpenML into $AGT_ROOT/.data here
cd "$AGT_REPO/scripts/poisoning_paper"
python -c "import magic_refinement; magic_refinement.get_datasets()"

cd "$AGT_ROOT/logs"
submit() {   # submit [sbatch options] <sbatch file> [args...]
    local id
    id=$(sbatch --parsable "$@")
    echo "$id: $*"
}
submit "$S/magic.sbatch"
submit --time=04:00:00 "$S/magic_sharded.sbatch" 4^5x3^5 4^10 5^5x4^5
submit --time=10:00:00 "$S/magic_sharded.sbatch" 5^10
submit --time=24:00:00 "$S/magic_sharded.sbatch" 6^5x5^5
