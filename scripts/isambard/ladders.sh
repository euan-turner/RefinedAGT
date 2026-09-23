#!/bin/bash
# Submit the refinement-ladder campaign (README section 5): the CIFAR-10 ladder (bisection and
# quadrisection) at k=200, eps=0.02, the OCT-MNIST ladder at d=15, eps=0.1, k in {200, 400}, and the
# half-moons ladder extended past 6^6 on 4 GPUs. Run on a login node after sourcing env.sh, from a
# checkout whose production campaign has finished (the ladders reuse its cached runs, pre-trained models
# and CIFAR selection). No dependencies between jobs; submit ladders_aggregate.sbatch once they have all
# finished.
# Safe to re-run: the CIFAR and OCT-MNIST lists contain only uncached runs, and half-moons runs return
# at once from the cache.
#
# --time values are estimates. CIFAR 2^16, 4^8, 4^10: measured 62 s, 61 s, 911 s on 4 GPUs (E2).
# OCT-MNIST: scaled from the threat grid's 2^15 run (49 s on 4 GPUs) by leaf count, so 3^12 ~13 min,
# 4^10 ~26 min, 3^15 ~6 h. Half-moons: scaled from 3.9 ms/leaf on one RTX 5090 in float64, over 4 GPUs,
# so 8^6+10^6+12^6 ~1.2 h, 16^6 ~5 h.
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh first}"
T_CIFAR_D20=00:45:00; T_OCT_MID=01:30:00; T_OCT_BIG=12:00:00; T_HM_MID=04:00:00; T_HM_BIG=12:00:00

S="$AGT_REPO/scripts/isambard"
M="$AGT_ROOT/manifests/ladders-$(date +%Y%m%d-%H%M)"
mkdir -p "$M"
cd "$AGT_REPO/scripts/poisoning_paper"
python cifar_pca_manifest.py --gpus 1 --missing > "$M/cifar_1gpu.txt"
python cifar_pca_manifest.py --gpus 4 --missing > "$M/cifar_4gpu.txt"
python octmnist_pca_ladder.py --gpus 1 --missing > "$M/octmnist_1gpu.txt"
python octmnist_pca_ladder.py --gpus 4 --missing > "$M/octmnist_4gpu.txt"
grep -Ev -- '--n-splits 3 --split-dims 15( |$)' "$M/octmnist_4gpu.txt" > "$M/octmnist_4gpu_mid.txt" || true
grep -E  -- '--n-splits 3 --split-dims 15( |$)' "$M/octmnist_4gpu.txt" > "$M/octmnist_4gpu_big.txt" || true
wc -l "$M"/*.txt
if grep -qv -- '--pca-dims 20 --k 200 --eps 0.02 ' "$M"/cifar_*.txt; then
    echo "WARNING: the CIFAR lists hold runs outside the ladder; the production campaign is incomplete" >&2
fi

cd "$AGT_ROOT/logs"
submit() {   # submit [sbatch options] <sbatch file> [args...]
    local id
    id=$(sbatch --parsable "$@")
    echo "$id: $*"
}
submit_if_any() {   # submit_if_any <manifest> [sbatch options] <sbatch file>: skip an empty list
    local manifest=$1; shift
    if [ -s "$manifest" ]; then submit "$@" "$manifest"; fi
}
submit_array() {   # submit_array <time> <manifest> <sbatch file>: one task per manifest line
    local n
    n=$(wc -l < "$2")
    if [ "$n" -gt 0 ]; then submit --time="$1" --array="0-$((n - 1))" "$3" "$2"; fi
}
submit_if_any "$M/cifar_1gpu.txt" "$S/cifar_cheap.sbatch"
submit_array "$T_CIFAR_D20" "$M/cifar_4gpu.txt" "$S/cifar_sharded.sbatch"
submit_if_any "$M/octmnist_1gpu.txt" "$S/octmnist_ladder_cheap.sbatch"
submit_array "$T_OCT_MID" "$M/octmnist_4gpu_mid.txt" "$S/octmnist_ladder_sharded.sbatch"
submit_array "$T_OCT_BIG" "$M/octmnist_4gpu_big.txt" "$S/octmnist_ladder_sharded.sbatch"
submit "$S/halfmoons.sbatch"
submit --time="$T_HM_MID" "$S/halfmoons_sharded.sbatch" 8 10 12
submit --time="$T_HM_BIG" "$S/halfmoons_sharded.sbatch" 16
echo "Manifests in $M"
