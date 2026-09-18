#!/bin/bash
# Submit every production job at once, with no dependencies between them: the CIFAR, OCT-MNIST, UCI
# and half-moons experiments, and the E3 pre-training-radius control. Run on a login node after
# sourcing env.sh, once prepare.sbatch has finished. Times from benchmark 6560456 (README §3.3).
# Safe to re-run: CIFAR manifests list only uncached runs (and any float64 reruns an unsound run
# needs), and every other job reads its finished runs from the cache.
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh first}"
T_D20=00:45:00; T_D22=03:00:00; T_D24=12:00:00; T_UCI3=06:00:00; T_UCI4=20:00:00

S="$AGT_REPO/scripts/isambard"
M="$AGT_ROOT/manifests/$(date +%Y%m%d-%H%M)"
mkdir -p "$M"
cd "$AGT_REPO/scripts/poisoning_paper"
python cifar_pca_manifest.py --gpus 1 --missing > "$M/cifar_1gpu.txt"
python cifar_pca_manifest.py --gpus 4 --missing > "$M/cifar_4gpu.txt"
grep -Ev -- '--split-dims 2[24]( |$)' "$M/cifar_4gpu.txt" > "$M/cifar_4gpu_d20.txt" || true
grep -E  -- '--split-dims 22( |$)'    "$M/cifar_4gpu.txt" > "$M/cifar_4gpu_d22.txt" || true
grep -E  -- '--split-dims 24( |$)'    "$M/cifar_4gpu.txt" > "$M/cifar_4gpu_d24.txt" || true
wc -l "$M"/*.txt

cd "$AGT_ROOT/logs"
submit() {   # submit [sbatch options] <sbatch file> [args...]
    local id
    id=$(sbatch --parsable "$@")
    echo "$id: $*"
}
submit_array() {   # submit_array <time> <manifest>: one cifar_sharded task per manifest line
    local n
    n=$(wc -l < "$2")
    if [ "$n" -gt 0 ]; then submit --time="$1" --array="0-$((n - 1))" "$S/cifar_sharded.sbatch" "$2"; fi
}
submit "$S/cifar_cheap.sbatch" "$M/cifar_1gpu.txt"
submit "$S/octmnist_single.sbatch"
submit "$S/octmnist_threat.sbatch"
submit "$S/uci_single.sbatch"
submit "$S/halfmoons.sbatch"
submit --time="$T_UCI3" "$S/uci_sharded.sbatch" 3
submit --time="$T_UCI4" "$S/uci_sharded.sbatch" 4
submit_array "$T_D20" "$M/cifar_4gpu_d20.txt"
submit_array "$T_D22" "$M/cifar_4gpu_d22.txt"
submit_array "$T_D24" "$M/cifar_4gpu_d24.txt"
submit "$S/pt_control_cheap.sbatch"
submit --job-name=agt-pt-control-d22 --time="$T_D22" "$S/pt_control_sharded.sbatch" 22
submit --job-name=agt-pt-control-d24 --time="$T_D24" "$S/pt_control_sharded.sbatch" 24
echo "Manifests in $M"
