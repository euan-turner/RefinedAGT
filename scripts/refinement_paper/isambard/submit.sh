#!/bin/bash
# Submit the runs of one task that are not yet cached: by default the unrefined run and the task's
# whole refinement ladder at its default attack. Run on a login node after sourcing env.sh, once
# prepare.sbatch has finished.
#
# Usage: submit.sh <task> [run.py options]
#   submit.sh halfmoons                                  the half-moons ladder
#   submit.sh cifar --k 50 100 200 --eps 0.01 0.02       a threat-model sweep of the CIFAR ladder
#   submit.sh octmnist --rungs unrefined 2^15            chosen rungs only
#
# Runs under 2^16 leaves go to one 1-GPU job, in sequence; each larger run is its own 4-GPU job. Time
# limits are twice the estimate from the task's SECONDS_PER_LEAF plus 30 minutes, capped at the 24 h
# maximum. Safe to re-run: only runs with no cached record are submitted, plus a float64 rerun of any
# run whose bounds were violated.
set -euo pipefail
: "${AGT_REPO:?source scripts/refinement_paper/isambard/env.sh first}"
TASK=${1:?usage: submit.sh <task> [run.py options]}
shift
D="$AGT_REPO/scripts/refinement_paper"
mkdir -p "$AGT_ROOT/manifests" "$AGT_ROOT/logs"
# unique even when one task is submitted several times a second (reproduce.sh): a 1-GPU job reads its
# run list when it starts, so a later submission must never overwrite it
M=$(mktemp -d "$AGT_ROOT/manifests/$TASK-$(date +%Y%m%d-%H%M%S)-XXXX")

cd "$D"
python run.py "$TASK" "$@" --list --gpus 1 --missing > "$M/1gpu.txt"
python run.py "$TASK" "$@" --list --gpus 4 --missing > "$M/4gpu.txt"
wc -l "$M"/*.txt

limit() {   # limit <GPU-seconds> <gpus>: a --time value
    local seconds=$((2 * $1 / $2 + 1800))
    if [ "$seconds" -gt 86400 ]; then
        echo "WARNING: estimated $((seconds / 3600)) h exceeds the 24 h limit; submitting at 24 h" >&2
        seconds=86400
    fi
    printf '%02d:%02d:00' $((seconds / 3600)) $((seconds % 3600 / 60))
}
submit() {   # submit [sbatch options] <sbatch file> [args...]
    local id
    id=$(sbatch --parsable "$@")
    echo "$id: $*"
}

cd "$AGT_ROOT/logs"   # job logs (%x_%j.out) are written to the submit directory
if [ -s "$M/1gpu.txt" ]; then
    total=$(awk -F'# ' '{s += $2} END {print int(s)}' "$M/1gpu.txt")
    submit --time="$(limit "$total" 1)" --job-name="agt-$TASK" "$D/isambard/run_1gpu.sbatch" "$M/1gpu.txt"
fi
while read -r line; do
    args=${line%%#*}
    submit --time="$(limit "${line##*# }" 4)" --job-name="agt-$TASK-4gpu" "$D/isambard/run_4gpu.sbatch" $args
done < "$M/4gpu.txt"
echo "Run lists in $M"
