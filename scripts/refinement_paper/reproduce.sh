#!/bin/bash
# Single entrypoint for every experiment of the input-ball refinement paper: the runs, their results
# files and the figures. EXPERIMENTS below is the one list of what the paper reports; every mode reads it.
#
# Usage: bash reproduce.sh <mode>
#   list [--missing]   every run, cheapest first per experiment, with its estimated GPU-seconds, and
#                      the total; --missing lists only runs with no cached record (and float64 reruns)
#   local              train every run on this machine, in sequence on one GPU (CUDA_VISIBLE_DEVICES
#                      picks it). The largest rungs take a day each on one GPU; see README.md
#   isambard           from an Isambard-AI login node, after sourcing isambard/env.sh and once
#                      isambard/smoke.sbatch has passed: submits isambard/prepare.sbatch and, held until
#                      it succeeds, every uncached run as Slurm jobs (isambard/submit.sh)
#   results            the results file of every experiment (.results/<name>.json) and the figures
#                      (.figures/refinement_ladders.*, .figures/threat_grid.*); never trains, and lists
#                      the runs still missing
#
# Every mode skips runs that are already cached, so each is safe to run again, and finished runs are
# never recomputed: commenting out an experiment is only ever needed to avoid submitting its jobs.
# Outputs go under $AGT_ROOT when it is set, otherwise next to this script (harness.ROOT).
#
# Design doc: docs/input_refinement_soundness.md (what the certificates mean). The choice of attacks is
# argued in README.md, "Experiments".
#
# Key external dependencies: bash, the sibling scripts run.py, aggregate.py, plot_ladders.py and
# plot_threat_grid.py, and Slurm (sbatch) for the isambard mode.
set -euo pipefail

# <results name>  <task> [--k K ...] [--eps E ...] [--rungs RUNG ...]
# The options are those of run.py and aggregate.py; without them, a task's ladder attack and ladder.
EXPERIMENTS=(
    # Ladders at large k and small epsilon: the paper figure (plot_ladders.py)
    "cifar             cifar"
    "octmnist          octmnist"
    "halfmoons         halfmoons"
    "magic             magic"
    # Ladders at small k and large epsilon
    "halfmoons_smallk  halfmoons --k 25 --eps 0.05"
    "magic_smallk      magic --k 25 --eps 0.1"
    # Image tasks at small k: certified close to nominal unrefined; one rung bisecting every feature
    "cifar_smallk      cifar --k 25 --eps 0.1 --rungs unrefined 2^20"
    "octmnist_smallk   octmnist --k 25 --eps 0.1 --rungs unrefined 2^15"
    # Unrefined certificate over the threat-model grid (plot_threat_grid.py)
    "cifar_threat      cifar --k 5 10 25 50 100 200 --eps 0.005 0.01 0.02 0.05 0.1 0.2 --rungs unrefined"
    "octmnist_threat   octmnist --k 5 10 25 50 100 200 --eps 0.005 0.01 0.02 0.05 0.1 0.2 --rungs unrefined"
    "halfmoons_threat  halfmoons --k 5 10 25 50 100 200 --eps 0.005 0.01 0.02 0.05 0.1 0.2 --rungs unrefined"
    "magic_threat      magic --k 5 10 25 50 100 200 --eps 0.005 0.01 0.02 0.05 0.1 0.2 --rungs unrefined"
)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${AGT_ROOT:-$HERE}/.results"
MODE=${1:?usage: reproduce.sh list [--missing] | local | isambard | results}
shift
cd "$HERE"

for_each_experiment() {   # for_each_experiment <function>: calls <function> <name> <task> [options...]
    local line
    for line in "${EXPERIMENTS[@]}"; do
        # shellcheck disable=SC2086 # word splitting of the manifest line is intended
        "$1" $line
    done
}

list_runs() { shift; python run.py "$@" --list ${MISSING:+--missing}; }
train() { shift; python run.py "$@"; }
submit() { shift; bash "$HERE/isambard/submit.sh" "$@"; }
aggregate() {
    local name=$1
    shift
    python aggregate.py "$@" --out "$RESULTS/$name.json" || FAILED=1
}

case "$MODE" in
    list)
        MISSING=""
        [ "${1:-}" = "--missing" ] && MISSING=1
        # a run shared by two experiments (the grid holds every unrefined run) is listed once
        for_each_experiment list_runs | awk -F'# ' '!seen[$1]++ {print; s += $2} END {printf "# total %.1f GPU-hours\n", s / 3600}'
        ;;
    local)
        for_each_experiment train
        ;;
    isambard)
        : "${AGT_REPO:?source isambard/env.sh first}"
        mkdir -p "$AGT_ROOT/logs"
        prepare=$(cd "$AGT_ROOT/logs" && sbatch --parsable "$HERE/isambard/prepare.sbatch")
        echo "$prepare: prepare.sbatch"
        export SBATCH_DEPENDENCY="afterok:$prepare"   # read by every sbatch that submit.sh calls
        for_each_experiment submit
        ;;
    results)
        mkdir -p "$RESULTS"
        FAILED=0
        for_each_experiment aggregate
        python plot_ladders.py || FAILED=1
        python plot_threat_grid.py || FAILED=1
        exit "$FAILED"
        ;;
    *)
        echo "unknown mode '$MODE'; choose list, local, isambard or results" >&2
        exit 2
        ;;
esac
