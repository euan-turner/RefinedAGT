"""
The complete list of certified runs behind the three CIFAR-10 experiments, kept in one place so the
Slurm job lists and the analysis scripts cannot disagree about what was run. A run is a feature width,
an attack (``k``, ``eps``) and an optional refinement (``n_splits``, ``n_dims``); see cifar_pca for
the terms.

The experiments (each analysis script states its hypotheses):
    - Threat grid (``e1_runs``, analysed by cifar_pca_threat_sweep.py). At d=20, every pair of
      ``k`` in K_POISONS and ``eps`` in EPSILONS, with four runs each: unrefined; the 12 most
      sensitive coordinates bisected (2^12 leaves); all 20 bisected (2^20 leaves); and an unrefined
      reference at eps/2.
    - Refinement depth (``e2_runs``, analysed by cifar_pca_leaf_sweep.py). One attack (d=20, k=100,
      eps=0.05), unrefined and at every partition in E2_SCHEDULE.
    - Feature width (``e3_runs``, analysed by cifar_pca_dims_sweep.py). The refinement-depth attack
      at each d in E3_DIMS: unrefined, top 12 bisected, all d bisected, and the eps/2 reference.
Runs that two experiments share are listed once (``all_runs``).

As a command, it prints runs as cifar_pca_run.py argument lines, cheapest first, for the Slurm jobs:
    --gpus 1   only runs with fewer than 2^16 leaves (one GPU is enough)
    --gpus 4   only runs with at least 2^16 leaves (sharded across a 4-GPU node)
    --missing  only runs with no cached result, plus a ``--float64`` rerun for every float32 run whose
               bounds were violated (needs the selection for each d to exist)

On the analysis side, ``collect`` loads the reported result of each run. It never trains: if any run is
missing, it exits with the list.

Key external dependencies: the sibling module cifar_pca.
"""

import argparse
import collections
import sys

import torch

import cifar_pca

Run = collections.namedtuple("Run", ["d", "k", "eps", "refine"])  # refine: None or (n_splits, n_dims)

K_POISONS = [10, 25, 50, 100, 200]
EPSILONS = [0.005, 0.01, 0.02, 0.05, 0.1]
SCREEN_SPLIT_DIMS = 12

E2_D, E2_K, E2_EPS = 20, 100, 0.05
# the bisection ladder up to every dimension, then equal-cost alternatives on fewer, more finely cut ones
E2_SCHEDULE = [(2, 4), (2, 8), (2, 12), (2, 16), (2, 20), (16, 3), (4, 8), (4, 10)]

E3_DIMS = [20, 22, 24]

SHARDED_FROM_LEAVES = 2**16  # runs with at least this many leaves take a 4-GPU node


def e1_runs():
    """Threat grid at d=20: baseline, bisect top 12, bisect all, and the eps/2 baseline diagnostic."""
    runs = []
    for k in K_POISONS:
        for eps in EPSILONS:
            runs += [Run(20, k, eps, None), Run(20, k, eps, (2, SCREEN_SPLIT_DIMS)), Run(20, k, eps, (2, 20)),
                     Run(20, k, eps / 2, None)]
    return runs


def e2_runs():
    """Refinement depth at one cell: the baseline and every rung of the schedule."""
    return [Run(E2_D, E2_K, E2_EPS, None)] + [Run(E2_D, E2_K, E2_EPS, rung) for rung in E2_SCHEDULE]


def e3_runs():
    """Feature width at the E2 cell: baseline, bisect top 12, bisect all d, and the eps/2 diagnostic."""
    return [run for d in E3_DIMS for run in (Run(d, E2_K, E2_EPS, None), Run(d, E2_K, E2_EPS, (2, SCREEN_SPLIT_DIMS)),
                                             Run(d, E2_K, E2_EPS, (2, d)), Run(d, E2_K, E2_EPS / 2, None))]


def leaves(run):
    return 1 if run.refine is None else run.refine[0] ** run.refine[1]


def all_runs():
    """Every run of E1-E3 once, cheapest first."""
    return sorted(dict.fromkeys(e1_runs() + e2_runs() + e3_runs()), key=lambda run: (leaves(run), run.d))


def collect(runs):
    """
    The reported result (``cifar_pca.reported_cell``) of every run, keyed by run. Never trains: exits
    listing the missing runs, and the float64 reruns that unsound float32 runs still need, instead.
    """
    results, missing = {}, []
    for run in runs:
        try:
            results[run] = cifar_pca.reported_cell(run.d, run.k, run.eps, run.refine)
        except FileNotFoundError as error:
            missing.append(f"  {run_args(run)}: {error}")
    if missing:
        sys.exit(f"{len(missing)} of {len(runs)} runs not available:\n" + "\n".join(missing))
    return results


def run_args(run):
    args = f"--pca-dims {run.d} --k {run.k} --eps {run.eps}"
    return args if run.refine is None else f"{args} --n-splits {run.refine[0]} --split-dims {run.refine[1]}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, choices=(1, 4))
    parser.add_argument("--missing", action="store_true")
    args = parser.parse_args()
    for run in all_runs():
        if args.gpus is not None and (args.gpus == 4) != (leaves(run) >= SHARDED_FROM_LEAVES):
            continue
        if not args.missing:
            print(run_args(run))
            continue
        record = cifar_pca.cached_record(run.d, run.k, run.eps, run.refine)
        if record is None:
            print(run_args(run))
        elif (record["violation"] > cifar_pca.VIOLATION_TOLERANCE
              and cifar_pca.cached_record(run.d, run.k, run.eps, run.refine, dtype=torch.float64) is None):
            print(f"{run_args(run)} --float64")
