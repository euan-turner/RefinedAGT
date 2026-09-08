"""
Launches certified-training runs of the input-ball refinement experiments and prints their records.
This is the only script that trains; it is the unit of work the Slurm jobs launch (isambard/).

Usage:
    [torchrun --standalone --nproc-per-node N] python run.py <task> [--k K ...] [--eps E ...]
        [--rungs RUNG ...] [--dtype float64]
    python run.py <task> --prepare
    python run.py <task> [--k K ...] [--eps E ...] [--rungs RUNG ...] --list [--gpus {1,4}] [--missing]

``<task>`` is one of tasks.NAMES. ``--k`` and ``--eps`` default to the attack of the task's ladder;
giving several values runs every combination (a threat-model sweep). ``--rungs`` are partitions of the
eps-ball: ``unrefined``, ``N^D`` (the D most sensitive features cut into N pieces) or ``N^DxM^E``
(and the next E into M); the default is the unrefined run and the task's whole ladder. Runs are
cached, so launching one again returns at once. Under torchrun the leaves are sharded across the GPUs,
with the same result as on one. Rank 0 prints one JSON record per run.

``--prepare`` downloads the task's data and builds what it caches (PCA basis, pre-trained model).

``--list`` prints the runs as run.py argument lines, cheapest first, instead of running them:
    --gpus 1   only runs with fewer than 2^16 leaves (one GPU is enough)
    --gpus 4   only runs with at least 2^16 leaves (sharded across a 4-GPU node)
    --missing  only runs with no cached record, plus a ``--dtype float64`` rerun for every run whose
               bounds were violated in the task's default precision
Each line ends with ``# <estimated GPU-seconds>``, which isambard/submit.sh turns into a time limit.

Key external dependencies: torch.distributed (NCCL, for sharded refinement), and the sibling modules
harness and tasks.
"""

import argparse
import itertools
import json
import sys

import torch

import harness
import tasks

parser = argparse.ArgumentParser()
parser.add_argument("task", choices=tasks.NAMES)
parser.add_argument("--k", type=int, nargs="+", help="attack sizes; default the task's K_POISON")
parser.add_argument("--eps", type=float, nargs="+", help="attack radii; default the task's EPSILON")
parser.add_argument("--rungs", nargs="+", help="partitions; default 'unrefined' and the task's LADDER")
parser.add_argument("--dtype", choices=list(harness.DTYPES), help="default the task's DTYPE")
parser.add_argument("--prepare", action="store_true")
parser.add_argument("--list", action="store_true")
parser.add_argument("--gpus", type=int, choices=(1, 4))
parser.add_argument("--missing", action="store_true")
args = parser.parse_args()

task = tasks.get(args.task)
rungs = args.rungs or [harness.UNREFINED] + task.LADDER
runs = [
    (k, epsilon, harness.parse_rung(rung))
    for k, epsilon, rung in itertools.product(args.k or [task.K_POISON], args.eps or [task.EPSILON], rungs)
]
dtype = harness.DTYPES[args.dtype] if args.dtype else task.DTYPE

if args.prepare:
    task.prepare()
    sys.exit()

if args.list:
    for k, epsilon, splits in sorted(runs, key=lambda run: harness.n_leaves(run[2])):
        leaves = harness.n_leaves(splits)
        if args.gpus is not None and (args.gpus == 4) != (leaves >= harness.SHARDED_FROM_LEAVES):
            continue
        run_dtype = dtype
        if args.missing:
            record = harness.load_record(task, k, epsilon, splits, dtype)
            if record is not None:
                unsound = record["violation"] > harness.VIOLATION_TOLERANCE and dtype != torch.float64
                if not unsound or harness.load_record(task, k, epsilon, splits, torch.float64) is not None:
                    continue
                run_dtype = torch.float64
        print(f"{harness.run_args(task, k, epsilon, splits, run_dtype)}  # {leaves * task.SECONDS_PER_LEAF:.0f}")
    sys.exit()

harness.init_distributed()
task.prepare()
for k, epsilon, splits in runs:
    record = harness.run(task, k, epsilon, splits, dtype)
    if harness.rank() == 0:
        print(json.dumps(record), flush=True)
        if record["violation"] > harness.VIOLATION_TOLERANCE:
            print(f"WARNING: bound violation {record['violation']:.2e}; this certificate is unsound. "
                  f"Rerun with --dtype float64.", file=sys.stderr)
harness.finalize_distributed()
