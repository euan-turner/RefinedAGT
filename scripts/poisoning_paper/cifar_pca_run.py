"""
Runs one certified-training run of the CIFAR-10 experiments and prints its result. This is the unit of
work the Slurm jobs launch: each line of the job lists written by cifar_pca_manifest.py is one set of
arguments to this script.

A run is identified by its feature width, attack (``k``, ``eps``) and optional refinement
(``n_splits`` pieces on each of ``split_dims`` coordinates). Every other setting comes from the
hyperparameters selected for that width (cifar_pca_selection.py). The result is cached (see
cifar_pca), so launching a finished run again returns at once. Rank 0 prints one JSON line with the
wall time, number of GPUs, worst bound violation, and the val and test metrics.

Usage: [torchrun --nproc-per-node 4] cifar_pca_run.py --pca-dims 20 --k 100 --eps 0.05
           [--n-splits 2 --split-dims 20] [--float64]
Omitting ``--split-dims`` gives the unrefined run. Under torchrun the leaves are sharded across the
GPUs, with the same result as on one. ``--float64`` reruns a run whose float32 bounds were violated.

Key external dependencies: torch.distributed (NCCL, for sharded refinement), and the sibling module
cifar_pca.
"""

import argparse
import json
import sys

import torch
import torch.distributed as dist

import cifar_pca

parser = argparse.ArgumentParser()
# not --d or --n-dims: torchrun reads abbreviations of its own options (--d -> --duplicate-*) even after the script
parser.add_argument("--pca-dims", type=int, required=True)
parser.add_argument("--k", type=int, required=True)
parser.add_argument("--eps", type=float, required=True)
parser.add_argument("--n-splits", type=int, default=2)
parser.add_argument("--split-dims", type=int, help="coordinates to split; omit for the unrefined baseline")
parser.add_argument("--float64", action="store_true", help="the rerun of a run whose float32 bounds were violated")
args = parser.parse_args()

cifar_pca.init_distributed()
refine = None if args.split_dims is None else (args.n_splits, args.split_dims)
result = cifar_pca.run_cell(args.pca_dims, args.k, args.eps, refine, dtype=torch.float64 if args.float64 else torch.float32)

if cifar_pca.rank() == 0:
    print(json.dumps({"d": args.pca_dims, "k": args.k, "eps": args.eps, "refine": refine, **result}))
    if result["violation"] > cifar_pca.VIOLATION_TOLERANCE:
        print(f"WARNING: bound violation {result['violation']:.2e}; rerun with --float64 before reporting",
              file=sys.stderr)
if dist.is_initialized():
    dist.destroy_process_group()
