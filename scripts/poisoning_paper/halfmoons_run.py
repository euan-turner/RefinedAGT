"""
Runs certified-training runs of the half-moons refinement experiment and prints their results. This is
the unit of work the Slurm jobs launch (halfmoons.sbatch, halfmoons_sharded.sbatch). The fixed
configuration, and the meaning of the metrics, are in halfmoons_refinement.

Usage: [torchrun --standalone --nproc-per-node N] halfmoons_run.py [--n-splits N [N ...]] [--baseline]
Each ``--n-splits`` value cuts every one of the 6 input features into N pieces (N ** 6 leaves);
``--baseline`` adds the unrefined run. Under torchrun the leaves are sharded across the GPUs, with the
same result as on one. Runs are cached, so launching one again returns at once. Rank 0 prints one JSON
line per run: the run record (violation, seconds, world size, peak memory) and the certified test
metrics.

Key external dependencies: torch.distributed (NCCL, for sharded refinement), and the sibling module
halfmoons_refinement.
"""

import argparse
import json

import torch.distributed as dist

import halfmoons_refinement

parser = argparse.ArgumentParser()
parser.add_argument("--n-splits", type=int, nargs="*", default=[], help="refinement splits per input feature")
parser.add_argument("--baseline", action="store_true", help="also run the unrefined baseline")
args = parser.parse_args()

halfmoons_refinement.init_distributed()
for n_splits in ([None] if args.baseline else []) + args.n_splits:
    bounded_model, record = halfmoons_refinement.run(n_splits)
    if halfmoons_refinement.rank() == 0:
        print(json.dumps({"n_splits": n_splits, **record, **halfmoons_refinement.certified_metrics(bounded_model)}),
              flush=True)
if dist.is_initialized():
    dist.destroy_process_group()
