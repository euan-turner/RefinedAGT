"""
Runs certified-training runs of the MAGIC refinement experiment and prints their results. This is the
unit of work the Slurm jobs launch (magic.sbatch, magic_sharded.sbatch). The fixed configuration, and
the meaning of the metrics, are in magic_refinement.

Usage: [torchrun --standalone --nproc-per-node N] magic_run.py [--baseline] [--rungs RUNG [RUNG ...]]
Each rung is ``N^D`` (the D most sensitive features cut into N pieces, N^D leaves) or ``N^DxM^E`` (and
the next E features cut into M, N^D * M^E leaves); ``--baseline`` adds the unrefined run. Under torchrun the leaves are sharded
across the GPUs, with the same result as on one. Runs are cached, so launching one again returns at
once. Rank 0 prints one JSON line per run: the run record (violation, seconds, world size, peak memory)
and the certified test metrics.

Key external dependencies: torch.distributed (NCCL, for sharded refinement), and the sibling module
magic_refinement.
"""

import argparse
import json

import torch.distributed as dist

import magic_refinement

parser = argparse.ArgumentParser()
parser.add_argument("--rungs", type=magic_refinement.parse_rung, nargs="*", default=[],
                    help="refinement rungs, e.g. 2^5 2^10 3^5x2^5")
parser.add_argument("--baseline", action="store_true", help="also run the unrefined baseline")
args = parser.parse_args()

magic_refinement.init_distributed()
for refine in ([None] if args.baseline else []) + args.rungs:
    bounded_model, record = magic_refinement.run(refine)
    if magic_refinement.rank() == 0:
        print(json.dumps({"rung": magic_refinement.rung_label(refine), **record, **magic_refinement.certified_metrics(bounded_model)}),
              flush=True)
if dist.is_initialized():
    dist.destroy_process_group()
