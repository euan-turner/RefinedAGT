"""
Runs one certified-training run of the UCI house-electric refinement experiment and prints its result.
This is the unit of work the Slurm jobs launch (uci_single.sbatch, uci_sharded.sbatch). The fixed
configuration, and the meaning of the metrics, are in uci_refinement.

Usage: [torchrun --standalone --nproc-per-node N] uci_run.py [--n-splits N]
Omitting ``--n-splits`` gives the unrefined run. Otherwise every one of the 11 input features is cut
into N pieces (N ** 11 leaves). Under torchrun the leaves are sharded across the GPUs, with the same
result as on one. The run is cached, so launching it again returns at once. Rank 0 prints one JSON
line: the run record (violation, seconds, world size, peak memory) and the certified test metrics.

Key external dependencies: torch.distributed (NCCL, for sharded refinement), and the sibling module
uci_refinement.
"""

import argparse
import json

import torch.distributed as dist

import uci_refinement

parser = argparse.ArgumentParser()
parser.add_argument("--n-splits", type=int, help="refinement splits per input dimension; omit for the unrefined baseline")
args = parser.parse_args()

uci_refinement.init_distributed()
bounded_model, record = uci_refinement.run(args.n_splits)

if uci_refinement.rank() == 0:
    print(json.dumps({"n_splits": args.n_splits, **record, **uci_refinement.certified_metrics(bounded_model)}))
if dist.is_initialized():
    dist.destroy_process_group()
