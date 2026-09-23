"""
Runs one certified-training run of the OCT-MNIST refinement ladder and prints its result. This is the
unit of work the Slurm jobs launch (octmnist_ladder_cheap.sbatch, octmnist_ladder_sharded.sbatch); each
line printed by ``python octmnist_pca_ladder.py`` is one set of arguments. The run list and the reason
for it are in octmnist_pca_ladder.

Usage: [torchrun --standalone --nproc-per-node 4] octmnist_pca_ladder_run.py --k 200
           [--n-splits 3 --split-dims 15]
Omitting ``--split-dims`` gives the unrefined run. Under torchrun the leaves are sharded across the
GPUs, with the same result as on one. The run is cached, so launching it again returns at once. Rank 0
prints one JSON line with the run record (seconds, world size, peak memory), the worst bound violation
and the certified test metrics.

Key external dependencies: torch.distributed (NCCL, for sharded refinement), and the sibling modules
octmnist_pca and octmnist_pca_ladder.
"""

import argparse
import json
import sys

import torch.distributed as dist

import octmnist_pca
import octmnist_pca_ladder as ladder

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, required=True)
parser.add_argument("--n-splits", type=int, default=2)
parser.add_argument("--split-dims", type=int, help="features to split; omit for the unrefined baseline")
args = parser.parse_args()

octmnist_pca.init_distributed()
refine = None if args.split_dims is None else (args.n_splits, args.split_dims)
pca_state = octmnist_pca.fit_pca()
drusen_train, drusen_test, clean_train, clean_test = octmnist_pca.get_datasets(ladder.PCA_DIMS, pca_state)
model = octmnist_pca.get_pretrained_model(ladder.PCA_DIMS, pca_state)
bounded_model, violation = ladder.run(args.k, refine, model, drusen_train, clean_train)

if octmnist_pca.rank() == 0:
    metrics = ladder.certified_metrics(bounded_model, drusen_test, clean_test)
    print(json.dumps({"d": ladder.PCA_DIMS, "k": args.k, "eps": ladder.EPSILON, "refine": refine,
                      "violation": violation, "cost": ladder.cost_record(args.k, refine), **metrics}))
    if violation > ladder.VIOLATION_TOLERANCE:
        print(f"WARNING: bound violation {violation:.2e}; this certificate is unsound", file=sys.stderr)
if dist.is_initialized():
    dist.destroy_process_group()
