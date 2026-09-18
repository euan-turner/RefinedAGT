"""
Control for the feature-width experiment (cifar_pca_dims_sweep.py): does the baseline certificate
change with ``d`` because of ``d`` itself, or because of a different pre-training radius?

Hypothesis. H6b says the unrefined certificate loosens as ``d`` grows. The feature-width experiment
uses each width's own selected hyperparameters. Selection picked ``pt_epsilon`` = 0.01 at d=20 but
0.02 at d=22 and d=24, with everything else identical. Going from 0.01 to 0.02 shrinks the
certified box width by 22-24% at those widths, more than ``d`` itself moves it, so that comparison
mixes ``d`` with ``pt_epsilon``. This control fixes ``pt_epsilon`` at d=20's value. If H6b holds:
    - the baseline box width at d=22 and d=24 with ``pt_epsilon`` = 0.01 is still wider than at d=20;
    - the bisect-all relative gain stays within about 1 point of d=20's (H6a).

Result (Isambard jobs 6668781, 6668905, 6668917; scripts/isambard/README.md §2.1). Both hold. At
``pt_epsilon`` = 0.01 the baseline box width is 1.03x d=20's at d=22 and 1.14x at d=24. The bisect-all
gains are 6.64%, 6.59% and 6.21% at d = 20, 22 and 24. Nominal accuracy does not rise with ``d``
(0.795, 0.802, 0.791, within sampling noise). Every run reported violation 0.

Mechanism. This is the same run as cifar_pca_run.py at the feature-width attack (k=100, eps=0.05 by
default), with the hyperparameters selected for ``d`` except ``pt_epsilon``, which comes from the
command line. The matching pre-trained model is already cached from the selection grid. The runs are
cached under their own key, because the pre-training radius is part of it, so they never collide with
the main experiments' runs. The d=20 comparison point is the feature-width experiment's d=20 row,
which already uses 0.01.

Output: rank 0 prints one JSON line per run with the run record and the val and test metrics, the
same shape as cifar_pca_run.py. No aggregation script reads these yet.

Jobs: scripts/isambard/pt_control_cheap.sbatch (unrefined, top-12 and eps/2 runs) and
pt_control_sharded.sbatch (bisect-all, one job per ``d``).

Usage: [torchrun --standalone --nproc-per-node 4] cifar_pca_pt_epsilon_control.py --pca-dims 22
           --pt-epsilon 0.01 [--k 100 --eps 0.05] [--n-splits 2 --split-dims 22]
Omitting ``--split-dims`` gives the unrefined run.

Key external dependencies: torch.distributed (NCCL, for sharded refinement), and the sibling
modules cifar_pca and cifar_pca_manifest.
"""

import argparse
import dataclasses
import json
import sys

import torch
import torch.distributed as dist

import cifar_pca
from cifar_pca_manifest import E2_EPS, E2_K

parser = argparse.ArgumentParser()
# not --d or --n-dims: torchrun reads abbreviations of its own options even after the script
parser.add_argument("--pca-dims", type=int, required=True)
parser.add_argument("--pt-epsilon", type=float, required=True,
                    help="pre-training radius to use in place of the one E0 selected at this width")
parser.add_argument("--k", type=int, default=E2_K)
parser.add_argument("--eps", type=float, default=E2_EPS)
parser.add_argument("--n-splits", type=int, default=2)
parser.add_argument("--split-dims", type=int, help="coordinates to split; omit for the unrefined baseline")
args = parser.parse_args()

cifar_pca.init_distributed()
refine = None if args.split_dims is None else (args.n_splits, args.split_dims)
selected = cifar_pca.selected_hyperparameters(args.pca_dims)
hp = dataclasses.replace(selected, pt_epsilon=args.pt_epsilon)

datasets = cifar_pca.get_datasets(args.pca_dims)
model = cifar_pca.get_pretrained_model(args.pca_dims, hp)
bounded, record = cifar_pca.train_certified(
    cifar_pca.make_config(args.k, args.eps, hp, refine), args.pca_dims, model, datasets,
    cifar_pca.pretrain_id(hp, torch.float32),
)

if cifar_pca.rank() == 0:
    print(json.dumps({
        "d": args.pca_dims, "k": args.k, "eps": args.eps, "refine": refine,
        "pt_epsilon": args.pt_epsilon, "selected_pt_epsilon": selected.pt_epsilon,
        **record,
        "val": cifar_pca.certified_metrics(bounded, datasets["held_out_val"], datasets["clean_val"]),
        "test": cifar_pca.certified_metrics(bounded, datasets["held_out_test"], datasets["clean_test"]),
    }))
    if record["violation"] > cifar_pca.VIOLATION_TOLERANCE:
        print(f"WARNING: bound violation {record['violation']:.2e}; this cell is unsound", file=sys.stderr)

if dist.is_initialized():
    dist.destroy_process_group()
