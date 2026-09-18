"""
Hyperparameter selection for the CIFAR-10 feature-poisoning experiments, run once per feature width
``d``. It tests no hypothesis. It fixes the training configuration that every later run at that width
uses, so the experiments compare attacks and refinement settings on one model chosen in advance,
instead of tuning each result.

What is searched. A 324-point grid, with every point scored at one fixed attack (k=50, eps=0.01),
without refinement, on the validation splits:
    - the pre-trained model: robust-regularisation radius ``pt_epsilon``, regulariser strength and
      hidden width (36 models);
    - crossed with the fine-tuning learning rate and number of epochs (9 schedules per model).

Selection rule. A point is admissible if:
    - its nominal clean val accuracy is at least 0.75 (a model that predicts one class everywhere
      scores 1/3 or 2/3 there);
    - fine-tuning raised its nominal held-out val accuracy above the pre-trained model's;
    - no bound was violated.
Among admissible points, the one with the highest certified held-out val accuracy wins; lower certified
cross-entropy breaks ties. The test split is scored for the selected point only.

Output: the selection JSON that ``cifar_pca.selected_hyperparameters`` reads for this ``d``, plus the
top of the ranking on stderr.

Usage: python cifar_pca_selection.py [--d 20] [--shard i n]. Shards split the pre-training grid across
processes (one GPU each, via CUDA_VISIBLE_DEVICES). Every run is cached, so a final unsharded call
reads the grid back, ranks it and writes the selection.

Key external dependencies: abstract_gradient_training, and the sibling modules cifar_pca and
script_utils.
"""

import argparse
import dataclasses
import itertools
import json
import sys

import torch
import tqdm

import abstract_gradient_training as agt
from abstract_gradient_training.bounded_models import IntervalBoundedModel

import cifar_pca
import script_utils

K_POISON = 50
EPSILON = 0.01

PT_EPSILONS = [0.01, 0.02, 0.05, 0.1]
REG_STRENGTHS = [0.1, 0.3, 1.0]
HIDDEN_DIMS = [128, 256, 512]
LEARNING_RATES = [0.05, 0.1, 0.2]
N_EPOCHS = [2, 3, 5]  # two iterations per epoch

MIN_CLEAN_ACC = 0.75  # the degenerate predictors score 1/3 and 2/3 on the clean split

parser = argparse.ArgumentParser()
parser.add_argument("--d", type=int, default=20)
parser.add_argument("--shard", type=int, nargs=2, metavar=("I", "N"))
args = parser.parse_args()

pt_grid = list(itertools.product(PT_EPSILONS, REG_STRENGTHS, HIDDEN_DIMS))
if args.shard:
    pt_grid = pt_grid[args.shard[0]::args.shard[1]]
datasets = cifar_pca.get_datasets(args.d)

rows = []
for pt_epsilon, reg_strength, hidden_dim in tqdm.tqdm(pt_grid, desc=f"selection d={args.d}", file=sys.stderr):
    for learning_rate, n_epochs in itertools.product(LEARNING_RATES, N_EPOCHS):
        hp = cifar_pca.Hyperparameters(pt_epsilon, reg_strength, hidden_dim, learning_rate, n_epochs)
        model = cifar_pca.get_pretrained_model(args.d, hp)
        pretrained = IntervalBoundedModel(model)
        bounded, record = cifar_pca.train_certified(
            cifar_pca.make_config(K_POISON, EPSILON, hp), args.d, model, datasets, cifar_pca.pretrain_id(hp, torch.float32)
        )
        rows.append({
            "hp": hp,
            "violation": record["violation"],
            "pt_held_out_val_acc": agt.test_metrics.test_accuracy(pretrained, *datasets["held_out_val"].tensors)[1],
            "val": cifar_pca.certified_metrics(bounded, datasets["held_out_val"], datasets["clean_val"]),
            "bounded": bounded,
        })

if args.shard:
    sys.exit(0)


def admissible(row):
    """Not degenerate, actually moved by fine-tuning, and sound."""
    return (row["val"]["nominal_clean_acc"] >= MIN_CLEAN_ACC
            and row["val"]["nominal_acc"] > row["pt_held_out_val_acc"]
            and row["violation"] <= cifar_pca.VIOLATION_TOLERANCE)


ranked = sorted(filter(admissible, rows), key=lambda r: (-r["val"]["cert_acc"], r["val"]["cert_ce"]))
if not ranked:
    sys.exit(f"no admissible point at d={args.d}")

print(f"\nCIFAR-10 PCA selection | d={args.d} k={K_POISON} eps={EPSILON} | {len(rows)} runs, {len(ranked)} "
      f"admissible | validation metrics", file=sys.stderr)
print(f"  {'pt_eps':>6s} {'reg':>4s} {'H':>4s} {'lr':>5s} {'ep':>3s}  {'pt held':>7s}  {'nom held':>8s} "
      f"{'nom clean':>9s}  {'cert held':>9s} {'cert clean':>10s} {'cert CE':>8s} {'box':>10s}", file=sys.stderr)
for r in ranked[:15]:
    hp, val = r["hp"], r["val"]
    print(f"  {hp.pt_epsilon:>6} {hp.reg_strength:>4} {hp.hidden_dim:>4d} {hp.learning_rate:>5} {hp.n_epochs:>3d}  "
          f"{r['pt_held_out_val_acc']:7.4f}  {val['nominal_acc']:8.4f} {val['nominal_clean_acc']:9.4f}  "
          f"{val['cert_acc']:9.4f} {val['cert_clean_acc']:10.4f} {val['cert_ce']:8.4f} {val['box_width']:10.3e}",
          file=sys.stderr)

# the test split is read once, for the selected point only
best = ranked[0]
test = cifar_pca.certified_metrics(best["bounded"], datasets["held_out_test"], datasets["clean_test"])
print(f"\n  selected: {best['hp']}\n  test: certified held-out {test['cert_acc']:.4f} (nominal "
      f"{test['nominal_acc']:.4f}), certified clean {test['cert_clean_acc']:.4f} (nominal "
      f"{test['nominal_clean_acc']:.4f})", file=sys.stderr)

def write_selection(tmp):
    with open(tmp, "w") as file:
        json.dump({
            "hyperparameters": dataclasses.asdict(best["hp"]),
            "threat_model": {"k_poison": K_POISON, "epsilon": EPSILON},
            "val": best["val"],
            "test": test,
            "n_runs": len(rows),
            "n_admissible": len(ranked),
        }, file, indent=1)


script_utils.atomic_write(cifar_pca.selection_path(args.d), write_selection)
print(f"  written: {cifar_pca.selection_path(args.d)}", file=sys.stderr)
