"""
Shared module for the OCT-MNIST refinement-ladder experiment: the run list and one cached certified
run. octmnist_pca_ladder_run.py launches runs through it, and octmnist_pca_ladder_sweep.py reads them
back. See octmnist_pca for the task, threat model and metrics.

Why this experiment. The refinement-depth sweep at d=16 (octmnist_pca_leaf_sweep.py, k=50, eps=0.1)
moved certified accuracy by 1-3 of 250 test images. The threat grid (octmnist_pca_threat_sweep.py,
d=15) found the largest refinement gains at large attacks: +0.040 at k=200 and +0.052 at k=400, both
at eps=0.1 with all 15 features bisected. This experiment builds a full ladder of partitions at those
two cells.

Run list. At d=15, eps=0.1 and each k in LADDER_KS: the unrefined run, and a run at every
``(n_splits, n_dims)`` rung of LADDER_SCHEDULE. That is a bisection ladder up to all 15 features, then
finer cuts: 3^10 against 2^15 at similar cost, 3^12 and 4^10, and 3^15, which cuts every feature into
three pieces.

Cache compatibility with the threat grid. Configurations are built exactly as
``octmnist_pca_threat_sweep.make_config`` builds them, with the same cache tag. ``max_leaves`` is
set to each rung's leaf count, as the threat grid does for 2^15. ``leaf_chunk`` and ``max_leaves``
are both in ``AGTConfig.hash()``, so the unrefined and 2^15 runs are cache hits on the threat grid's
results.

Distribution. Rungs of at least SHARDED_FROM_LEAVES leaves are meant for a 4-GPU node under torchrun,
where the leaves are sharded across ranks with the same result. The cached run record stores the
world size, so the figures can mark which rungs used data parallelism.

Key external dependencies: torch.distributed (NCCL, for sharded refinement), abstract_gradient_training,
and the sibling module octmnist_pca.
"""

import argparse
import copy
import json
import os

import abstract_gradient_training as agt

import octmnist_pca

PCA_DIMS = 15  # the threat grid's width, so its cached runs are reused
EPSILON = 0.1
LADDER_KS = [200, 400]
LADDER_SCHEDULE = [(2, 5), (2, 10), (2, 15), (3, 10), (3, 12), (4, 10), (3, 15)]
STRATEGY = "sensitivity"
LEAF_CHUNK = 16  # the threat grid's chunk; part of the cache key
TAG = "threat"  # the threat grid's cache tag
SHARDED_FROM_LEAVES = 2**16  # rungs with at least this many leaves take a 4-GPU node
VIOLATION_TOLERANCE = 1e-3  # above this a run's certificate is unsound


def leaves(refine):
    return 1 if refine is None else refine[0] ** refine[1]


def make_config(k_poison, refine):
    """Tuned config at (k_poison, EPSILON), refined along the ``refine = (n_splits, n_dims)`` most
    sensitive features (None for the unrefined baseline). Leaves are sharded across ranks whenever the
    run is launched with more than one process."""
    config = copy.deepcopy(octmnist_pca.FINETUNE_CONFIG)
    config.k_poison = k_poison
    config.epsilon = EPSILON
    if refine is not None:
        n_splits, n_dims = refine
        config.input_refinement = agt.InputRefinementConfig(
            n_splits=n_splits, n_dims=n_dims, strategy=STRATEGY, max_leaves=leaves(refine),
            leaf_chunk=LEAF_CHUNK, shard_leaves=octmnist_pca.world_size() > 1,
        )
    return config


def cache_path(k_poison, refine):
    return octmnist_pca.cache_path(make_config(k_poison, refine), PCA_DIMS, TAG)


def cost_record(k_poison, refine):
    """The ``{"seconds", "world_size", "peak_gib"}`` record of a cached run, or None if it has none."""
    path = f"{cache_path(k_poison, refine)}.json"
    if not os.path.isfile(path):
        return None
    with open(path) as file:
        return json.load(file)


def all_runs():
    """Every ``(k, refine)`` run of the ladder, cheapest first."""
    runs = [(k, refine) for k in LADDER_KS for refine in [None] + LADDER_SCHEDULE]
    return sorted(runs, key=lambda run: (leaves(run[1]), run[0]))


def run(k_poison, refine, model, drusen_train, clean_train):
    """One certified run through ``octmnist_pca.run_certified`` (cached). Returns ``(bounded_model,
    violation)``."""
    return octmnist_pca.run_certified(make_config(k_poison, refine), PCA_DIMS, model, drusen_train, clean_train, tag=TAG)


def certified_metrics(bounded_model, drusen_test, clean_test):
    """Certified box width and, over the box, worst-case and nominal Drusen accuracy and cross-entropy,
    and clean-split accuracy, as octmnist_pca_threat_sweep.py reports them."""
    acc_w, acc_n, _ = agt.test_metrics.test_accuracy(bounded_model, *drusen_test.tensors, epsilon=0)
    ce_w, ce_n, _ = agt.test_metrics.test_cross_entropy(bounded_model, *drusen_test.tensors, epsilon=0)
    clean_w, clean_n, _ = agt.test_metrics.test_accuracy(bounded_model, *clean_test.tensors, epsilon=0)
    box_width = sum((u - l).sum().item() for l, u in zip(bounded_model.param_l, bounded_model.param_u))
    return {
        "cert_acc": acc_w, "nominal_acc": acc_n, "cert_ce": ce_w, "nominal_ce": ce_n,
        "cert_clean_acc": clean_w, "nominal_clean_acc": clean_n, "box_width": box_width,
    }


def run_args(k_poison, refine):
    args = f"--k {k_poison}"
    return args if refine is None else f"{args} --n-splits {refine[0]} --split-dims {refine[1]}"


if __name__ == "__main__":
    # Prints runs as octmnist_pca_ladder_run.py argument lines, cheapest first, for the Slurm jobs:
    #   --gpus 1   only runs with fewer than SHARDED_FROM_LEAVES leaves; --gpus 4 only the others
    #   --missing  only runs with no cached result
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, choices=(1, 4))
    parser.add_argument("--missing", action="store_true")
    args = parser.parse_args()
    for k, refine in all_runs():
        if args.gpus is not None and (args.gpus == 4) != (leaves(refine) >= SHARDED_FROM_LEAVES):
            continue
        if args.missing and os.path.isfile(cache_path(k, refine)):
            continue
        print(run_args(k, refine))
