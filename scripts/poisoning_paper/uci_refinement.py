"""
Shared module for the UCI house-electric refinement experiment: its fixed configuration and one
cached run of certified training. uci_run.py launches runs through it, and
uci_refinement_sweep.py reads them back.

Task. Regression on the UCI house-electric dataset (11 continuous input features), with a one-layer
MLP (64 hidden units) trained from a fixed initialisation with MSE loss. Training stops after 150
iterations of batch 10000.

Threat model and certificate. Up to ``K_POISON`` = 200 samples per training batch may have their
input features moved anywhere within an l_inf ball of radius ``EPSILON`` = 0.01. Abstract Gradient
Training (AGT) returns a parameter box that contains every model any such attack could produce.
Reported over the box: worst-case test MSE, and box width (the sum of per-parameter interval widths;
smaller means tighter). The nominal model does not depend on the attack or on refinement.

Refinement. Every one of the 11 features is cut into ``n_splits`` equal pieces, giving
``n_splits ** 11`` leaves, which are bounded separately and then combined. Because every coordinate
is already split, the only lever is ``n_splits``: 2, 3 and 4 give 2048, 177147 and 4194304 leaves.
Interval products use Rump's method rather than the exact product, which would not fit in memory at
these leaf counts.

Caching and distribution. A run is cached under its configuration. Under torchrun the leaves are
sharded across GPUs with the same result, rank 0 decides whether the cache is hit, and rank 0 writes
the record ``{violation, seconds, world_size, peak_gib}``.

Key external dependencies: torch.distributed (NCCL, for sharded refinement), abstract_gradient_training,
and the sibling modules train_uci, octmnist_pca (count_violations) and script_utils.
"""

import copy
import functools
import itertools
import json
import os
import time

import torch
import torch.distributed as dist

import abstract_gradient_training as agt
from abstract_gradient_training.bounded_models import IntervalBoundedModel

import octmnist_pca
import script_utils
import train_uci

USE_CACHED = True  # whether to reuse cached parameter boxes, keyed on config.hash()

SEED = 15
BATCHSIZE = 10000
HIDDEN_LAY = 1
HIDDEN_SIZE = 64
MAX_ITERS = 150  # early-stopping cap, as in train_uci.get_training_bounds

EPSILON = 0.01  # fixed l_inf feature-poisoning radius; matches notebooks/UCI - Poisoning.ipynb
K_POISON = 200  # fixed attack size; matches notebooks/UCI - Poisoning.ipynb

# Priority: split every input dimension first, then raise the number of splits per dimension.
# house-electric has 11 features, so every rung splits all 11 and only n_splits grows: leaves =
# n_splits ** 11 -> 2048, 177147, 4194304. Uniform splits make the rungs coarse; (3, 11) is ~10^5
# leaves and takes hours, (4, 11) is impractical without sharding.
STRATEGY = "sensitivity"
SPLIT_DIMS = 11
MAX_LEAVES = 10_000_000  # raise the InputRefinementConfig cost guard for the larger rungs
LEAF_CHUNK = 192  # leaves stacked per bounding call; 192 peaks ~27 GB here (256 OOMs a 32 GB card).
#                   Wall time is Python-loop-bound at batch 10k, so a larger chunk barely changes it
#                   -- it is the memory ceiling, not a speed knob, for this problem.
INTERVAL_MATMUL = "rump"  # exact materialises a [chunk*batch, in, hidden] tensor and OOMs at large leaf_chunk

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"


def init_distributed():
    """Join the torchrun process group when launched with more than one process. Returns the world size."""
    n_processes = int(os.environ.get("WORLD_SIZE", 1))
    if n_processes > 1 and not dist.is_initialized():
        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group("nccl", device_id=torch.device(DEVICE))
    return n_processes


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def make_config(n_splits):
    """NOMINAL_CONFIG at the fixed attack size, refined across all SPLIT_DIMS coordinates when
    n_splits is given (None for the unrefined baseline). Leaves are sharded across ranks whenever
    the run is launched with more than one process."""
    config = copy.deepcopy(train_uci.NOMINAL_CONFIG)
    config.device = DEVICE
    config.log_level = "WARNING"
    config.k_poison = K_POISON
    config.epsilon = EPSILON
    if n_splits is not None:
        config.input_refinement = agt.InputRefinementConfig(
            n_splits=n_splits, n_dims=SPLIT_DIMS, strategy=STRATEGY, max_leaves=MAX_LEAVES,
            leaf_chunk=LEAF_CHUNK, shard_leaves=world_size() > 1,
        )
    return config


@functools.cache
def _test_split():
    """The fixed (test_point, test_label) batch certified_metrics reports against."""
    torch.manual_seed(SEED)
    _, dl_test = train_uci.get_dataset(BATCHSIZE)
    return next(iter(dl_test))


def certified_metrics(bounded_model):
    """Certified parameter-box width and (worst, nominal, best) test MSE over that box."""
    test_point, test_label = _test_split()
    worst, nominal, best = agt.test_metrics.test_mse(bounded_model, test_point, test_label)
    box_width = sum((u - l).sum().item() for l, u in zip(bounded_model.param_l, bounded_model.param_u))
    return {
        "mse_worst": worst, "mse_nominal": nominal, "mse_best": best,
        "mse_width": abs(worst - best), "box_width": box_width,
    }


def _cache_path(config):
    results_dir, _, _, _ = script_utils.make_dirs()
    return f"{results_dir}/uci_refinement_{SEED}_{HIDDEN_LAY}_{HIDDEN_SIZE}_{INTERVAL_MATMUL}_{config.hash()}"


def run(n_splits, require_cached=False):
    """
    Certified training from a fresh model (train_uci.get_model, same seed, interval_matmul and
    MAX_ITERS early stop as train_uci.get_training_bounds), refined across all SPLIT_DIMS coordinates
    when n_splits is given (None for the unrefined baseline). Cached on config.hash(); under torchrun
    rank 0 decides whether the cache is hit and writes it.

    require_cached raises FileNotFoundError instead of training, so an aggregation script can never
    start a refinement run by accident.

    Returns:
        tuple[IntervalBoundedModel, dict]: the bounded model, and the run record {"violation",
            "seconds", "world_size", "peak_gib"}.
    """
    config = make_config(n_splits)
    fname = _cache_path(config)

    torch.manual_seed(SEED)
    model = train_uci.get_model(HIDDEN_LAY, HIDDEN_SIZE, SEED)
    bounded_model = IntervalBoundedModel(model, interval_matmul=INTERVAL_MATMUL).to(DEVICE)

    cached = torch.tensor(int(os.path.isfile(fname) and USE_CACHED))
    if dist.is_initialized():
        cached = cached.to(DEVICE)
        dist.broadcast(cached, src=0)
    if cached.item():
        bounded_model.load_params(fname)
        with open(f"{fname}.json") as file:
            return bounded_model, json.load(file)
    if require_cached:
        raise FileNotFoundError(f"UCI refinement run not cached: {fname}")

    torch.manual_seed(SEED)
    dl_train, dl_test = train_uci.get_dataset(BATCHSIZE)
    dl_train = list(itertools.islice(dl_train, MAX_ITERS))

    iter_count = 0

    def early_stop(_):
        nonlocal iter_count
        iter_count += 1
        return iter_count >= MAX_ITERS

    config.early_stopping_callback = early_stop

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
    start = time.time()
    with octmnist_pca.count_violations() as counter:
        agt.poison_certified_training(bounded_model, config, dl_train, dl_test)
    record = {
        "violation": counter.worst,
        "seconds": time.time() - start,
        "world_size": world_size(),
        "peak_gib": torch.cuda.max_memory_allocated(DEVICE) / 2**30 if torch.cuda.is_available() else None,
    }
    if rank() == 0:
        def write_record(tmp):
            with open(tmp, "w") as file:
                json.dump(record, file)
        script_utils.atomic_write(f"{fname}.json", write_record)
        script_utils.atomic_write(fname, bounded_model.save_params)
    return bounded_model, record
