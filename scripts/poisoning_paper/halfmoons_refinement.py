"""
Shared module for the half-moons refinement experiment: its fixed configuration and one cached run of
certified training. halfmoons_run.py launches runs through it, and halfmoons_refinement_sweep.py
reads them back.

Task. Binary classification on 2-D half-moons with quadratic and cubic features appended (6 inputs),
as in halfmoons.py, with a ``6 -> 128 ReLU -> 2`` network trained in float64 from a fixed
initialisation with cross-entropy loss.

Threat model and certificate. Up to ``K_POISON`` = 200 samples of the 3000-sample training batch may
have their input features moved anywhere within an l_inf ball of radius ``EPSILON`` = 0.01. Abstract
Gradient Training (AGT) returns a parameter box that contains every model any such attack could
produce. Reported over the box: worst-case test accuracy on 500 held-out points, and box width (the
sum of per-parameter interval widths; smaller means tighter). The nominal model does not depend on the
attack or on refinement.

Refinement. Every one of the 6 features is cut into ``n_splits`` equal pieces, giving
``n_splits ** 6`` leaves, which are bounded separately and then combined. Every coordinate is split
at the first rung, so the only lever is ``n_splits``.

Caching and distribution. A run is cached under its configuration. Under torchrun the leaves are
sharded across GPUs with the same result, rank 0 decides whether the cache is hit, and rank 0 writes
the record ``{violation, seconds, world_size, peak_gib}``.

Key external dependencies: scikit-learn (make_moons), torch.distributed (NCCL, for sharded
refinement), abstract_gradient_training, and the sibling modules octmnist_pca (count_violations) and
script_utils.
"""

import copy
import functools
import json
import os
import time

import numpy as np
import sklearn.datasets
import sklearn.model_selection
import torch
import torch.distributed as dist
import torch.utils.data

import abstract_gradient_training as agt
from abstract_gradient_training import AGTConfig, test_metrics
from abstract_gradient_training.bounded_models import IntervalBoundedModel

import octmnist_pca
import script_utils

USE_CACHED = True  # whether to reuse cached parameter boxes, keyed on config.hash()

SEED = 1234
HIDDEN_DIM = 128
BATCHSIZE = 3000
TEST_BATCHSIZE = 500
EPSILON = 0.01  # fixed l_inf feature-poisoning radius; halfmoons.py frame 1 uses 0.01
K_POISON = 200  # fixed attack size, from the halfmoons.py frame 1 sweep [50, 100, 200, 300]

STRATEGY = "sensitivity"
SPLIT_DIMS = 6  # every feature
MAX_LEAVES = 20_000_000  # raise the InputRefinementConfig cost guard for the larger rungs (16^6 = 16.8M)
LEAF_CHUNK = 192  # leaves stacked per bounding call; 192 peaks ~22 GB here (256 OOMs a 32 GB card)

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"

BASE_CONFIG = AGTConfig(
    learning_rate=2.0,
    n_epochs=4,
    device=DEVICE,
    loss="cross_entropy",
    lr_decay=0.6,
    lr_min=1e-3,
    log_level="WARNING",
    paired_poison=False,
    label_k_poison=0,
    epsilon=EPSILON,
)


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


def get_dataloaders(train_batchsize=BATCHSIZE, test_batchsize=TEST_BATCHSIZE, random_state=0, noise=0.1, sep=0.2):
    """
    Half-moons with quadratic and cubic features appended, as in halfmoons.py.
    Returns (train, test) dataloaders over 6-feature float64 inputs.
    """
    x, y = sklearn.datasets.make_moons(
        noise=noise, random_state=random_state, n_samples=train_batchsize + test_batchsize
    )
    x[y == 0, 1] += sep
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=test_batchsize / (train_batchsize + test_batchsize), random_state=42
    )
    x_train = np.hstack((x_train, x_train**2, x_train**3))
    x_test = np.hstack((x_test, x_test**2, x_test**3))
    dl_train = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x_train).double(), torch.from_numpy(y_train)),
        batch_size=train_batchsize,
        shuffle=False,
    )
    dl_test = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x_test).double(), torch.from_numpy(y_test)),
        batch_size=test_batchsize,
        shuffle=False,
    )
    return dl_train, dl_test


@functools.cache
def _data():
    torch.manual_seed(SEED)
    return get_dataloaders()


def get_model():
    """The fixed float64 initialisation every run starts from."""
    torch.manual_seed(SEED)
    return torch.nn.Sequential(
        torch.nn.Linear(6, HIDDEN_DIM),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, 2),
    ).double()


def make_config(n_splits):
    """BASE_CONFIG at the fixed attack size, refined across all SPLIT_DIMS features when n_splits is
    given (None for the unrefined baseline). Leaves are sharded across ranks whenever the run is
    launched with more than one process."""
    config = copy.deepcopy(BASE_CONFIG)
    config.k_poison = K_POISON
    if n_splits is not None:
        config.input_refinement = agt.InputRefinementConfig(
            n_splits=n_splits, n_dims=SPLIT_DIMS, strategy=STRATEGY, max_leaves=MAX_LEAVES,
            leaf_chunk=LEAF_CHUNK, shard_leaves=world_size() > 1,
        )
    return config


def certified_metrics(bounded_model):
    """Certified parameter-box width and (worst, nominal, best) test accuracy over that box."""
    x_test, y_test = _data()[1].dataset.tensors
    worst, nominal, best = test_metrics.test_accuracy(bounded_model, x_test, y_test)
    box_width = sum((u - l).sum().item() for l, u in zip(bounded_model.param_l, bounded_model.param_u))
    return {"cert_acc_worst": worst, "nominal_acc": nominal, "cert_acc_best": best, "box_width": box_width}


def _cache_path(config):
    results_dir, _, _, _ = script_utils.make_dirs()
    return f"{results_dir}/halfmoons_refinement_{SEED}_{HIDDEN_DIM}_float64_{config.hash()}"


def run(n_splits, require_cached=False):
    """
    Certified training from the fixed initialisation, refined across all SPLIT_DIMS features when
    n_splits is given (None for the unrefined baseline). Cached on config.hash(); under torchrun rank 0
    decides whether the cache is hit and writes it.

    require_cached raises FileNotFoundError instead of training, so an aggregation script can never
    start a refinement run by accident.

    Returns:
        tuple[IntervalBoundedModel, dict]: the bounded model, and the run record {"violation",
            "seconds", "world_size", "peak_gib"}.
    """
    config = make_config(n_splits)
    fname = _cache_path(config)
    bounded_model = IntervalBoundedModel(get_model())

    cached = torch.tensor(int(os.path.isfile(fname) and USE_CACHED))
    if dist.is_initialized():
        cached = cached.to(DEVICE)
        dist.broadcast(cached, src=0)
    if cached.item():
        bounded_model.load_params(fname)
        with open(f"{fname}.json") as file:
            return bounded_model, json.load(file)
    if require_cached:
        raise FileNotFoundError(f"half-moons refinement run not cached (n_splits={n_splits}): {fname}")

    dl_train, dl_test = _data()
    if torch.cuda.is_available():
        torch.cuda.init()  # the model is still on the CPU here; AGT moves it to config.device
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
