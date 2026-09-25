"""
Shared module for the MAGIC gamma-telescope refinement experiment: its fixed configuration and one
cached run of certified training. magic_run.py launches runs through it, and magic_refinement_sweep.py
reads them back.

Task. Binary classification (gamma = 0, hadron = 1; 65/35) on the MAGIC gamma telescope dataset
(OpenML 1120, 10 continuous features, 19020 samples), split 80/20 stratified and standardised with the
training-split mean and standard deviation. A ``10 -> 128 ReLU -> 2`` network is trained in float64
from a fixed initialisation with cross-entropy loss, as in halfmoons_refinement, by full-batch
gradient descent (one step per epoch over the 15216 training samples).

Threat model and certificate. The half-moons threat model: up to ``K_POISON`` = 200 training samples
may have their input features moved anywhere within an l_inf ball of radius ``EPSILON`` = 0.01 (in
standardised units); labels are clean. Abstract Gradient Training (AGT) returns a parameter box that
contains every model any such attack could produce. Reported over the box: worst-case test accuracy on
the 3804 held-out points, and box width (the sum of per-parameter interval widths; smaller means
tighter). The nominal model does not depend on the attack or on refinement. The majority-class test
accuracy, 0.648, is the floor a certificate must beat to mean anything.

Training schedule. Three full-batch steps at learning rate 1.0 (decay 0.6). Bounds compound with the
number of steps: at k=200 the unrefined certificate falls from 0.73 after two steps to 0.60 after
three and 0.10 after four, and 3000-sample minibatches (20 steps) certify nothing. Three steps leave
the unrefined certificate below the majority baseline, so refinement has room to show a gain.

Refinement. A rung is one or two tiers ``N^D``: the D most sensitive features are each cut into N
equal pieces, and a second tier ``M^E`` cuts the next E features into M (written ``N^DxM^E``). The
leaves, ``N^D * M^E`` of them, are bounded separately and then combined. The ladder alternates
between cutting the more sensitive half of the features one step finer and catching the other half up:
2^5, 2^10, 3^5x2^5, 3^10, 4^5x3^5, ... (see magic_refinement_sweep).

Caching and distribution. A run is cached under its configuration. Under torchrun the leaves are
sharded across GPUs with the same result, rank 0 decides whether the cache is hit, and rank 0 writes
the record ``{violation, seconds, world_size, peak_gib}``.

Key external dependencies: scikit-learn (fetch_openml, cached under the .data directory; the first call
needs network access), torch.distributed (NCCL, for sharded refinement), abstract_gradient_training,
and the sibling modules halfmoons_refinement (init_distributed, rank, world_size), octmnist_pca
(count_violations) and script_utils.
"""

import copy
import functools
import json
import math
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
from halfmoons_refinement import DEVICE, init_distributed, rank, world_size  # noqa: F401 (re-exported)

USE_CACHED = True  # whether to reuse cached parameter boxes, keyed on config.hash()

SEED = 1234
HIDDEN_DIM = 128
N_FEATURES = 10
TEST_FRACTION = 0.2
EPSILON = 0.01  # l_inf feature-poisoning radius, as in halfmoons_refinement
K_POISON = 200  # attack size, as in halfmoons_refinement

STRATEGY = "sensitivity"
MAX_LEAVES = 30_000_000  # raise the InputRefinementConfig cost guard for the larger rungs (6^5x5^5 = 24.3M)
FRAGSIZE = 3000  # rows per bounding call; excluded from config.hash(): it changes memory, not the result
LEAF_CHUNK = 96  # leaves stacked per bounding call; 192 OOMs a 32 GB card (the first-layer gradient
#                  interval is [chunk * FRAGSIZE, 10, 128] float64, 5.5 GiB per tensor at 192)

BASE_CONFIG = AGTConfig(
    learning_rate=1.0,
    n_epochs=3,
    device=DEVICE,
    loss="cross_entropy",
    lr_decay=0.6,
    lr_min=1e-3,
    fragsize=FRAGSIZE,
    log_level="WARNING",
    paired_poison=False,
    label_k_poison=0,
    epsilon=EPSILON,
)


def get_datasets():
    """
    MAGIC gamma telescope, split 80/20 stratified and standardised on the training split. Downloads from
    OpenML on first use and caches under the .data directory.

    Returns:
        tuple[TensorDataset, TensorDataset]: (train, test) over float64 features and int64 labels.
    """
    _, _, data_dir, _ = script_utils.make_dirs()
    bunch = sklearn.datasets.fetch_openml("MagicTelescope", version=1, data_home=data_dir, as_frame=False)
    x = bunch.data.astype(np.float64)
    y = (bunch.target == "h").astype(np.int64)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=TEST_FRACTION, random_state=42, stratify=y
    )
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    return (
        torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
    )


@functools.cache
def _data():
    """Full-batch (train, test) dataloaders."""
    train, test = get_datasets()
    return (
        torch.utils.data.DataLoader(train, batch_size=len(train), shuffle=False),
        torch.utils.data.DataLoader(test, batch_size=len(test), shuffle=False),
    )


def get_model():
    """The fixed float64 initialisation every run starts from."""
    torch.manual_seed(SEED)
    return torch.nn.Sequential(
        torch.nn.Linear(N_FEATURES, HIDDEN_DIM),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, 2),
    ).double()


def parse_rung(text):
    """'N^D' or 'N^DxM^E' -> ((N, D),) or ((N, D), (M, E)): the refinement tiers, most sensitive first."""
    tiers = tuple(tuple(int(v) for v in tier.split("^")) for tier in text.split("x"))
    if not 1 <= len(tiers) <= 2 or any(len(tier) != 2 for tier in tiers):
        raise ValueError(f"rung {text!r} is not of the form N^D or N^DxM^E")
    return tiers


def rung_label(refine):
    """The inverse of parse_rung; 'unrefined' for None."""
    return "unrefined" if refine is None else "x".join(f"{n}^{d}" for n, d in refine)


def n_leaves(refine):
    return 1 if refine is None else math.prod(n**d for n, d in refine)


def make_config(refine):
    """BASE_CONFIG at the fixed attack size, refined by the tiers ``refine`` (see parse_rung; None for
    the unrefined baseline). Leaves are sharded across ranks whenever the run is launched with more
    than one process."""
    config = copy.deepcopy(BASE_CONFIG)
    config.k_poison = K_POISON
    if refine is not None:
        (n_splits, split_dims), *secondary = refine
        secondary_n_splits, secondary_n_dims = secondary[0] if secondary else (2, 0)
        config.input_refinement = agt.InputRefinementConfig(
            n_splits=n_splits, n_dims=split_dims, secondary_n_splits=secondary_n_splits,
            secondary_n_dims=secondary_n_dims, strategy=STRATEGY, max_leaves=MAX_LEAVES,
            leaf_chunk=LEAF_CHUNK, shard_leaves=world_size() > 1,
        )
    return config


def majority_accuracy():
    """Test accuracy of always predicting the training-split majority class."""
    train, test = get_datasets()
    majority = train.tensors[1].mode().values
    return (test.tensors[1] == majority).double().mean().item()


def certified_metrics(bounded_model):
    """Certified parameter-box width and (worst, nominal, best) test accuracy over that box."""
    x_test, y_test = _data()[1].dataset.tensors
    worst, nominal, best = test_metrics.test_accuracy(bounded_model, x_test, y_test)
    box_width = sum((u - l).sum().item() for l, u in zip(bounded_model.param_l, bounded_model.param_u))
    return {"cert_acc_worst": worst, "nominal_acc": nominal, "cert_acc_best": best, "box_width": box_width}


def _cache_path(config):
    results_dir, _, _, _ = script_utils.make_dirs()
    return f"{results_dir}/magic_refinement_{SEED}_{HIDDEN_DIM}_float64_{config.hash()}"


def run(refine, require_cached=False):
    """
    Certified training from the fixed initialisation, refined by the tiers ``refine`` (see
    parse_rung; None for the unrefined baseline). Cached on config.hash(); under torchrun rank 0 decides whether
    the cache is hit and writes it.

    require_cached raises FileNotFoundError instead of training, so an aggregation script can never
    start a refinement run by accident.

    Returns:
        tuple[IntervalBoundedModel, dict]: the bounded model, and the run record {"violation",
            "seconds", "world_size", "peak_gib"}.
    """
    config = make_config(refine)
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
        raise FileNotFoundError(f"MAGIC refinement run not cached ({rung_label(refine)}): {fname}")

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
