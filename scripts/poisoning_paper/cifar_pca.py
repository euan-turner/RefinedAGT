"""
Shared pipeline for the CIFAR-10 feature-poisoning experiments: data splits, the PCA feature map, the
pre-trained model, and one cached run of certified training. Every other ``cifar_pca_*`` script builds
on it, and the terms defined here are used throughout them.

Task. CIFAR-10 relabelled as binary: vehicle (airplane, automobile, ship, truck) = 1, animal = 0.
    - Held-out data: the automobile images. A model pre-trained on the other nine classes does worst on
      them, so fine-tuning has real work to do. They are the only data the attacker can touch.
    - Clean data: the other nine classes, trusted.
    Each side is split into train, val and test. Val is used only to select hyperparameters, test only
    for reported numbers.

Features. Pixels (3072 values) are projected onto the leading ``d`` principal components, fit once on
the clean training images, and each component is min-max scaled to [0, 1] on that split. The
projection is fixed and trusted. ``d`` is kept small (20-24) so the input can be partitioned one
coordinate at a time (see Refinement).

Model and training. A dense ``d -> H ReLU -> 1`` network is pre-trained on the clean data with a robust
regulariser of radius ``pt_epsilon``. It is then fine-tuned with Abstract Gradient Training (AGT) on
batches of 2000 held-out and 3000 clean samples. ``H``, ``pt_epsilon`` and the fine-tuning schedule
are selected per ``d`` by cifar_pca_selection.py.

Threat model. In every held-out batch, up to ``k`` samples may have their feature vector moved
anywhere within an l_inf ball of radius ``eps``, chosen with full knowledge and adaptively.

Certificate. AGT carries the attack through training and returns a parameter box that contains every
model any such attack could produce. Over that box it reports:
    - certified held-out accuracy and cross-entropy (worst case over the box);
    - certified clean accuracy, which exposes a model that has collapsed onto one class;
    - box width, the sum of the per-parameter interval widths. Smaller means a tighter certificate.
The nominal model (training with no attack) does not depend on ``k``, ``eps`` or refinement.

Refinement. Optionally, each poisoned sample's eps-ball is partitioned. ``(n_splits, n_dims)`` cuts
the ``n_dims`` most sensitive feature coordinates (ranked by the nominal first-layer weights) into
``n_splits`` pieces each. That gives ``n_splits ** n_dims`` sub-boxes ("leaves"), which are bounded
separately and then combined. "Bisecting" means ``n_splits = 2``. More leaves give a tighter
certificate at proportional compute cost. Under torchrun the leaves are sharded across GPUs, which
leaves the result unchanged.

Caching and soundness. Every certified run is cached on disk under its full configuration, so the
experiment scripts can be rerun cheaply and the aggregation scripts never train. Each run records its
worst floating-point violation of the interval bounds. A float32 run above ``VIOLATION_TOLERANCE`` is
unsound, and its float64 rerun is reported in its place.

Key external dependencies: torchvision (CIFAR-10), scikit-learn (PCA), torch.distributed (sharded
refinement), and the sibling modules octmnist_pca, robust_regularization and script_utils.
"""

import copy
import dataclasses
import functools
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
import torchvision
import tqdm

import sklearn.decomposition

import abstract_gradient_training as agt
from abstract_gradient_training import AGTConfig
from abstract_gradient_training.bounded_models import IntervalBoundedModel

import octmnist_pca
import robust_regularization
import script_utils

USE_CACHED = True  # whether to reuse cached PCA bases, pre-trained models and parameter boxes
SEED = 1

VEHICLE_CLASSES = (0, 1, 8, 9)  # airplane, automobile, ship, truck -> label 1; the animals -> 0
HELD_OUT_CLASS = 1  # automobile: the class a model pre-trained on the other nine does worst on (0.52 at d=20)
CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

HELD_OUT_VAL_SIZE = 1000  # of the 5000 held-out training images
CLEAN_VAL_SIZE = 4500  # of the 45000 clean training images
SPLIT_TAG = f"val{HELD_OUT_VAL_SIZE}-{CLEAN_VAL_SIZE}"  # in every cache key, so pre-split pilot runs are never reused

PCA_MAX_DIMS = 32
PCA_SVD_SOLVER = "covariance_eigh"  # exact, so the leading d components of the fit are the d-component fit

HELD_OUT_BATCHSIZE = 2000  # incomplete batches are dropped: 4000 training images give two full batches
CLEAN_BATCHSIZE = 3000
FRAGSIZE = 2000

PT_BATCHSIZE = 100
PT_N_EPOCHS = 10
PT_LEARNING_RATE = 0.001
PT_MODEL_EPSILON = 0.001

# Part of the config hash, so fixed: a different value is a different cache key. 32 fits a 32 GB GPU at
# H=512, d=20 (11 GiB peak); larger chunks are no faster, the refinement being memory-bandwidth bound.
LEAF_CHUNK = 32
MAX_LEAVES = 2**24  # bisecting all of d=24

VIOLATION_TOLERANCE = 1e-3  # above this a run is unsound and is only reported from its float64 rerun

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"


@dataclasses.dataclass(frozen=True)
class Hyperparameters:
    """A pre-training and fine-tuning point of the selection grid (cifar_pca_selection.py)."""

    pt_epsilon: float
    reg_strength: float
    hidden_dim: int
    learning_rate: float
    n_epochs: int


def init_distributed():
    """Join the torchrun process group when launched with more than one process. Returns the world size."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group("nccl", device_id=torch.device(DEVICE))
    return world_size


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def dirs():
    """``(results, models, data, figures)``, delegated to ``script_utils.make_dirs()`` so CIFAR,
    OCT-MNIST, UCI and half-moons share one root."""
    return script_utils.make_dirs()


@functools.cache
def _load_pixels(train):
    """A CIFAR-10 split as float [N, 3072] pixels in [0, 1] and int64 class indices."""
    split = torchvision.datasets.CIFAR10(root=dirs()[2], train=train, download=True)
    images = torch.tensor(split.data, dtype=torch.float32).flatten(1) / 255
    return images, torch.tensor(split.targets, dtype=torch.int64)


@functools.cache
def pixel_splits():
    """
    The six pixel-space splits, ``{held_out,clean}_{train,val,test}``, each an ``(images, labels)`` pair
    of [N, 3072] pixels and [N, 1] binary labels (vehicle = 1). The val splits are a fixed, seeded draw
    from the training images and are disjoint from the train splits.
    """
    generator = torch.Generator().manual_seed(SEED)
    splits = {}
    for train, stage in ((True, "train"), (False, "test")):
        images, classes = _load_pixels(train)
        labels = torch.isin(classes, torch.tensor(VEHICLE_CLASSES)).to(torch.int64).unsqueeze(1)
        for name, mask, n_val in (("held_out", classes == HELD_OUT_CLASS, HELD_OUT_VAL_SIZE),
                                  ("clean", classes != HELD_OUT_CLASS, CLEAN_VAL_SIZE)):
            idx = torch.where(mask)[0]
            if train:
                idx = idx[torch.randperm(len(idx), generator=generator)]
                splits[f"{name}_val"] = (images[idx[:n_val]], labels[idx[:n_val]])
                idx = idx[n_val:]
            splits[f"{name}_{stage}"] = (images[idx], labels[idx])
    return splits


@functools.cache
def fit_pca():
    """
    The PCA basis and per-component min-max scaling, fit on the clean training split.

    Returns:
        tuple[sklearn.decomposition.PCA, np.ndarray, np.ndarray]: the projection, and the per-component
            minimum and range that scale scores into [0, 1].
    """
    cache_path = f"{dirs()[2]}/cifar_pca_{HELD_OUT_CLASS}_{SPLIT_TAG}_{PCA_MAX_DIMS}_{PCA_SVD_SOLVER}_{SEED}.npz"
    pca = sklearn.decomposition.PCA(n_components=PCA_MAX_DIMS, svd_solver=PCA_SVD_SOLVER)

    if os.path.isfile(cache_path) and USE_CACHED:
        cached = np.load(cache_path)
        pca.components_ = cached["components"]
        pca.mean_ = cached["mean"]
        pca.explained_variance_ = cached["explained_variance"]
        pca.explained_variance_ratio_ = cached["explained_variance_ratio"]
        pca.n_components_ = PCA_MAX_DIMS
        pca.n_features_in_ = cached["components"].shape[1]
        return pca, cached["score_min"], cached["score_range"]

    scores = pca.fit_transform(pixel_splits()["clean_train"][0].numpy())
    score_min = scores.min(axis=0)
    score_range = scores.max(axis=0) - score_min
    if rank() == 0:
        script_utils.atomic_write(cache_path, lambda tmp: np.savez(
            tmp, components=pca.components_, mean=pca.mean_,
            explained_variance=pca.explained_variance_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            score_min=score_min, score_range=score_range))
    return pca, score_min, score_range


def project(images, n_dims, dtype):
    """Project [N, 3072] pixels onto the leading ``n_dims`` scaled PCA features."""
    pca, score_min, score_range = fit_pca()
    scores = pca.transform(images.numpy())[:, :n_dims]
    return torch.tensor((scores - score_min[:n_dims]) / score_range[:n_dims], dtype=dtype)


def pixel_equivalent_epsilon(n_dims):
    """The largest per-component scaled-feature radius a one-grey-level (1/255) l_inf pixel perturbation
    induces, ``max_j ||v_j||_1 / (255 * range_j)``: a feature radius ``eps`` covers every pixel
    perturbation of ``eps / pixel_equivalent_epsilon(n_dims)`` grey levels."""
    pca, _, score_range = fit_pca()
    return float((np.abs(pca.components_[:n_dims]).sum(axis=1) / score_range[:n_dims]).max() / 255)


@functools.cache
def get_datasets(n_dims, dtype=torch.float32):
    """The six splits of ``pixel_splits`` as TensorDatasets over the leading ``n_dims`` scaled PCA features."""
    return {name: torch.utils.data.TensorDataset(project(images, n_dims, dtype), labels)
            for name, (images, labels) in pixel_splits().items()}


def selection_path(n_dims):
    return f"{dirs()[0]}/cifar_pca_selected_{SPLIT_TAG}_d{n_dims}.json"


def selected_hyperparameters(n_dims):
    """The point cifar_pca_selection.py chose at this feature width. Raises if selection has not run."""
    path = selection_path(n_dims)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no selection at d={n_dims}: run cifar_pca_selection.py --d {n_dims} first ({path})")
    with open(path) as file:
        return Hyperparameters(**json.load(file)["hyperparameters"])


def pretrain_id(hp, dtype):
    """Identity of a pre-trained starting point, for the certified-run cache key (``AGTConfig.hash()`` cannot see it)."""
    return (f"{SEED}_{SPLIT_TAG}_{HELD_OUT_CLASS}_{hp.hidden_dim}_{hp.pt_epsilon}_{PT_MODEL_EPSILON}_"
            f"{hp.reg_strength}_{str(dtype).split('.')[-1]}")


def get_pretrained_model(n_dims, hp, dtype=torch.float32):
    """
    ``n_dims -> hidden_dim -> 1`` dense model pre-trained on the balanced clean training split with the
    robust regularizer of ``octmnist_train.get_pretrained_model``. Trained and cached if absent; under
    torchrun only rank 0 trains, and the other ranks load its checkpoint.
    """
    model_path = (f"{dirs()[1]}/cifar_pca_{SPLIT_TAG}_held={HELD_OUT_CLASS}_{n_dims=}_hidden={hp.hidden_dim}_"
                  f"{SEED=}_eps={hp.pt_epsilon}_model_eps={PT_MODEL_EPSILON}_reg={hp.reg_strength}.ckpt")
    device = torch.device(DEVICE)
    torch.manual_seed(SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(n_dims, hp.hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hp.hidden_dim, 1)
    ).to(device)

    if not (os.path.exists(model_path) and USE_CACHED) and rank() == 0:
        _pretrain(model, n_dims, hp, device)
        script_utils.atomic_write(model_path, lambda tmp: torch.save(model.state_dict(), tmp))
    if dist.is_initialized():
        dist.barrier()
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    return model.to(dtype)


def _pretrain(model, n_dims, hp, device):
    torch.manual_seed(SEED)
    images, labels = pixel_splits()["clean_train"]
    n_per_class = int(labels.sum())  # vehicles are the minority class of the clean split
    idx = torch.cat([
        torch.where(labels.squeeze(1) == c)[0][torch.randperm(int((labels == c).sum()))[:n_per_class]]
        for c in (1, 0)
    ])
    dataset = torch.utils.data.TensorDataset(project(images[idx], n_dims, torch.float32), labels[idx])
    loader = torch.utils.data.DataLoader(dataset, batch_size=PT_BATCHSIZE, shuffle=True)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=PT_LEARNING_RATE)
    for _ in tqdm.trange(PT_N_EPOCHS, desc=f"Pre-train (d={n_dims})", leave=False):
        for x, u in loader:
            x, u = x.to(device), u.to(device)
            loss = criterion(model(x).squeeze().float(), u.squeeze().float())
            if hp.reg_strength > 0:
                loss = loss + hp.reg_strength * robust_regularization.parameter_gradient_interval_regularizer(
                    model, x, u, "binary_cross_entropy", hp.pt_epsilon, PT_MODEL_EPSILON
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def make_config(k_poison, epsilon, hp, refine=None):
    """
    The fine-tuning config of ``hp`` at the given attack, refined with the ``"sensitivity"`` heuristic
    when ``refine`` is an ``(n_splits, n_dims)`` pair. The leaves are sharded across ranks whenever the
    run is launched with more than one process.
    """
    config = AGTConfig(
        fragsize=FRAGSIZE,
        learning_rate=hp.learning_rate,
        n_epochs=hp.n_epochs,
        lr_decay=0.0,
        lr_min=0.0,
        device=DEVICE,
        loss="binary_cross_entropy",
        log_level="WARNING",
        k_poison=k_poison,
        epsilon=epsilon,
    )
    if refine is not None:
        n_splits, n_dims = refine
        config.input_refinement = agt.InputRefinementConfig(
            n_splits=n_splits, n_dims=n_dims, strategy="sensitivity", max_leaves=MAX_LEAVES,
            leaf_chunk=LEAF_CHUNK, shard_leaves=world_size() > 1,
        )
    return config


def _cache_path(config, n_dims, pt_id):
    return f"{dirs()[0]}/cifar_{n_dims}_{pt_id}_{config.hash()}"


def train_certified(config, n_dims, model, datasets, pt_id, require_cached=False):
    """
    Fine-tune the whole dense model with AGT on the held-out training split, with the clean training
    split as unpoisoned data. Cached on the config hash, the feature width and ``pt_id``; under torchrun
    rank 0 decides whether the cache is hit and writes it.

    ``require_cached`` raises ``FileNotFoundError`` instead of training, so an aggregation script can
    never start a refinement run by accident.

    Returns:
        tuple[IntervalBoundedModel, dict]: the bounded model, and the run record ``{"violation",
            "seconds", "world_size"}`` -- worst interval-validation violation (0.0 when sound), wall
            time of the training call, and the number of ranks that ran it.
    """
    fname = _cache_path(config, n_dims, pt_id)
    bounded_model = IntervalBoundedModel(copy.deepcopy(model))
    cached = torch.tensor(int(os.path.isfile(fname) and USE_CACHED))
    if dist.is_initialized():
        cached = cached.to(DEVICE)
        dist.broadcast(cached, src=0)

    if cached.item():
        bounded_model.load_params(fname)
        with open(f"{fname}.json") as file:
            return bounded_model, json.load(file)
    if require_cached:
        raise FileNotFoundError(f"certified run not cached: {fname}")

    torch.manual_seed(SEED)
    dl_train = torch.utils.data.DataLoader(datasets["held_out_train"], batch_size=HELD_OUT_BATCHSIZE, shuffle=True)
    dl_clean = torch.utils.data.DataLoader(datasets["clean_train"], batch_size=CLEAN_BATCHSIZE, shuffle=True)
    start = time.time()
    with octmnist_pca.count_violations() as counter:
        agt.poison_certified_training(bounded_model, config, dl_train, dl_clean=dl_clean)
    record = {"violation": counter.worst, "seconds": time.time() - start, "world_size": world_size()}
    if rank() == 0:
        def write_record(tmp):
            with open(tmp, "w") as file:
                json.dump(record, file)
        script_utils.atomic_write(f"{fname}.json", write_record)
        script_utils.atomic_write(fname, bounded_model.save_params)
    return bounded_model, record


def certified_metrics(bounded_model, held_out, clean):
    """
    Certified parameter-box width and, over the box, worst-case and nominal accuracy and cross-entropy
    on the ``held_out`` split and accuracy on the ``clean`` split. Clean accuracy is what exposes a
    collapsed single-class predictor.
    """
    acc_w, acc_n, _ = agt.test_metrics.test_accuracy(bounded_model, *held_out.tensors, epsilon=0)
    ce_w, ce_n, _ = agt.test_metrics.test_cross_entropy(bounded_model, *held_out.tensors, epsilon=0)
    clean_w, clean_n, _ = agt.test_metrics.test_accuracy(bounded_model, *clean.tensors, epsilon=0)
    box_width = sum((u - l).sum().item() for l, u in zip(bounded_model.param_l, bounded_model.param_u))
    return {
        "cert_acc": acc_w, "nominal_acc": acc_n, "cert_ce": ce_w, "nominal_ce": ce_n,
        "cert_clean_acc": clean_w, "nominal_clean_acc": clean_n, "box_width": box_width,
    }


def run_cell(n_dims, k_poison, epsilon, refine=None, dtype=torch.float32, require_cached=False):
    """
    One certified run of the experiments (E1-E3) at the selected hyperparameters for ``n_dims``.

    Returns:
        dict: the run record, plus ``"val"`` and ``"test"`` metrics from ``certified_metrics``.
    """
    hp = selected_hyperparameters(n_dims)
    datasets = get_datasets(n_dims, dtype)
    model = get_pretrained_model(n_dims, hp, dtype)
    bounded, record = train_certified(
        make_config(k_poison, epsilon, hp, refine), n_dims, model, datasets, pretrain_id(hp, dtype), require_cached
    )
    return {
        **record,
        "dtype": str(dtype).split(".")[-1],
        "val": certified_metrics(bounded, datasets["held_out_val"], datasets["clean_val"]),
        "test": certified_metrics(bounded, datasets["held_out_test"], datasets["clean_test"]),
    }


def cached_record(n_dims, k_poison, epsilon, refine=None, dtype=torch.float32):
    """The run record ``train_certified`` stored for a run of ``run_cell``, or None if it has not run."""
    hp = selected_hyperparameters(n_dims)
    fname = _cache_path(make_config(k_poison, epsilon, hp, refine), n_dims, pretrain_id(hp, dtype))
    if not os.path.isfile(fname):
        return None
    with open(f"{fname}.json") as file:
        return json.load(file)


def reported_cell(n_dims, k_poison, epsilon, refine=None):
    """
    The cached result to report for a run: the float32 run when sound, otherwise its float64 rerun.
    Raises ``FileNotFoundError`` when the run, or the rerun an unsound run needs, is missing.
    """
    result = run_cell(n_dims, k_poison, epsilon, refine, require_cached=True)
    if result["violation"] <= VIOLATION_TOLERANCE:
        return result
    return run_cell(n_dims, k_poison, epsilon, refine, dtype=torch.float64, require_cached=True)


if __name__ == "__main__":
    pca = fit_pca()[0]
    splits = pixel_splits()
    print(f"CIFAR-10, held out: {CLASS_NAMES[HELD_OUT_CLASS]} | "
          + ", ".join(f"{name} {len(labels)}" for name, (_, labels) in splits.items()))
    for d in (20, 22, 24):
        print(f"  d={d}: {pca.explained_variance_ratio_[:d].sum():.1%} of pixel variance, "
              f"one grey level <= {pixel_equivalent_epsilon(d):.4f} in scaled feature units")
