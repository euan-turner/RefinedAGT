"""
Shared pipeline for the OCT-MNIST feature-poisoning experiments: data splits, the PCA feature map, the
pre-trained model, and one cached run of certified training. Every ``octmnist_pca_*`` script builds on
it, and the terms defined here are used throughout them.

Task. OCT-MNIST retinal scans relabelled as binary: normal = 0; abnormal (CNV, DME, drusen) = 1.
    - Drusen data: the drusen scans. The model is pre-trained without them, so fine-tuning on them has
      real work to do. They are the only data the attacker can touch.
    - Clean data: normal, CNV and DME scans, trusted.
The splits are those of the original pixel-space experiment (``octmnist_train.py``), which used a
fixed convolutional feature extractor instead of PCA.

Why PCA. Input refinement (below) partitions the input one coordinate at a time. With 784 pixels, or
the convolutional features, that is out of reach. A fixed PCA projection gives a model with a
low-dimensional input (``d`` of roughly 4-20), so most or all coordinates can be split.

Model and training. A dense ``d -> HIDDEN_DIM ReLU -> 1`` network is pre-trained on the clean data with
a robust regulariser of radius ``PT_EPSILON``, then fine-tuned with Abstract Gradient Training (AGT)
on drusen and clean batches using ``FINETUNE_CONFIG``. These settings were tuned at d=16 (see the
comments on each constant), and every sweep uses them unless it says otherwise.

Certificate. Up to ``k_poison`` samples per drusen batch may be moved anywhere within an l_inf ball of
radius ``epsilon``. AGT carries this through training and returns a parameter box that contains every
model any such attack could produce. The sweeps report:
    - certified Drusen accuracy and cross-entropy (worst case over the box);
    - clean accuracy, which exposes a model that has collapsed onto "abnormal everywhere" (1.0 on
      Drusen, 0.667 on clean);
    - box width, the sum of per-parameter interval widths. Smaller means a tighter certificate.
The nominal model (training with no attack) does not depend on the attack or on refinement.

Refinement. ``InputRefinementConfig(n_splits, n_dims)`` cuts each poisoned sample's eps-ball along
the ``n_dims`` most sensitive coordinates into ``n_splits`` pieces each. That gives
``n_splits ** n_dims`` sub-boxes ("leaves"), which are bounded separately and then combined.
"Bisecting" means ``n_splits = 2``. More leaves give a tighter certificate at proportional cost.

Caching. Every certified run is cached on disk under its configuration, a sweep ``tag`` and the
pre-trained model, so the sweeps are cheap to rerun and share nothing by accident.

Threat model (differs from the pixel-space scripts, deliberately):
    The l_inf poisoning ball of radius ``epsilon`` is on the *PCA feature vector*, not on the pixels.
    The projection is treated as a fixed, public feature map fit on the clean (non-drusen) training
    split -- data the poisoning adversary does not control -- so the basis is constant across the
    baseline and refined runs. An l_inf ball on the PCA features is not the image of an l_inf ball
    on pixels; ``pixel_equivalent_epsilon`` reports how the two radii compare.

Feature scaling:
    Each retained component is min-max scaled to [0, 1] over the clean training split, so that
    ``epsilon`` is a fraction of the observed feature range exactly as it is for [0, 1] pixels.

Precision:
    Certified training runs in float32, which is exact for the shipped configuration but not by a
    wide margin. When the nominal gradient falls outside its own interval bounds AGT only logs it,
    leaving an unsound certificate behind, so ``count_violations`` checks every run rather than
    assuming the arithmetic held.

    Measured: at d=16 with ``FINETUNE_CONFIG``, over epsilon in {0.01, 0.05, 0.1, 0.2}, refined and
    unrefined, float32 reports zero violations and matches float64 box widths to five significant
    figures. It does fail elsewhere -- at d=12 the same schedule violates by up to 1.1e-01, and a
    longer schedule (lr=0.05 over 10 epochs, ~30 iterations against ~9 here) violates by 5e-02 --
    and float64 removes both at identical accuracy. Set ``DTYPE = torch.float64`` for a
    configuration whose guard fires.

Distributed runs:
    Launched under torchrun, each process uses the GPU of its ``LOCAL_RANK``; call
    ``init_distributed`` first. Rank 0 pre-trains and writes every cache, and the other ranks load
    from it. A refined run's leaves are only sharded across ranks when its config sets
    ``InputRefinementConfig.shard_leaves``; that flag is outside ``AGTConfig.hash()``, so a sharded
    run shares its cache entry with the single-process run of the same configuration.

Key external dependencies: medmnist (OCT-MNIST), scikit-learn (PCA), torch (torch.distributed for
sharded refinement), ``abstract_gradient_training``, and the sibling modules ``octmnist_train`` and
``robust_regularization``.
"""

# %%
import contextlib
import copy
import json
import logging
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
import tqdm

import sklearn.decomposition

import abstract_gradient_training as agt
from abstract_gradient_training import AGTConfig
from abstract_gradient_training.bounded_models import IntervalBoundedModel

import octmnist_train
import robust_regularization
import script_utils

USE_CACHED = True  # whether to reuse cached PCA bases, pre-trained models and parameter boxes
SEED = octmnist_train.SEED

# One GPU per torchrun process; outside torchrun LOCAL_RANK is unset and this is cuda:0, the device
# octmnist_train.NOMINAL_CONFIG uses.
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"

# The PCA basis is fit once at PCA_MAX_DIMS and sliced to the requested width. sklearn orders
# components by explained variance and the leading d of a D-component fit are exactly the
# d-component fit, so every rung of a dimension sweep uses strictly nested feature sets.
PCA_MAX_DIMS = 32
PCA_SVD_SOLVER = "covariance_eigh"  # exact (unlike "randomized"), so the nesting above holds

# Nominal pre-training of the dense model on the clean (non-drusen) split, mirroring
# octmnist_train.get_pretrained_model. PT_EPSILON is in scaled PCA feature units. The pixel-space
# value of 0.5 does not port across: scaled PCA components have a standard deviation of ~0.09, so
# 0.5 is >5 sigma of perturbation and the regularizer drives the model to a constant predictor.
#
# Within the usable range, PT_EPSILON is the single strongest lever on how tight the final
# certificate is, and it is not monotone in nominal accuracy either. Measured at
# d=16, HIDDEN_DIM=256, k_poison=50, epsilon=0.01 with the schedule below, and reproduced by
# octmnist_pca_plots.py:
#
#     0.01 -> box 1.64e+01, certified 0.544, nominal 0.912, clean 0.759
#     0.02 -> box 7.93e+00, certified 0.780, nominal 0.908, clean 0.761
#     0.05 -> box 1.06e+00, certified 0.896, nominal 0.932, clean 0.728   (shipped)
#     0.1  -> box 2.76e-02, certified 1.000, nominal 1.000, clean 0.667
#
# The last row is the failure mode the clean-split column exists to catch: the regularizer has
# zeroed the network, so it predicts "abnormal" everywhere and its certificate is trivially
# perfect. Longer fine-tuning schedules stretch this scale out -- in the selection search the
# weakly regularized starting points reached boxes of 1e+02 to 1e+04 and certified accuracy 0.
PT_BATCHSIZE = 100
PT_N_EPOCHS = 10
PT_LEARNING_RATE = 0.001
PT_EPSILON = 0.05
PT_MODEL_EPSILON = 0.001
PT_REG_STRENGTH = 0.3

# Widening the hidden layer past the convolutional model's 100 is what makes the PCA pipeline
# competitive: it lets the model absorb the stronger robust regularizer above, which is what keeps
# the certified box tight. At HIDDEN_DIM=100 the pre-trained clean accuracy is 0.70; at 256 it is
# 0.82 with the same regularization.
HIDDEN_DIM = 256

# See the "Precision" note in the module docstring. float32 is exact for the shipped schedule, but
# the margin is thin enough that count_violations checks rather than trusts it. Reassign this to
# torch.float64 (before building any dataset or model) for a configuration that trips the guard.
DTYPE = torch.float32

# Fine-tuning schedule. octmnist_train.NOMINAL_CONFIG (lr=0.05, lr_decay=5.0, n_epochs=2) decays the
# learning rate so fast that the PCA model barely leaves its pre-trained state: it reaches a nominal
# Drusen accuracy of 0.44 against the 0.93 of the pixel-space convolutional pipeline. Removing the
# decay recovers that. Selected by grid search over (d, HIDDEN_DIM, PT_EPSILON, PT_REG_STRENGTH,
# learning_rate, n_epochs, clean batch size) at k_poison=50, epsilon=0.01, scoring certified Drusen
# accuracy subject to the model not being degenerate. At d=16 the selected point gives:
#
#     nominal Drusen 0.932 / clean 0.728      (pixel-space conv reference: 0.932 / 0.805)
#     certified Drusen accuracy 0.896, certified cross-entropy 0.520 against a nominal 0.494
#
# measured at k_poison=50, epsilon=0.01. Drusen accuracy matches the convolutional pipeline; the
# clean-split gap is the cost of the PCA representation, and is what a wider basis would buy.
#
# Longer schedules keep raising nominal Drusen accuracy but widen the box faster than they help:
# n_epochs=6 at this learning rate reaches nominal 0.928 with certified accuracy 0.004.
FINETUNE_CONFIG = AGTConfig(
    fragsize=2000,
    learning_rate=0.1,
    n_epochs=3,
    lr_decay=0.0,
    lr_min=0.0,
    device=DEVICE,
    loss="binary_cross_entropy",
    log_level="WARNING",
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


def fit_pca(n_components=PCA_MAX_DIMS):
    """
    Fit the PCA basis and the per-component min-max scaling on the clean (non-drusen) training split.

    Returns:
        tuple[sklearn.decomposition.PCA, np.ndarray, np.ndarray]: the fitted projection, and the
            per-component minimum and range used to scale scores into [0, 1].
    """
    _, _, data_dir, _ = script_utils.make_dirs()
    cache_path = f"{data_dir}/octmnist_pca_{n_components}_{PCA_SVD_SOLVER}_{SEED}.npz"

    if os.path.isfile(cache_path) and USE_CACHED:
        cached = np.load(cache_path)
        pca = sklearn.decomposition.PCA(n_components=n_components, svd_solver=PCA_SVD_SOLVER)
        pca.components_ = cached["components"]
        pca.mean_ = cached["mean"]
        pca.explained_variance_ = cached["explained_variance"]
        pca.explained_variance_ratio_ = cached["explained_variance_ratio"]
        pca.n_components_ = n_components
        pca.n_features_in_ = cached["components"].shape[1]
        return pca, cached["score_min"], cached["score_range"]

    clean_train, _ = octmnist_train.get_dataset(exclude_classes=[2])
    images = clean_train.tensors[0].flatten(1).numpy()
    pca = sklearn.decomposition.PCA(n_components=n_components, svd_solver=PCA_SVD_SOLVER)
    scores = pca.fit_transform(images)
    score_min = scores.min(axis=0)
    score_range = scores.max(axis=0) - score_min

    if rank() == 0:
        def write_cache(tmp):
            np.savez(
                tmp,
                components=pca.components_,
                mean=pca.mean_,
                explained_variance=pca.explained_variance_,
                explained_variance_ratio=pca.explained_variance_ratio_,
                score_min=score_min,
                score_range=score_range,
            )
        script_utils.atomic_write(cache_path, write_cache)
    return pca, score_min, score_range


def project(images, pca, score_min, score_range, n_dims, dtype=None):
    """Project a [N, 1, 28, 28] image tensor onto the leading ``n_dims`` scaled PCA features.

    ``dtype`` defaults to the module-level ``DTYPE`` read at call time, so a caller that raises the
    precision (see the "Precision" note) affects the data as well as the model.
    """
    scores = pca.transform(images.flatten(1).numpy())[:, :n_dims]
    scaled = (scores - score_min[:n_dims]) / score_range[:n_dims]
    return torch.tensor(scaled, dtype=DTYPE if dtype is None else dtype)


def pixel_equivalent_epsilon(pca, score_range, n_dims, pixel_epsilon):
    """
    Per-component PCA-space radius induced by an l_inf pixel perturbation of ``pixel_epsilon``.

    Lets the PCA-space ``epsilon`` be read against the pixel-space one used by the sibling scripts:
    a pixel perturbation of radius r moves scaled component j by at most ``r * ||v_j||_1 / range_j``.

    Returns:
        tuple[float, float]: the mean and maximum induced radius over the retained components.
    """
    induced = pixel_epsilon * np.abs(pca.components_[:n_dims]).sum(axis=1) / score_range[:n_dims]
    return float(induced.mean()), float(induced.max())


def get_datasets(n_dims, pca_state=None):
    """
    OCT-MNIST in the scaled PCA feature space.

    Returns:
        tuple: ``(drusen_train, drusen_test, clean_train, clean_test)`` TensorDatasets over
            ``[N, n_dims]`` inputs, split exactly as octmnist_train's pixel-space experiment does.
    """
    pca, score_min, score_range = pca_state if pca_state is not None else fit_pca()
    datasets = []
    for exclude in ([0, 1, 3], [2]):
        train, test = octmnist_train.get_dataset(exclude_classes=exclude)
        for split in (train, test):
            images, labels = split.tensors
            datasets.append(
                torch.utils.data.TensorDataset(project(images, pca, score_min, score_range, n_dims), labels)
            )
    drusen_train, drusen_test, clean_train, clean_test = datasets
    return drusen_train, drusen_test, clean_train, clean_test


def pretrain_tag(epsilon=PT_EPSILON, model_epsilon=PT_MODEL_EPSILON, reg_strength=PT_REG_STRENGTH):
    """
    Identity of the pre-trained starting point. ``AGTConfig.hash()`` does not see it, so a certified
    run cached under the hash alone would be silently reused after the pre-training changed.
    """
    return f"{SEED}_{HIDDEN_DIM}_{epsilon}_{model_epsilon}_{reg_strength}_{str(DTYPE).split('.')[-1]}"


def get_pretrained_model(n_dims, pca_state=None, epsilon=PT_EPSILON, model_epsilon=PT_MODEL_EPSILON,
                         reg_strength=PT_REG_STRENGTH):
    """
    Nominally pre-trained dense model on the clean (non-drusen) PCA features, with the same robust
    regularizer as ``octmnist_train.get_pretrained_model``. Trains and caches one if absent; under
    torchrun only rank 0 trains, and the other ranks load its checkpoint.
    """
    _, model_dir, _, _ = script_utils.make_dirs()
    model_path = (
        f"{model_dir}/octmnist_pca_{n_dims=}_{HIDDEN_DIM=}_{SEED=}_{epsilon=}_{model_epsilon=}"
        f"_{reg_strength=}.ckpt"
    )
    device = torch.device(DEVICE)
    torch.manual_seed(SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(n_dims, HIDDEN_DIM),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, 1),
    ).to(device)

    if not (os.path.exists(model_path) and USE_CACHED) and rank() == 0:
        _pretrain(model, n_dims, pca_state, epsilon, model_epsilon, reg_strength, device)
        script_utils.atomic_write(model_path, lambda tmp: torch.save(model.state_dict(), tmp))
    if dist.is_initialized():
        dist.barrier()
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    return model.to(DTYPE)


def _pretrain(model, n_dims, pca_state, epsilon, model_epsilon, reg_strength, device):
    pca, score_min, score_range = pca_state if pca_state is not None else fit_pca()
    torch.manual_seed(SEED)
    pretrain_split, _ = octmnist_train.get_dataset(exclude_classes=[2], balanced=True)
    images, labels = pretrain_split.tensors
    dataset = torch.utils.data.TensorDataset(
        project(images, pca, score_min, score_range, n_dims, dtype=torch.float32), labels)
    dl_pretrain = torch.utils.data.DataLoader(dataset, batch_size=PT_BATCHSIZE, shuffle=True)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=PT_LEARNING_RATE)
    progress_bar = tqdm.trange(PT_N_EPOCHS, desc=f"Pre-train (d={n_dims})")
    for _ in progress_bar:
        for i, (x, u) in enumerate(dl_pretrain):
            u, x = u.to(device), x.to(device)
            output = model(x)
            bce_loss = criterion(output.squeeze().float(), u.squeeze().float())
            if reg_strength > 0:
                regularization = robust_regularization.parameter_gradient_interval_regularizer(
                    model, x, u, "binary_cross_entropy", epsilon, model_epsilon
                )
            else:
                regularization = torch.tensor(0.0)
            loss = bce_loss + reg_strength * regularization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                progress_bar.set_postfix(bce_loss=bce_loss.item(), reg=regularization.item())


class _ViolationCounter(logging.Handler):
    """Records AGT's interval-validation violations so a caller can tell whether a run is sound."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.worst = 0.0
        self.n_error = 0

    def emit(self, record):
        if "Violated bound" in record.getMessage():
            self.worst = max(self.worst, float(record.args[1]))
            self.n_error += record.levelno >= logging.ERROR


@contextlib.contextmanager
def count_violations():
    """
    Capture the interval-validation violations AGT logs during the enclosed run.

    A violation means the nominal gradient fell outside its own bounds, so the certificate is not
    sound. AGT only logs these, so a sweep that does not watch for them can report numbers from a
    run that silently failed. Yields the counter; read ``.worst`` and ``.n_error`` after the block.
    """
    counter = _ViolationCounter()
    logger = logging.getLogger("abstract_gradient_training")
    logger.addHandler(counter)
    try:
        yield counter
    finally:
        logger.removeHandler(counter)


def cache_path(config, n_dims, tag="pca", pretrain_id=None):
    """Where ``run_certified`` caches a run. ``{path}.violation`` holds its worst violation and, for
    runs trained since the record was introduced, ``{path}.json`` its ``{"seconds", "world_size",
    "peak_gib"}`` cost record."""
    results_dir, _, _, _ = script_utils.make_dirs()
    pretrain_id = pretrain_tag() if pretrain_id is None else pretrain_id
    return f"{results_dir}/octmnist_{tag}_{n_dims}_{pretrain_id}_{config.hash()}"


def run_certified(config, n_dims, model, drusen_train, clean_train, tag="pca", pretrain_id=None):
    """
    Fine-tune the whole dense model with AGT on the drusen split, with the clean split supplied as
    unpoisoned data. Results are cached on ``config.hash()``, which sees neither the feature width,
    the pre-trained starting point nor the dtype, so all three are in the filename.

    Unlike the pixel-space script this model has no fixed ``transform``: the PCA projection is a
    data preprocessing step, so the eps-ball sits directly on the first trainable layer's input and
    the ``"sensitivity"`` split-dimension heuristic is available.

    ``pretrain_id`` identifies the starting point when it is not the module-default one -- pass
    ``pretrain_tag(epsilon=...)`` whenever ``model`` came from a non-default ``get_pretrained_model``
    call, or its run will collide in the cache with the default model's.

    Under torchrun every rank must call this with the same arguments: rank 0 decides whether the
    cache is hit and is the only rank that writes it.

    Returns:
        tuple[IntervalBoundedModel, float]: the bounded model, and the worst interval-validation
            violation observed during training (0.0 when the run was sound). The violation is
            cached alongside the parameters, so it survives a cache hit.
    """
    fname = cache_path(config, n_dims, tag, pretrain_id)
    bounded_model = IntervalBoundedModel(copy.deepcopy(model))

    cached = torch.tensor(int(os.path.isfile(fname) and USE_CACHED))
    if dist.is_initialized():
        cached = cached.to(DEVICE)
        dist.broadcast(cached, src=0)
    if cached.item():
        bounded_model.load_params(fname)
        with open(f"{fname}.violation") as file:
            return bounded_model, float(file.read())

    torch.manual_seed(SEED)
    dl_train = torch.utils.data.DataLoader(
        drusen_train, batch_size=octmnist_train.DRUSEN_BATCHSIZE, shuffle=True
    )
    dl_train_clean = torch.utils.data.DataLoader(
        clean_train, batch_size=octmnist_train.CLEAN_BATCHSIZE, shuffle=True
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
    start = time.time()
    with count_violations() as counter:
        agt.poison_certified_training(bounded_model, config, dl_train, dl_clean=dl_train_clean)
    record = {
        "seconds": time.time() - start,
        "world_size": world_size(),
        "peak_gib": torch.cuda.max_memory_allocated(DEVICE) / 2**30 if torch.cuda.is_available() else None,
    }
    if rank() == 0:
        def write_violation(tmp):
            with open(tmp, "w") as file:
                file.write(str(counter.worst))

        def write_record(tmp):
            with open(tmp, "w") as file:
                json.dump(record, file)

        script_utils.atomic_write(f"{fname}.violation", write_violation)
        script_utils.atomic_write(f"{fname}.json", write_record)
        script_utils.atomic_write(fname, bounded_model.save_params)
    return bounded_model, counter.worst


# %%
if __name__ == "__main__":
    PCA_STATE = fit_pca()
    PCA, _, SCORE_RANGE = PCA_STATE
    print(f"OCT-MNIST PCA basis: {PCA_MAX_DIMS} components, "
          f"{PCA.explained_variance_ratio_.sum():.1%} of pixel variance explained")
    for d in (4, 8, 16, 32):
        mean_eq, max_eq = pixel_equivalent_epsilon(PCA, SCORE_RANGE, d, 0.01)
        print(f"  d={d:>2d}: cumulative variance {PCA.explained_variance_ratio_[:d].sum():.1%}, "
              f"a pixel eps of 0.01 induces a PCA-space radius of {mean_eq:.4f} (mean) / "
              f"{max_eq:.4f} (max) over the retained components")

    _, drusen_test, _, clean_test = get_datasets(PCA_MAX_DIMS, PCA_STATE)
    nominal = IntervalBoundedModel(get_pretrained_model(PCA_MAX_DIMS, PCA_STATE))
    for name, split in (("clean (non-drusen)", clean_test), ("drusen", drusen_test)):
        accs = agt.test_metrics.test_accuracy(nominal, *split.tensors, epsilon=0)
        print(f"  pre-trained model accuracy on the {name} test split: {accs[1]:.4f}")
