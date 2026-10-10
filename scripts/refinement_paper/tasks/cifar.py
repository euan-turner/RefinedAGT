"""
CIFAR-10 task of the input-ball refinement experiments (interface: tasks/__init__.py; the PCA
feature map and pre-training are described in tasks/pca_features.py).

Task. CIFAR-10 relabelled as binary: vehicle (airplane, automobile, ship, truck) = 1, animal = 0.
    - Held-out data: the automobile images. A model pre-trained on the other nine classes does worst on
      them, so fine-tuning has real work to do. They are the only data the attacker can touch.
    - Clean data: the other nine classes, trusted.
A validation split (1000 held-out and 4500 clean training images, a fixed seeded draw) was used once
to select the hyperparameters below and is kept out of training; reported numbers use the test split.

Model and training. Pixels are projected onto ``PCA_DIMS`` = 20 PCA features fit on the clean training
images. A dense ``20 -> 512 ReLU -> 1`` network is pre-trained on the (class-balanced) clean data with
a robust regulariser, then fine-tuned with Abstract Gradient Training on batches of 2000 held-out and
3000 clean samples for 2 epochs, in float32.

Hyperparameters. The pre-training radius and strength, hidden width, learning rate and number of
epochs were selected at d=20 by a 324-point grid search scoring certified held-out validation accuracy
at k=50, epsilon=0.01, subject to a clean validation accuracy of at least 0.75, and are frozen here.

Threat model and certificate. Up to ``k`` samples of every held-out batch may have their feature
vector moved anywhere within an l_inf ball of radius ``epsilon``. Reported over the certified
parameter box (tasks/pca_features.certified_metrics): certified and nominal accuracy and cross-entropy
on the held-out test split, accuracy on the clean test split, and box width.

Precision. Runs are float32; every run records its worst bound violation and aggregate.py reports the
float64 rerun of any run that exceeds harness.VIOLATION_TOLERANCE.

Refinement ladder. A bisection ladder up to all 20 features, and quadrisection at equal cost to 2^16
and 2^20.

Key external dependencies: torchvision (CIFAR-10), abstract_gradient_training, and the sibling
modules harness and tasks.pca_features.
"""

import functools

import torch
import torch.utils.data
import torchvision

from abstract_gradient_training import AGTConfig

import harness
from tasks import pca_features

NAME = "cifar"
PCA_DIMS = 20
TITLE = f"CIFAR-10 (PCA, $d={PCA_DIMS}$)"
DTYPE = torch.float32
K_POISON = 200
EPSILON = 0.02
LADDER = ["2^4", "2^8", "2^12", "2^16", "2^20", "4^8", "4^10"]
LEAF_CHUNK = 32
SECONDS_PER_LEAF = 3.5e-3  # one GH200, float32

SEED = 1
VEHICLE_CLASSES = (0, 1, 8, 9)  # airplane, automobile, ship, truck -> label 1; the animals -> 0
HELD_OUT_CLASS = 1  # automobile: the class a model pre-trained on the other nine does worst on
HELD_OUT_VAL_SIZE = 1000  # of the 5000 held-out training images
CLEAN_VAL_SIZE = 4500  # of the 45000 clean training images
PCA_MAX_DIMS = 32

HELD_OUT_BATCHSIZE = 2000  # incomplete batches are dropped: 4000 training images give two full batches
CLEAN_BATCHSIZE = 3000

# selected at d=20 (see "Hyperparameters" above)
HIDDEN_DIM = 512
PT_EPSILON = 0.01
PT_REG_STRENGTH = 0.3
LEARNING_RATE = 0.05
N_EPOCHS = 2


def prepare():
    _datasets(DTYPE)
    initial_model(DTYPE)


@functools.cache
def _pixel_splits():
    """
    ``{held_out,clean}_{train,val,test}``, each an ``(images, labels)`` pair of [N, 3072] pixels in
    [0, 1] and [N, 1] binary labels (vehicle = 1). The val splits are a fixed, seeded draw from the
    training images and are disjoint from the train splits.
    """
    generator = torch.Generator().manual_seed(SEED)
    splits = {}
    for train, stage in ((True, "train"), (False, "test")):
        data = torchvision.datasets.CIFAR10(root=harness.directory(".data"), train=train, download=True)
        images = torch.tensor(data.data, dtype=torch.float32).flatten(1) / 255
        classes = torch.tensor(data.targets, dtype=torch.int64)
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
def _pca_state():
    cache_name = f"cifar_pca_held{HELD_OUT_CLASS}_val{HELD_OUT_VAL_SIZE}-{CLEAN_VAL_SIZE}_seed{SEED}"
    return pca_features.fit_pca(cache_name, lambda: _pixel_splits()["clean_train"][0].numpy(), PCA_MAX_DIMS)


@functools.cache
def _datasets(dtype):
    """The splits of ``_pixel_splits`` as TensorDatasets over the leading ``PCA_DIMS`` scaled PCA features."""
    return {
        name: torch.utils.data.TensorDataset(pca_features.project(_pca_state(), images, PCA_DIMS, dtype), labels)
        for name, (images, labels) in _pixel_splits().items()
    }


def initial_model(dtype):
    def pretraining_data():
        torch.manual_seed(SEED)
        images, labels = _pixel_splits()["clean_train"]
        n_per_class = int(labels.sum())  # vehicles are the minority class of the clean split
        idx = torch.cat([
            torch.where(labels.squeeze(1) == c)[0][torch.randperm(int((labels == c).sum()))[:n_per_class]]
            for c in (1, 0)
        ])
        features = pca_features.project(_pca_state(), images[idx], PCA_DIMS, torch.float32)
        return torch.utils.data.TensorDataset(features, labels[idx])

    name = (f"cifar_pca_held{HELD_OUT_CLASS}_val{HELD_OUT_VAL_SIZE}-{CLEAN_VAL_SIZE}_d{PCA_DIMS}_h{HIDDEN_DIM}_"
            f"seed{SEED}_eps{PT_EPSILON}_reg{PT_REG_STRENGTH}")
    return pca_features.pretrained_model(
        name, PCA_DIMS, HIDDEN_DIM, SEED, pretraining_data, PT_EPSILON, PT_REG_STRENGTH, dtype
    )


def make_config(k, epsilon):
    return AGTConfig(
        fragsize=2000,
        learning_rate=LEARNING_RATE,
        n_epochs=N_EPOCHS,
        lr_decay=0.0,
        lr_min=0.0,
        loss="binary_cross_entropy",
        log_level="WARNING",
        k_poison=k,
        epsilon=epsilon,
    )


def training_data(dtype):
    datasets = _datasets(dtype)
    torch.manual_seed(SEED)
    return (
        torch.utils.data.DataLoader(datasets["held_out_train"], batch_size=HELD_OUT_BATCHSIZE, shuffle=True),
        torch.utils.data.DataLoader(datasets["clean_train"], batch_size=CLEAN_BATCHSIZE, shuffle=True),
    )


def evaluate(bounded_model, dtype):
    datasets = _datasets(dtype)
    return pca_features.certified_metrics(bounded_model, datasets["held_out_test"], datasets["clean_test"])
