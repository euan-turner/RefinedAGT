"""
OCT-MNIST task of the input-ball refinement experiments (interface: tasks/__init__.py; the PCA
feature map and pre-training are described in tasks/pca_features.py).

Task. OCT-MNIST retinal scans relabelled as binary: normal = 0; abnormal (CNV, DME, drusen) = 1.
    - Drusen data: the drusen scans. The model is pre-trained without them, so fine-tuning on them has
      real work to do. They are the only data the attacker can touch.
    - Clean data: normal, CNV and DME scans, trusted.
The splits are those of the upstream pixel-space experiment (scripts/poisoning_paper/octmnist_train.py).

Model and training. Images are projected onto ``PCA_DIMS`` = 15 PCA features fit on the clean training
split. A dense ``15 -> 256 ReLU -> 1`` network is pre-trained on the (class-balanced) clean data with
a robust regulariser, then fine-tuned with Abstract Gradient Training on batches of 3000 drusen and
3000 clean samples for 3 epochs, in float32.

Threat model and certificate. Up to ``k`` samples of every drusen batch may have their feature vector
moved anywhere within an l_inf ball of radius ``epsilon``. Reported over the certified parameter box
(tasks/pca_features.certified_metrics): certified and nominal accuracy and cross-entropy on the
drusen test split, accuracy on the clean test split, and box width.

Precision. float32 holds for this schedule at this width but not by a wide margin (at d=12 the same
schedule violates its bounds by up to 1e-1), so every run records its worst bound violation and
aggregate.py reports the float64 rerun of any run that exceeds harness.VIOLATION_TOLERANCE.

Refinement ladder. A bisection ladder up to all 15 features, then finer cuts: 3^10 against 2^15 at
similar cost, 3^12 and 4^10, and 3^15, which cuts every feature into three.

Key external dependencies: medmnist (OCT-MNIST), abstract_gradient_training, the sibling module
tasks.pca_features, and scripts/poisoning_paper/octmnist_train.py.
"""

import functools

import torch
import torch.utils.data

from abstract_gradient_training import AGTConfig

from tasks import pca_features

import octmnist_train  # upstream, made importable by tasks.pca_features

NAME = "octmnist"
PCA_DIMS = 15
TITLE = f"OCT-MNIST (PCA, $d={PCA_DIMS}$)"
DTYPE = torch.float32
K_POISON = 200
EPSILON = 0.1
LADDER = ["2^5", "2^10", "2^15", "3^10", "3^12", "4^10", "3^15"]
LEAF_CHUNK = 16
SECONDS_PER_LEAF = 6e-3  # one GH200, float32

SEED = octmnist_train.SEED
PCA_MAX_DIMS = 32
HIDDEN_DIM = 256
# Radius of the pre-training regulariser, in scaled feature units. It is the strongest lever on how
# tight the final certificate is: 0.01 leaves a wide box, 0.1 zeroes the network into a constant
# predictor whose certificate is trivially perfect (which the clean-split accuracy exposes).
PT_EPSILON = 0.05
PT_REG_STRENGTH = 0.3


def prepare():
    from medmnist import OCTMNIST

    OCTMNIST(split="train", download=True)
    _datasets(DTYPE)
    initial_model(DTYPE)


@functools.cache
def _pca_state():
    def clean_training_images():
        clean_train, _ = octmnist_train.get_dataset(exclude_classes=[2])
        return clean_train.tensors[0].flatten(1).numpy()

    return pca_features.fit_pca(f"octmnist_pca_{SEED}", clean_training_images, PCA_MAX_DIMS)


@functools.cache
def _datasets(dtype):
    """(drusen_train, drusen_test, clean_train, clean_test) TensorDatasets over [N, PCA_DIMS] features."""
    datasets = []
    for exclude in ([0, 1, 3], [2]):
        for split in octmnist_train.get_dataset(exclude_classes=exclude):
            images, labels = split.tensors
            features = pca_features.project(_pca_state(), images, PCA_DIMS, dtype)
            datasets.append(torch.utils.data.TensorDataset(features, labels))
    return tuple(datasets)


def initial_model(dtype):
    def pretraining_data():
        torch.manual_seed(SEED)
        balanced_clean_train, _ = octmnist_train.get_dataset(exclude_classes=[2], balanced=True)
        images, labels = balanced_clean_train.tensors
        features = pca_features.project(_pca_state(), images, PCA_DIMS, torch.float32)
        return torch.utils.data.TensorDataset(features, labels)

    name = f"octmnist_pca_d{PCA_DIMS}_h{HIDDEN_DIM}_seed{SEED}_eps{PT_EPSILON}_reg{PT_REG_STRENGTH}"
    return pca_features.pretrained_model(
        name, PCA_DIMS, HIDDEN_DIM, SEED, pretraining_data, PT_EPSILON, PT_REG_STRENGTH, dtype
    )


def make_config(k, epsilon):
    return AGTConfig(
        fragsize=2000,
        learning_rate=0.1,
        n_epochs=3,
        lr_decay=0.0,
        lr_min=0.0,
        loss="binary_cross_entropy",
        log_level="WARNING",
        k_poison=k,
        epsilon=epsilon,
    )


def training_data(dtype):
    drusen_train, _, clean_train, _ = _datasets(dtype)
    torch.manual_seed(SEED)
    return (
        torch.utils.data.DataLoader(drusen_train, batch_size=octmnist_train.DRUSEN_BATCHSIZE, shuffle=True),
        torch.utils.data.DataLoader(clean_train, batch_size=octmnist_train.CLEAN_BATCHSIZE, shuffle=True),
    )


def evaluate(bounded_model, dtype):
    _, drusen_test, _, clean_test = _datasets(dtype)
    return pca_features.certified_metrics(bounded_model, drusen_test, clean_test)
