"""
MAGIC gamma-telescope task of the input-ball refinement experiments (interface: tasks/__init__.py).

Task. Binary classification (gamma = 0, hadron = 1; 65/35) on the MAGIC gamma telescope dataset
(OpenML 1120, 10 continuous features, 19020 samples), split 80/20 stratified and standardised with the
training-split mean and standard deviation. A ``10 -> 128 ReLU -> 2`` network is trained in float64
from a fixed initialisation with cross-entropy loss by full-batch gradient descent.

Threat model and certificate. As half-moons: up to ``k`` training samples may have their features
moved anywhere within an l_inf ball of radius ``epsilon`` (in standardised units). Reported over the
certified parameter box: worst-case (certified), nominal and best-case accuracy on the 3804 held-out
points, and box width. ``majority_acc`` (0.648) is the test accuracy of always predicting the majority
class, the floor a certificate must beat to mean anything.

Training schedule. Three full-batch steps at learning rate 1.0 (decay 0.6). Bounds compound with the
number of steps: at k=200 the unrefined certificate falls from 0.73 after two steps to 0.60 after
three and 0.10 after four. Three steps leave the unrefined certificate below the majority baseline, so
refinement has room to show a gain.

Refinement ladder. Ten features are too many to cut every one finely, so the ladder alternates between
cutting the more sensitive half of the features one step finer and catching the other half up.

Key external dependencies: scikit-learn (fetch_openml; the first call needs network access),
abstract_gradient_training.
"""

import functools

import numpy as np
import sklearn.datasets
import sklearn.model_selection
import torch
import torch.utils.data

from abstract_gradient_training import AGTConfig, test_metrics

import harness

NAME = "magic"
TITLE = "MAGIC ($d=10$)"
DTYPE = torch.float64
K_POISON = 200
EPSILON = 0.01
LADDER = ["2^5", "2^10", "3^5x2^5", "3^10", "4^5x3^5", "4^10"]
LEAF_CHUNK = 96  # 192 runs a 32 GB card out of memory: the first-layer gradient interval is
#                  [chunk * FRAGSIZE, 10, 128] float64, 5.5 GiB per tensor at 192
SECONDS_PER_LEAF = 1.8e-2  # one RTX 5090, float64; not yet measured on a GH200, where it should be lower

SEED = 1234
N_FEATURES = 10
HIDDEN_DIM = 128
TEST_FRACTION = 0.2
FRAGSIZE = 3000


def prepare():
    _datasets()


@functools.cache
def _datasets():
    """(x_train, y_train, x_test, y_test) as float64 standardised features and int64 labels."""
    bunch = sklearn.datasets.fetch_openml(
        "MagicTelescope", version=1, data_home=harness.directory(".data"), as_frame=False
    )
    x = bunch.data.astype(np.float64)
    y = (bunch.target == "h").astype(np.int64)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=TEST_FRACTION, random_state=42, stratify=y
    )
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    return tuple(torch.from_numpy(a) for a in (x_train, y_train, x_test, y_test))


def initial_model(dtype):
    torch.manual_seed(SEED)
    return torch.nn.Sequential(
        torch.nn.Linear(N_FEATURES, HIDDEN_DIM),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, 2),
    ).to(dtype)


def make_config(k, epsilon):
    return AGTConfig(
        learning_rate=1.0,
        n_epochs=3,
        loss="cross_entropy",
        lr_decay=0.6,
        lr_min=1e-3,
        fragsize=FRAGSIZE,
        log_level="WARNING",
        k_poison=k,
        epsilon=epsilon,
    )


def training_data(dtype):
    x_train, y_train, _, _ = _datasets()
    dataset = torch.utils.data.TensorDataset(x_train.to(dtype), y_train)
    return torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False), None


def evaluate(bounded_model, dtype):
    _, y_train, x_test, y_test = _datasets()
    worst, nominal, best = test_metrics.test_accuracy(bounded_model, x_test.to(dtype), y_test)
    majority = (y_test == y_train.mode().values).double().mean().item()
    return {
        "cert_acc": worst,
        "nominal_acc": nominal,
        "best_acc": best,
        "majority_acc": majority,
        "box_width": harness.box_width(bounded_model),
    }
