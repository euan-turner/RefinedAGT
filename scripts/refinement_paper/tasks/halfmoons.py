"""
Half-moons task of the input-ball refinement experiments (interface: tasks/__init__.py).

Task. Binary classification on 2-D half-moons with quadratic and cubic features appended (6 inputs),
as in scripts/poisoning_paper/halfmoons.py, with a ``6 -> 128 ReLU -> 2`` network trained in float64
from a fixed initialisation with cross-entropy loss: four full-batch steps over 3000 training points.

Threat model and certificate. Up to ``k`` training samples may have their input features moved
anywhere within an l_inf ball of radius ``epsilon``. Abstract Gradient Training returns a parameter
box that contains every model any such attack could produce. Reported over the box: worst-case
(certified), nominal and best-case accuracy on 500 held-out points, and box width.

Refinement ladder. All 6 features are cut into N equal pieces, N^6 leaves, for increasing N.

Key external dependencies: scikit-learn (make_moons), abstract_gradient_training.
"""

import functools

import numpy as np
import sklearn.datasets
import sklearn.model_selection
import torch
import torch.utils.data

from abstract_gradient_training import AGTConfig, test_metrics

import harness

NAME = "halfmoons"
TITLE = "half-moons ($d=6$)"
DTYPE = torch.float64
K_POISON = 200
EPSILON = 0.01
LADDER = ["2^6", "3^6", "4^6", "5^6", "6^6", "8^6", "10^6", "12^6", "16^6"]
LEAF_CHUNK = 192  # peaks ~22 GB here (256 runs a 32 GB card out of memory)
SECONDS_PER_LEAF = 1.5e-3  # one GH200, float64

SEED = 1234
N_FEATURES = 6
HIDDEN_DIM = 128
N_TRAIN = 3000
N_TEST = 500


def prepare():
    _datasets()


@functools.cache
def _datasets():
    """(x_train, y_train, x_test, y_test) as float64 features and int64 labels."""
    x, y = sklearn.datasets.make_moons(noise=0.1, random_state=0, n_samples=N_TRAIN + N_TEST)
    x[y == 0, 1] += 0.2  # separate the moons
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=N_TEST / (N_TRAIN + N_TEST), random_state=42
    )
    x_train = np.hstack((x_train, x_train**2, x_train**3))
    x_test = np.hstack((x_test, x_test**2, x_test**3))
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
        learning_rate=2.0,
        n_epochs=4,
        loss="cross_entropy",
        lr_decay=0.6,
        lr_min=1e-3,
        log_level="WARNING",
        k_poison=k,
        epsilon=epsilon,
    )


def training_data(dtype):
    x_train, y_train, _, _ = _datasets()
    dataset = torch.utils.data.TensorDataset(x_train.to(dtype), y_train)
    return torch.utils.data.DataLoader(dataset, batch_size=N_TRAIN, shuffle=False), None


def evaluate(bounded_model, dtype):
    _, _, x_test, y_test = _datasets()
    worst, nominal, best = test_metrics.test_accuracy(bounded_model, x_test.to(dtype), y_test)
    return {"cert_acc": worst, "nominal_acc": nominal, "best_acc": best, "box_width": harness.box_width(bounded_model)}
