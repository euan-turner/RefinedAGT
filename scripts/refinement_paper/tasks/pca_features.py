"""
Shared pieces of the two image tasks (tasks/octmnist.py, tasks/cifar.py): a fixed PCA feature map, a
robustly pre-trained dense model, and their certified test metrics.

Why PCA. Input refinement partitions the input one coordinate at a time, which is out of reach for
raw pixels. A fixed PCA projection gives a model with a low-dimensional input, so most or all
coordinates can be cut. The projection is fit on trusted data the adversary does not control and is
treated as a fixed, public feature map: the l_inf poisoning ball of radius ``epsilon`` is on the PCA
feature vector, not on the pixels. Each retained component is min-max scaled to [0, 1] on the data it
was fit on, so ``epsilon`` is a fraction of the observed feature range.

The basis is fit once at a maximum width with an exact solver and sliced to the requested width, so
feature sets of different widths are nested.

Results of these tasks are not bit-reproducible across machines: the same runs gave certified
accuracies differing in the third decimal between an x86_64 workstation and the aarch64 cluster.
Both the PCA fit and the pre-trained model are cached, so every run on one machine starts from the
same files.

Key external dependencies: scikit-learn (PCA), tqdm, abstract_gradient_training, and
scripts/poisoning_paper/robust_regularization.py (the upstream robust regulariser).
"""

import os

import numpy as np
import sklearn.decomposition
import torch
import torch.utils.data
import tqdm

from abstract_gradient_training import test_metrics

import harness
from tasks import upstream_scripts  # noqa: F401 - makes the upstream poisoning_paper modules importable

import robust_regularization

PCA_SVD_SOLVER = "covariance_eigh"  # exact, so the leading d components of a fit are the d-component fit

PT_BATCHSIZE = 100
PT_N_EPOCHS = 10
PT_LEARNING_RATE = 0.001
PT_MODEL_EPSILON = 0.001


def fit_pca(cache_name, images, n_components):
    """
    The PCA basis of ``images()`` (an [N, P] float array, only evaluated when nothing is cached) and
    the per-component min-max scaling of its scores. Cached as ``.data/<cache_name>.npz``; under
    torchrun only rank 0 writes the cache.

    Returns:
        tuple[sklearn.decomposition.PCA, np.ndarray, np.ndarray]: the projection, and the
            per-component minimum and range that scale scores into [0, 1].
    """
    cache_path = os.path.join(harness.directory(".data"), f"{cache_name}_{n_components}_{PCA_SVD_SOLVER}.npz")
    pca = sklearn.decomposition.PCA(n_components=n_components, svd_solver=PCA_SVD_SOLVER)

    if os.path.isfile(cache_path):
        cached = np.load(cache_path)
        pca.components_ = cached["components"]
        pca.mean_ = cached["mean"]
        pca.explained_variance_ = cached["explained_variance"]
        pca.explained_variance_ratio_ = cached["explained_variance_ratio"]
        pca.n_components_ = n_components
        pca.n_features_in_ = cached["components"].shape[1]
        return pca, cached["score_min"], cached["score_range"]

    scores = pca.fit_transform(images())
    score_min = scores.min(axis=0)
    score_range = scores.max(axis=0) - score_min
    if harness.rank() == 0:
        harness.atomic_write(cache_path, lambda tmp: np.savez(
            tmp, components=pca.components_, mean=pca.mean_,
            explained_variance=pca.explained_variance_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            score_min=score_min, score_range=score_range))
    return pca, score_min, score_range


def project(pca_state, images, n_dims, dtype):
    """Project an [N, ...] image tensor onto the leading ``n_dims`` scaled PCA features."""
    pca, score_min, score_range = pca_state
    scores = pca.transform(images.flatten(1).numpy())[:, :n_dims]
    return torch.tensor((scores - score_min[:n_dims]) / score_range[:n_dims], dtype=dtype)


def pretrained_model(model_name, n_dims, hidden_dim, seed, pretraining_data, pt_epsilon, reg_strength, dtype):
    """
    An ``n_dims -> hidden_dim ReLU -> 1`` model pre-trained on ``pretraining_data()`` (a TensorDataset
    of float32 features and binary labels, only evaluated when nothing is cached) with binary
    cross-entropy plus ``reg_strength`` times the upstream robust regulariser of radius ``pt_epsilon``.
    Cached as ``.models/<model_name>.ckpt``; under torchrun only rank 0 trains, and the other ranks
    load its checkpoint.
    """
    model_path = os.path.join(harness.directory(".models"), f"{model_name}.ckpt")
    device = torch.device(harness.DEVICE)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(n_dims, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, 1)
    ).to(device)

    if not os.path.exists(model_path) and harness.rank() == 0:
        torch.manual_seed(seed)
        loader = torch.utils.data.DataLoader(pretraining_data(), batch_size=PT_BATCHSIZE, shuffle=True)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=PT_LEARNING_RATE)
        for _ in tqdm.trange(PT_N_EPOCHS, desc=f"Pre-train {model_name}", leave=False):
            for x, u in loader:
                x, u = x.to(device), u.to(device)
                loss = criterion(model(x).squeeze().float(), u.squeeze().float())
                if reg_strength > 0:
                    loss = loss + reg_strength * robust_regularization.parameter_gradient_interval_regularizer(
                        model, x, u, "binary_cross_entropy", pt_epsilon, PT_MODEL_EPSILON
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        harness.atomic_write(model_path, lambda tmp: torch.save(model.state_dict(), tmp))
    harness.barrier()
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    return model.cpu().to(dtype)


def certified_metrics(bounded_model, poisonable_test, clean_test):
    """
    Over the certified parameter box: worst-case (certified) and nominal accuracy and cross-entropy on
    the test split of the poisonable data, and accuracy on the test split of the trusted data, which
    exposes a model that has collapsed onto one class. Plus the width of the box.
    """
    acc_w, acc_n, _ = test_metrics.test_accuracy(bounded_model, *poisonable_test.tensors, epsilon=0)
    ce_w, ce_n, _ = test_metrics.test_cross_entropy(bounded_model, *poisonable_test.tensors, epsilon=0)
    clean_w, clean_n, _ = test_metrics.test_accuracy(bounded_model, *clean_test.tensors, epsilon=0)
    return {
        "cert_acc": acc_w, "nominal_acc": acc_n, "cert_ce": ce_w, "nominal_ce": ce_n,
        "cert_clean_acc": clean_w, "nominal_clean_acc": clean_n, "box_width": harness.box_width(bounded_model),
    }
