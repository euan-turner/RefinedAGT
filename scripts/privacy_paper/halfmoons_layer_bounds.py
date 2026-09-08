"""
Distribution of parameter bound tightness across layers for privacy-certified training.

Trains narrow-but-deep fully connected models on the halfmoons dataset with
`privacy_certified_training` and measures the width of the certified parameter interval of each
Linear layer. Hypothesis: the certified bounds on the gradients of the earlier layers are looser, so
their weight updates are wider and bound width decreases with layer depth.

Key external dependencies: torch, matplotlib, abstract_gradient_training.
"""

# %%
import logging
import os

import torch
import torch.utils.data
import matplotlib.pyplot as plt

import abstract_gradient_training as agt
from abstract_gradient_training import AGTConfig
from abstract_gradient_training.bounded_models import IntervalBoundedModel

import script_utils
import datasets
import models

# %%
"""Configure console logging so per-batch training statistics are printed."""

logging.basicConfig(level=logging.INFO, format="[AGT] [%(levelname)-8s] %(message)s")

# %%
"""Initialise helper functions."""


def layer_bound_widths(bounded_model):
    """
    Return the per-layer weight bound widths of a trained bounded model as a list of
    (layer_index, raw_width, relative_width) tuples, one per Linear module.

    The raw width is the mean of (u - l) over the weight matrix. The relative width normalises this
    by the mean magnitude of the nominal weights of the same layer, so that layers of differing
    scale are comparable. The ratio of means is used rather than the mean of the elementwise ratio,
    which is dominated by near-zero nominal weights.
    """
    widths = []
    for params_l, params_n, params_u in zip(bounded_model._param_l, bounded_model._param_n, bounded_model._param_u):
        if not params_l:  # activation modules hold no parameters
            continue
        w_l, w_n, w_u = params_l[0], params_n[0], params_u[0]
        raw = (w_u - w_l).mean().item()
        widths.append((len(widths), raw, raw / w_n.abs().mean().item()))
    return widths


def init_for_depth(model):
    """
    Re-initialise the Linear layers of `model` with Kaiming/He init tuned for ReLU (gain=sqrt(2), fan_in).

    torch.nn.Linear's default init (kaiming_uniform_ with a=sqrt(5)) under-scales for stacks of ReLU layers: as
    depth grows, more units die at each layer and the network's output collapses to a near-constant function of
    the input before training even starts. The ReLU-tuned gain compensates for the ~50% of units killed by each
    ReLU, keeping the forward signal from vanishing across depth.
    """
    for module in model:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            torch.nn.init.zeros_(module.bias)
    return model


def run_depth_sweep(depths, width, config, depth_hparams, dl_train, dl_test):
    """
    Train a model of each depth in `depths` with privacy-certified training and return a dict mapping each depth to
    the per-layer bound widths of the trained model.

    `depth_hparams` maps each depth to a dict of AGTConfig field overrides (e.g. learning_rate, n_epochs), since
    deeper networks need a higher learning rate and more epochs to reach comparable convergence.
    """
    results = {}
    for depth in depths:
        depth_config = config.model_copy(update=depth_hparams[depth])
        print(f"=== Training depth={depth} (lr={depth_config.learning_rate}, epochs={depth_config.n_epochs}) ===")
        model = models.fully_connected(SEED, width=width, depth=depth).to(DTYPE)
        init_for_depth(model)
        bounded_model = IntervalBoundedModel(model)
        agt.privacy_certified_training(bounded_model, depth_config, dl_train, dl_test)
        results[depth] = layer_bound_widths(bounded_model)
    return results


# %%
"""Set script parameters."""

SEED = 1234
DTYPE = torch.float64
WIDTH = 16
DEPTHS = [2, 4, 6, 8]
BATCHSIZE = 3000
CONFIG = AGTConfig(
    learning_rate=0.5,
    n_epochs=30,
    device="cuda:0",
    loss="cross_entropy",
    lr_decay=0.6,
    lr_min=1e-3,
    log_level="INFO",
    k_private=20,
    clip_gamma=0.06,
)

# learning_rate/n_epochs overrides per depth, found via a small grid search: deeper networks need a higher
# learning rate and more epochs to reach comparable test accuracy (see run_depth_sweep docstring).
DEPTH_HPARAMS = {
    2: dict(learning_rate=0.8, n_epochs=60),
    4: dict(learning_rate=1.5, n_epochs=250),
    6: dict(learning_rate=1.2, n_epochs=60),
    8: dict(learning_rate=1.5, n_epochs=350),
}

dataset_train, dataset_test = datasets.get_halfmoons(BATCHSIZE, SEED, dtype=DTYPE)
DL_TRAIN = torch.utils.data.DataLoader(dataset_train, batch_size=BATCHSIZE, shuffle=False)
DL_TEST = torch.utils.data.DataLoader(dataset_test, batch_size=BATCHSIZE, shuffle=False)

# %%
"""Run the depth sweep."""

results = run_depth_sweep(DEPTHS, WIDTH, CONFIG, DEPTH_HPARAMS, DL_TRAIN, DL_TEST)

# %%
"""Plot raw and relative bound width against layer index for each depth."""

subplots = (1, 2)
fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300)
palette = [script_utils.colours[c] for c in ("blue", "red", "green", "purple", "yellow", "light_blue")]

for (depth, widths), colour in zip(results.items(), palette):
    indices = [i for i, _, _ in widths]
    axs[0].plot(indices, [raw for _, raw, _ in widths], marker="o", color=colour, label=f"depth $={depth}$")
    axs[1].plot(indices, [rel for _, _, rel in widths], marker="o", color=colour, label=f"depth $={depth}$")

axs[0].set_ylabel("Mean bound width")
axs[1].set_ylabel(r"Mean bound width / mean $|w|$")
for ax in axs:
    ax.set_xlabel("Layer index")
    ax.set_yscale("log")
    ax.legend()

script_utils.apply_figure_size(fig, script_utils.set_size(1.0, subplots, shrink_height=0.8), dpi=300)
fig_dir = script_utils.make_dirs()[3]
fig_path = os.path.abspath(f"{fig_dir}/halfmoons_layer_bounds.png")
plt.savefig(fig_path, dpi=300)
print(f"saved figure to {fig_path}")
