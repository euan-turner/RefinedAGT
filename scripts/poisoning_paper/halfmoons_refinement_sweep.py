# %%
"""
Feature-poisoning sweep for half-moons: certified bounds with and without input-ball refinement.

Compares refinement to coarse interval bounding over the feature-poisoning sweep on half-moons.
All other hyperparameters are fixed, so the runs share the same nominal model, and their certified
parameter boxes and test accuracy are directly comparable.
"""

import copy

import numpy as np
import torch
import torch.utils.data
import sklearn.datasets
import sklearn.model_selection
import matplotlib.pyplot as plt

import abstract_gradient_training as agt
from abstract_gradient_training import AGTConfig, test_metrics
from abstract_gradient_training.bounded_models import IntervalBoundedModel

import script_utils

# %%
""" Script parameters. Everything below is shared by the refined and unrefined run at each k. """

SEED = 1234
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

HIDDEN_DIM = 128
BATCHSIZE = 3000
EPSILON = 0.01  # fixed l_inf feature-poisoning radius; halfmoons.py frame 1 uses 0.01
K_POISON_VALUES = [50, 100, 200, 300]  # halfmoons.py frame 1 sweep

# Input-ball refinement configuration applied to the "refined" run at every k. 2 ** 4 = 16 leaves
# per fragment, the headline setting from scripts/poisoning_paper/input_ball_refinement.py.
N_SPLITS = 2
N_DIMS = 4
STRATEGY = "sensitivity"


def get_dataloaders(train_batchsize, test_batchsize=500, random_state=0, noise=0.1, n_samples=3000, sep=0.2):
    """
    Half-moons with quadratic and cubic features appended, as in scripts/poisoning_paper/halfmoons.py.
    Returns (train, test) dataloaders over 6-feature inputs.
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


def certified_metrics(bounded_model, x_test, y_test):
    """Certified parameter-box width and (worst, nominal, best) test accuracy over that box."""
    worst, nominal, best = test_metrics.test_accuracy(bounded_model, x_test, y_test)
    box_width = sum((u - l).sum().item() for l, u in zip(bounded_model.param_l, bounded_model.param_u))
    return {"cert_acc_worst": worst, "nominal_acc": nominal, "cert_acc_best": best, "box_width": box_width}


def run_certified(model, config, dl_train, dl_test):
    """Certified training from a fresh IntervalBoundedModel wrapping `model`'s (untouched) weights."""
    bounded_model = IntervalBoundedModel(model)
    agt.poison_certified_training(bounded_model, config, dl_train, dl_test)
    return bounded_model


# %%
""" Build the shared dataset, model and base config. """

torch.manual_seed(SEED)
DL_TRAIN, DL_TEST = get_dataloaders(BATCHSIZE)
X_TEST, Y_TEST = DL_TEST.dataset.tensors  # type: ignore

torch.manual_seed(SEED)
MODEL = torch.nn.Sequential(
    torch.nn.Linear(6, HIDDEN_DIM),
    torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN_DIM, 2),
).double()

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

# %%
""" Sweep: at each k, one unrefined run and one refined run, identical otherwise. """

print(
    f"halfmoons feature poisoning | eps={EPSILON} | refinement: n_splits={N_SPLITS}, n_dims={N_DIMS} "
    f"({N_SPLITS ** N_DIMS} leaves), strategy={STRATEGY}\n"
)
header = (
    f"  {'k':>4s}  {'box width':>12s} {'box width':>12s} {'width':>7s}   "
    f"{'cert acc':>9s} {'cert acc':>9s}  {'nominal':>8s}"
)
print(header)
print(f"  {'':>4s}  {'(baseline)':>12s} {'(refined)':>12s} {'gain':>7s}   {'(baseline)':>9s} {'(refined)':>9s}")

rows = []
for k in K_POISON_VALUES:
    base_config = copy.deepcopy(BASE_CONFIG)
    base_config.k_poison = k

    refined_config = copy.deepcopy(BASE_CONFIG)
    refined_config.k_poison = k
    refined_config.input_refinement = agt.InputRefinementConfig(
        n_splits=N_SPLITS, n_dims=N_DIMS, strategy=STRATEGY
    )

    base_model = run_certified(MODEL, base_config, DL_TRAIN, DL_TEST)
    refined_model = run_certified(MODEL, refined_config, DL_TRAIN, DL_TEST)

    base = certified_metrics(base_model, X_TEST, Y_TEST)
    refined = certified_metrics(refined_model, X_TEST, Y_TEST)

    # the nominal trajectory is refinement-independent: same nominal model at each k
    assert abs(base["nominal_acc"] - refined["nominal_acc"]) < 1e-9, (
        f"nominal accuracy diverged at k={k}: {base['nominal_acc']} vs {refined['nominal_acc']}"
    )

    width_gain = 1 - refined["box_width"] / base["box_width"] if base["box_width"] else 0.0
    rows.append({"k": k, "baseline": base, "refined": refined, "width_gain": width_gain})
    print(
        f"  {k:>4d}  {base['box_width']:12.4e} {refined['box_width']:12.4e} {width_gain:6.1%}   "
        f"{base['cert_acc_worst']:9.4f} {refined['cert_acc_worst']:9.4f}  {base['nominal_acc']:8.4f}"
    )

# %%
""" Plot: certified box width and certified (worst-case) test accuracy against k. """

ks = [r["k"] for r in rows]
subplots = (1, 2)
fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300)

axs[0].plot(ks, [r["baseline"]["box_width"] for r in rows], marker="o",
            color=script_utils.colours["grey"], label="baseline")
axs[0].plot(ks, [r["refined"]["box_width"] for r in rows], marker="o",
            color=script_utils.colours["green"], label=f"refined ({N_SPLITS ** N_DIMS} leaves)")
axs[0].set_yscale("log")
axs[0].set_xlabel("$k_\\mathrm{poison}$")
axs[0].set_ylabel("certified parameter box width")
axs[0].legend(fontsize="x-small")

axs[1].plot(ks, [r["baseline"]["cert_acc_worst"] for r in rows], marker="o",
            color=script_utils.colours["grey"], label="baseline")
axs[1].plot(ks, [r["refined"]["cert_acc_worst"] for r in rows], marker="o",
            color=script_utils.colours["green"], label=f"refined ({N_SPLITS ** N_DIMS} leaves)")
axs[1].plot(ks, [r["baseline"]["nominal_acc"] for r in rows], linestyle="--",
            color=script_utils.colours["orange"], label="nominal")
axs[1].set_xlabel("$k_\\mathrm{poison}$")
axs[1].set_ylabel("certified worst-case test accuracy")
axs[1].legend(fontsize="x-small")

fig.suptitle(rf"half-moons feature poisoning, $\epsilon={EPSILON}$", fontsize="small")
script_utils.apply_figure_size(fig, script_utils.set_size(0.9, subplots, shrink_height=1.6), dpi=300)
fig_dir = script_utils.make_dirs()[3]
path = f"{fig_dir}/halfmoons_refinement_sweep.pdf"
plt.savefig(path, dpi=300)
print(f"\n  figure: {path}")
