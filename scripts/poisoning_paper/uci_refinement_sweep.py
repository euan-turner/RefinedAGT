# %%
"""
Feature-poisoning sweep for UCI house-electric: certified bounds with and without input-ball refinement.

Compares refinement to coarse interval bounding over the feature-poisoning sweep on UCI house-electric.
All other hyperparameters are fixed, so the runs share the same nominal model, and their certified
parameter boxes and test accuracy are directly comparable.

house-electric has 11 continuous features, so occupies the intermediate regime between half-moons (refinement
moves the certified metric) and OCT-MNIST (it does not): the raw gradient width tightens, but whether that reaches
the updated parameter box depends on how saturated the gradient endpoints are under ``clip_gamma``.
"""




import copy
import itertools
import os

import torch
import torch.utils.data
import tqdm
import matplotlib.pyplot as plt

import abstract_gradient_training as agt
from abstract_gradient_training.bounded_models import IntervalBoundedModel

import train_uci
import script_utils

# %%
""" Script parameters. Everything below is shared by the refined and unrefined run at each k. """

USE_CACHED = True  # reuse cached parameter boxes keyed on config.hash()

SEED = 15
BATCHSIZE = 10000
HIDDEN_LAY = 1
HIDDEN_SIZE = 64
MAX_ITERS = 150  # early-stopping cap, as in train_uci.get_training_bounds

EPSILON = 0.01  # fixed l_inf feature-poisoning radius; uci_training_plots.py frame 1 uses 0.01
K_POISON_VALUES = [1000, 5000, 10000]  # uci_training_plots.py frame 1 sweep

# Input-ball refinement applied to the "refined" run at every k. 11 continuous features, no
# transform -> the sensitivity heuristic is available. 2 ** 4 = 16 leaves per fragment.
N_SPLITS = 2
N_DIMS = 4
STRATEGY = "sensitivity"

DEVICE = train_uci.NOMINAL_CONFIG.device
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    DEVICE = "cpu"


def base_config(k_poison, refined):
    """NOMINAL_CONFIG with the feature-poisoning budget set and refinement optionally enabled."""
    config = copy.deepcopy(train_uci.NOMINAL_CONFIG)
    config.device = DEVICE
    config.log_level = "WARNING"
    config.k_poison = k_poison
    config.epsilon = EPSILON
    if refined:
        config.input_refinement = agt.InputRefinementConfig(
            n_splits=N_SPLITS, n_dims=N_DIMS, strategy=STRATEGY
        )
    return config


def run_certified(config, dl_train, dl_test):
    """
    Certified training from a fresh model, mirroring train_uci.get_training_bounds (same seed,
    interval_matmul, 150-iteration early stop) but returning the bounded model, with a
    config.hash() cache.
    """
    results_dir, _, _, _ = script_utils.make_dirs()
    fname = f"{results_dir}/uci_refinement_{SEED}_{HIDDEN_LAY}_{HIDDEN_SIZE}_{config.hash()}"

    torch.manual_seed(SEED)
    model = train_uci.get_model(HIDDEN_LAY, HIDDEN_SIZE, SEED)
    bounded_model = IntervalBoundedModel(model, interval_matmul="exact").to(DEVICE)

    if os.path.isfile(fname) and USE_CACHED:
        bounded_model.load_params(fname)
        return bounded_model

    iter_count = 0

    def early_stop(_):
        nonlocal iter_count
        iter_count += 1
        return iter_count >= MAX_ITERS

    config = copy.deepcopy(config)
    config.early_stopping_callback = early_stop
    torch.manual_seed(SEED)
    agt.poison_certified_training(bounded_model, config, dl_train, dl_test)
    bounded_model.save_params(fname)
    return bounded_model


def certified_metrics(bounded_model, test_point, test_label):
    """Certified parameter-box width and (worst, nominal, best) test MSE over that box."""
    worst, nominal, best = agt.test_metrics.test_mse(bounded_model, test_point, test_label)
    box_width = sum((u - l).sum().item() for l, u in zip(bounded_model.param_l, bounded_model.param_u))
    return {
        "mse_worst": worst,
        "mse_nominal": nominal,
        "mse_best": best,
        "mse_width": abs(worst - best),
        "box_width": box_width,
    }


# %%
""" Build the shared dataset. train_uci.get_model handles the (cached) model. """

torch.manual_seed(SEED)
DL_TRAIN, DL_TEST = train_uci.get_dataset(BATCHSIZE)
DL_TRAIN = list(itertools.islice(DL_TRAIN, MAX_ITERS))
TEST_POINT, TEST_LABEL = next(iter(DL_TEST))

# %%
""" Sweep: at each k, one unrefined run and one refined run, identical otherwise. """

print(
    f"UCI house-electric feature poisoning | eps={EPSILON} | refinement: n_splits={N_SPLITS}, "
    f"n_dims={N_DIMS} ({N_SPLITS ** N_DIMS} leaves), strategy={STRATEGY}\n"
)
print(
    f"  {'k':>6s}  {'box width':>12s} {'box width':>12s} {'width':>7s}   "
    f"{'cert MSE':>10s} {'cert MSE':>10s}  {'nominal':>8s}"
)
print(
    f"  {'':>6s}  {'(baseline)':>12s} {'(refined)':>12s} {'gain':>7s}   "
    f"{'(baseline)':>10s} {'(refined)':>10s}"
)

rows = []
for k in tqdm.tqdm(K_POISON_VALUES):
    base_model = run_certified(base_config(k, refined=False), DL_TRAIN, DL_TEST)
    refined_model = run_certified(base_config(k, refined=True), DL_TRAIN, DL_TEST)

    base = certified_metrics(base_model, TEST_POINT, TEST_LABEL)
    refined = certified_metrics(refined_model, TEST_POINT, TEST_LABEL)

    # the nominal trajectory is refinement-independent: same nominal model at each k
    assert abs(base["mse_nominal"] - refined["mse_nominal"]) < 1e-9, (
        f"nominal MSE diverged at k={k}: {base['mse_nominal']} vs {refined['mse_nominal']}"
    )

    width_gain = 1 - refined["box_width"] / base["box_width"] if base["box_width"] else 0.0
    rows.append({"k": k, "baseline": base, "refined": refined, "width_gain": width_gain})
    print(
        f"  {k:>6d}  {base['box_width']:12.4e} {refined['box_width']:12.4e} {width_gain:6.1%}   "
        f"{base['mse_worst']:10.4e} {refined['mse_worst']:10.4e}  {base['mse_nominal']:8.4e}"
    )

# %%
""" Plot: certified box width and certified worst-case test MSE against k. """

ks = [r["k"] for r in rows]
subplots = (1, 2)
fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300)

axs[0].plot(ks, [r["baseline"]["box_width"] for r in rows], marker="o",
            color=script_utils.colours["grey"], label="baseline")
axs[0].plot(ks, [r["refined"]["box_width"] for r in rows], marker="o",
            color=script_utils.colours["green"], label=f"refined ({N_SPLITS ** N_DIMS} leaves)")
axs[0].set_yscale("log")
axs[0].set_xlabel("attack size ($k_\\mathrm{poison}$)")
axs[0].set_ylabel("certified parameter box width")
axs[0].legend(fontsize="x-small")

axs[1].plot(ks, [r["baseline"]["mse_worst"] for r in rows], marker="o",
            color=script_utils.colours["grey"], label="baseline")
axs[1].plot(ks, [r["refined"]["mse_worst"] for r in rows], marker="o",
            color=script_utils.colours["green"], label=f"refined ({N_SPLITS ** N_DIMS} leaves)")
axs[1].plot(ks, [r["baseline"]["mse_nominal"] for r in rows], linestyle="--",
            color=script_utils.colours["orange"], label="nominal")
axs[1].set_yscale("log")
axs[1].set_xlabel("attack size ($k_\\mathrm{poison}$)")
axs[1].set_ylabel("certified worst-case test MSE")
axs[1].legend(fontsize="x-small")

fig.suptitle(rf"UCI house-electric feature poisoning, $\epsilon={EPSILON}$", fontsize="small")
script_utils.apply_figure_size(fig, script_utils.set_size(0.9, subplots, shrink_height=1.6), dpi=300)
fig_dir = script_utils.make_dirs()[3]
path = f"{fig_dir}/uci_refinement_sweep.pdf"
plt.savefig(path, dpi=300)
print(f"\n  figure: {path}")
