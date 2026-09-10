# %%
"""
Leaf-count scaling of input-ball refinement for UCI house-electric feature poisoning.

Fixes the attack size and every training hyperparameter (so the nominal model is constant) and
sweeps the refinement leaf count, following the "split every input dimension first, then raise the
number of splits per dimension" schedule. house-electric has 11 continuous features, so the schedule
splits all 11 and only n_splits grows. Uses Rump interval matmul (as in the half-moons script)
rather than the exact product, so the leaf chunk fits in memory at large leaf counts.
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

EPSILON = 0.01  # fixed l_inf feature-poisoning radius; matches notebooks/UCI - Poisoning.ipynb
K_POISON = 200  # fixed attack size; matches notebooks/UCI - Poisoning.ipynb

# Leaf-count sweep for the refined run. Priority: split every input dimension first, then raise
# the number of splits per dimension. house-electric has 11 features, so every entry splits all
# 11 and only n_splits grows: leaves = n_splits ** 11 -> 2048, 177147, 4194304. Uniform splits
# make the rungs coarse; the (3, 11) rung is ~10^5 leaves and takes hours, (4, 11) is impractical.
STRATEGY = "sensitivity"
LEAF_SCHEDULE = [(2, 11)]  # add (3, 11) for a second rung -- 177147 leaves, ~hours at 150 iters
MAX_LEAVES = 10_000_000  # raise the InputRefinementConfig cost guard for the larger rungs
LEAF_CHUNK = 192  # leaves stacked per bounding call; 192 peaks ~27 GB here (256 OOMs a 32 GB
#                   card). Wall time is Python-loop-bound at batch 10k, so a larger chunk barely
#                   changes it -- it is the memory ceiling, not a speed knob, for this problem.
INTERVAL_MATMUL = "rump"  # was "exact"; align with the half-moons script and free the leaf-chunk
#                           memory blow-up (exact materialises a [chunk*batch, in, hidden] tensor)

DEVICE = train_uci.NOMINAL_CONFIG.device
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    DEVICE = "cpu"


def make_config(refine):
    """NOMINAL_CONFIG at the fixed attack size, with refinement enabled when ``refine`` is a
    ``(n_splits, n_dims)`` pair (``None`` for the unrefined baseline)."""
    config = copy.deepcopy(train_uci.NOMINAL_CONFIG)
    config.device = DEVICE
    config.log_level = "WARNING"
    config.k_poison = K_POISON
    config.epsilon = EPSILON
    if refine is not None:
        n_splits, n_dims = refine
        config.input_refinement = agt.InputRefinementConfig(
            n_splits=n_splits, n_dims=n_dims, strategy=STRATEGY,
            max_leaves=MAX_LEAVES, leaf_chunk=LEAF_CHUNK,
        )
    return config


def run_certified(config, dl_train, dl_test):
    """
    Certified training from a fresh model, mirroring train_uci.get_training_bounds (same seed,
    interval_matmul, 150-iteration early stop) but returning the bounded model, with a
    config.hash() cache.
    """
    results_dir, _, _, _ = script_utils.make_dirs()
    fname = f"{results_dir}/uci_refinement_{SEED}_{HIDDEN_LAY}_{HIDDEN_SIZE}_{INTERVAL_MATMUL}_{config.hash()}"

    torch.manual_seed(SEED)
    model = train_uci.get_model(HIDDEN_LAY, HIDDEN_SIZE, SEED)
    bounded_model = IntervalBoundedModel(model, interval_matmul=INTERVAL_MATMUL).to(DEVICE)

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
""" Sweep: one unrefined run, then a refined run per rung of the leaf schedule. """

print(
    f"UCI house-electric feature poisoning | eps={EPSILON} | k_poison={K_POISON} | "
    f"matmul={INTERVAL_MATMUL} | strategy={STRATEGY} | leaf schedule {LEAF_SCHEDULE}\n"
)
print(
    f"  {'leaves':>9s}  {'box width':>12s} {'box width':>12s} {'width':>7s}   "
    f"{'cert MSE':>10s} {'cert MSE':>10s}  {'nominal':>8s}"
)
print(
    f"  {'':>9s}  {'(baseline)':>12s} {'(refined)':>12s} {'gain':>7s}   "
    f"{'(baseline)':>10s} {'(refined)':>10s}"
)

base_model = run_certified(make_config(refine=None), DL_TRAIN, DL_TEST)
base = certified_metrics(base_model, TEST_POINT, TEST_LABEL)

rows = []
for n_splits, n_dims in tqdm.tqdm(LEAF_SCHEDULE):
    refined_model = run_certified(make_config(refine=(n_splits, n_dims)), DL_TRAIN, DL_TEST)
    refined = certified_metrics(refined_model, TEST_POINT, TEST_LABEL)

    # the nominal trajectory is refinement-independent: same nominal model at every rung
    assert abs(base["mse_nominal"] - refined["mse_nominal"]) < 1e-9, (
        f"nominal MSE diverged at {n_splits=}, {n_dims=}: "
        f"{base['mse_nominal']} vs {refined['mse_nominal']}"
    )

    leaves = n_splits ** n_dims
    width_gain = 1 - refined["box_width"] / base["box_width"] if base["box_width"] else 0.0
    rows.append({"leaves": leaves, "baseline": base, "refined": refined, "width_gain": width_gain})
    print(
        f"  {leaves:>9d}  {base['box_width']:12.4e} {refined['box_width']:12.4e} {width_gain:6.1%}   "
        f"{base['mse_worst']:10.4e} {refined['mse_worst']:10.4e}  {base['mse_nominal']:8.4e}"
    )

# %%
""" Plot: certified box width and certified worst-case test MSE against leaf count. """

leaves = [r["leaves"] for r in rows]
subplots = (1, 2)
fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300)

axs[0].axhline(rows[0]["baseline"]["box_width"], linestyle=":",
               color=script_utils.colours["grey"], label="baseline (no refinement)")
axs[0].plot(leaves, [r["refined"]["box_width"] for r in rows], marker="o",
            color=script_utils.colours["green"], label="refined")
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_xlabel("refinement leaves per fragment")
axs[0].set_ylabel("certified parameter box width")
axs[0].legend(fontsize="x-small")

axs[1].axhline(rows[0]["baseline"]["mse_worst"], linestyle=":",
               color=script_utils.colours["grey"], label="baseline (no refinement)")
axs[1].plot(leaves, [r["refined"]["mse_worst"] for r in rows], marker="o",
            color=script_utils.colours["green"], label="refined")
axs[1].axhline(rows[0]["baseline"]["mse_nominal"], linestyle="--",
               color=script_utils.colours["orange"], label="nominal")
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].set_xlabel("refinement leaves per fragment")
axs[1].set_ylabel("certified worst-case test MSE")
axs[1].legend(fontsize="x-small")

fig.suptitle(rf"UCI house-electric: certified tightness vs refinement leaves, "
             rf"$k={K_POISON}$, $\epsilon={EPSILON}$", fontsize="small")
script_utils.apply_figure_size(fig, script_utils.set_size(0.9, subplots, shrink_height=1.6), dpi=300)
fig_dir = script_utils.make_dirs()[3]
path = f"{fig_dir}/uci_refinement_sweep.pdf"
plt.savefig(path, dpi=300)
print(f"\n  figure: {path}")
