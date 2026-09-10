# %%
"""
Leaf-count scaling of input-ball refinement for OCT-MNIST feature poisoning.

Fixes the attack size and every training hyperparameter (including the pre-trained model, so the
nominal model is constant) and sweeps the refinement leaf count. The eps-ball is on the 28x28
image, so "split every input dimension" is infeasible; the schedule instead grows the number of
split pixels first (at n_splits=2), then raises n_splits. Reports how the certified parameter box
and certified test accuracy respond as the leaf count grows.
"""

import copy
import os
import sys

import torch
import torch.utils.data
import tqdm
import matplotlib.pyplot as plt

import abstract_gradient_training as agt

import octmnist_train
import script_utils

# %%
""" Script parameters. Everything below is shared by the refined and unrefined run at each k. """

USE_CACHED = True  # reuse cached parameter boxes keyed on config.hash(); see octmnist_train.run_with_config

EPSILON = 0.01  # fixed l_inf feature-poisoning radius; octmnist_sweep.py uses 0.01
K_POISON = 50  # fixed attack size, the representative point in octmnist_sweep.py

# Leaf-count sweep for the refined run. The dense head sees a fixed conv transform, so only
# "widest"/"first" are sound and "split every dimension" is out of reach for a 784-pixel input;
# the schedule grows the number of split pixels first (at n_splits=2), then raises n_splits.
# leaves per fragment = n_splits ** n_dims.
STRATEGY = "widest"
LEAF_SCHEDULE = [(2, 4), (2, 6), (2, 8), (2, 10), (2, 12)]  # 16, 64, 256, 1024, 4096 leaves
MAX_LEAVES = 10_000_000  # raise the InputRefinementConfig cost guard for the larger rungs
LEAF_CHUNK = None  # None -> 1 here. The fixed conv transform is re-bounded per chunk over
#                    chunk*fragsize images, so even leaf_chunk=1 already peaks ~27 GB and
#                    leaf_chunk>=2 OOMs a 32 GB card -- this problem is transform-bound, not
#                    leaf-chunk-tunable. (Refinement does not move the certified metric here anyway.)

DEVICE = octmnist_train.NOMINAL_CONFIG.device
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    DEVICE = "cpu"


def make_config(refine):
    """NOMINAL_CONFIG at the fixed attack size, with refinement enabled when ``refine`` is a
    ``(n_splits, n_dims)`` pair (``None`` for the unrefined baseline)."""
    config = copy.deepcopy(octmnist_train.NOMINAL_CONFIG)
    config.device = DEVICE
    config.k_poison = K_POISON
    config.epsilon = EPSILON
    if refine is not None:
        n_splits, n_dims = refine
        config.input_refinement = agt.InputRefinementConfig(
            n_splits=n_splits, n_dims=n_dims, strategy=STRATEGY,
            max_leaves=MAX_LEAVES, leaf_chunk=LEAF_CHUNK,
        )
    return config


def run_certified(config, pretrained_model, dataset_drusen, dataset_clean):
    """
    Fine-tune the dense head with the fixed conv layers as a transform, mirroring
    octmnist_train.run_with_config (including its config.hash() cache and seed handling).
    """
    results_dir, _, _, _ = script_utils.make_dirs()
    fname = f"{results_dir}/octmnist_refinement_{config.hash()}"
    conv_bounded_model = agt.bounded_models.IntervalBoundedModel(pretrained_model[0:5], trainable=False)
    bounded_model = agt.bounded_models.IntervalBoundedModel(pretrained_model[5:], transform=conv_bounded_model)

    if os.path.isfile(fname) and USE_CACHED:
        bounded_model.load_params(fname)
        return bounded_model

    torch.manual_seed(octmnist_train.SEED)
    dl_train = torch.utils.data.DataLoader(
        dataset_drusen, batch_size=octmnist_train.DRUSEN_BATCHSIZE, shuffle=True
    )
    dl_train_clean = torch.utils.data.DataLoader(
        dataset_clean, batch_size=octmnist_train.CLEAN_BATCHSIZE, shuffle=True
    )
    agt.poison_certified_training(bounded_model, config, dl_train, dl_clean=dl_train_clean)
    bounded_model.save_params(fname)
    return bounded_model


def certified_metrics(bounded_model, test_tensors):
    """Certified parameter-box width and (worst, nominal, best) Drusen-test accuracy over that box."""
    worst, nominal, best = agt.test_metrics.test_accuracy(bounded_model, *test_tensors, epsilon=0)
    box_width = sum((u - l).sum().item() for l, u in zip(bounded_model.param_l, bounded_model.param_u))
    return {"cert_acc_worst": worst, "nominal_acc": nominal, "cert_acc_best": best, "box_width": box_width}


# %%
""" Build the shared datasets and pretrained model. """

torch.manual_seed(0)
dataset_drusen, test_dataset_drusen = octmnist_train.get_dataset(exclude_classes=[0, 1, 3])
dataset_clean, _ = octmnist_train.get_dataset(exclude_classes=[2])
pretrained_model = octmnist_train.get_pretrained_model()

# %%
""" Sweep: one unrefined run, then a refined run per rung of the leaf schedule. """

print(
    f"OCT-MNIST feature poisoning | eps={EPSILON} | k_poison={K_POISON} | strategy={STRATEGY} | "
    f"leaf schedule {LEAF_SCHEDULE}\n",
    file=sys.stderr,
)
print(
    f"  {'leaves':>9s}  {'box width':>12s} {'box width':>12s} {'width':>7s}   "
    f"{'cert acc':>9s} {'cert acc':>9s}  {'nominal':>8s}",
    file=sys.stderr,
)
print(
    f"  {'':>9s}  {'(baseline)':>12s} {'(refined)':>12s} {'gain':>7s}   {'(baseline)':>9s} {'(refined)':>9s}",
    file=sys.stderr,
)

base_model = run_certified(make_config(refine=None), pretrained_model, dataset_drusen, dataset_clean)
base = certified_metrics(base_model, test_dataset_drusen.tensors)

rows = []
for n_splits, n_dims in tqdm.tqdm(LEAF_SCHEDULE):
    refined_model = run_certified(
        make_config(refine=(n_splits, n_dims)), pretrained_model, dataset_drusen, dataset_clean
    )
    refined = certified_metrics(refined_model, test_dataset_drusen.tensors)

    # the nominal fine-tuning trajectory is refinement-independent: same nominal model at every rung
    assert abs(base["nominal_acc"] - refined["nominal_acc"]) < 1e-9, (
        f"nominal accuracy diverged at {n_splits=}, {n_dims=}: "
        f"{base['nominal_acc']} vs {refined['nominal_acc']}"
    )

    leaves = n_splits ** n_dims
    width_gain = 1 - refined["box_width"] / base["box_width"] if base["box_width"] else 0.0
    rows.append({"leaves": leaves, "baseline": base, "refined": refined, "width_gain": width_gain})
    print(
        f"  {leaves:>9d}  {base['box_width']:12.4e} {refined['box_width']:12.4e} {width_gain:6.1%}   "
        f"{base['cert_acc_worst']:9.4f} {refined['cert_acc_worst']:9.4f}  {base['nominal_acc']:8.4f}",
        file=sys.stderr,
    )

# %%
""" Plot: certified box width and certified (worst-case) Drusen-test accuracy against leaf count. """

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

axs[1].axhline(rows[0]["baseline"]["cert_acc_worst"], linestyle=":",
               color=script_utils.colours["grey"], label="baseline (no refinement)")
axs[1].plot(leaves, [r["refined"]["cert_acc_worst"] for r in rows], marker="o",
            color=script_utils.colours["green"], label="refined")
axs[1].axhline(rows[0]["baseline"]["nominal_acc"], linestyle="--",
               color=script_utils.colours["orange"], label="nominal")
axs[1].set_xscale("log")
axs[1].set_xlabel("refinement leaves per fragment")
axs[1].set_ylabel("certified Drusen-test accuracy")
axs[1].set_ylim(0, 1.0)
axs[1].legend(fontsize="x-small")

fig.suptitle(rf"OCT-MNIST: certified tightness vs refinement leaves, $k={K_POISON}$, $\epsilon={EPSILON}$",
             fontsize="small")
script_utils.apply_figure_size(fig, script_utils.set_size(0.9, subplots, shrink_height=1.6), dpi=300)
fig_dir = script_utils.make_dirs()[3]
path = f"{fig_dir}/octmnist_refinement_sweep.pdf"
plt.savefig(path, dpi=300)
print(f"\n  figure: {path}", file=sys.stderr)
