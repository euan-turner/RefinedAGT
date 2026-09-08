# %%
"""
Feature-poisoning sweep for OCT-MNIST: certified bounds with and without input-ball refinement.

Compares refinement to coarse interval bounding over the feature-poisoning sweep on OCT-MNIST.
All other hyperparameters are fixed, including the pre-trained model, so the runs share the
same nominal model, and their certified parameter boxes and test accuracy are directly comparable.
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

EPSILON = 0.01  # fixed l_inf feature-poisoning radius; octmnist_sweep.py frame 1 uses 0.01
K_POISON_VALUES = [0, 50, 100, 200, 300, 400, 600]

# Input-ball refinement applied to the "refined" run at every k. The dense head sees a fixed conv
# transform, so only "widest"/"first" are sound here. 2 ** 4 = 16 leaves per fragment.
N_SPLITS = 2
N_DIMS = 4
STRATEGY = "widest"

DEVICE = octmnist_train.NOMINAL_CONFIG.device
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    DEVICE = "cpu"


def base_config(k_poison, refined):
    """NOMINAL_CONFIG with the feature-poisoning budget set and refinement optionally enabled."""
    config = copy.deepcopy(octmnist_train.NOMINAL_CONFIG)
    config.device = DEVICE
    config.k_poison = k_poison
    config.epsilon = EPSILON
    if refined:
        config.input_refinement = agt.InputRefinementConfig(
            n_splits=N_SPLITS, n_dims=N_DIMS, strategy=STRATEGY
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
""" Sweep: at each k, one unrefined run and one refined run, identical otherwise. """

print(
    f"OCT-MNIST feature poisoning | eps={EPSILON} | refinement: n_splits={N_SPLITS}, n_dims={N_DIMS} "
    f"({N_SPLITS ** N_DIMS} leaves), strategy={STRATEGY}\n",
    file=sys.stderr,
)
print(
    f"  {'k':>4s}  {'box width':>12s} {'box width':>12s} {'width':>7s}   "
    f"{'cert acc':>9s} {'cert acc':>9s}  {'nominal':>8s}",
    file=sys.stderr,
)
print(
    f"  {'':>4s}  {'(baseline)':>12s} {'(refined)':>12s} {'gain':>7s}   {'(baseline)':>9s} {'(refined)':>9s}",
    file=sys.stderr,
)

rows = []
for k in tqdm.tqdm(K_POISON_VALUES):
    base_model = run_certified(base_config(k, refined=False), pretrained_model, dataset_drusen, dataset_clean)
    refined_model = run_certified(base_config(k, refined=True), pretrained_model, dataset_drusen, dataset_clean)

    base = certified_metrics(base_model, test_dataset_drusen.tensors)
    refined = certified_metrics(refined_model, test_dataset_drusen.tensors)

    # the nominal fine-tuning trajectory is refinement-independent: same nominal model at each k
    assert abs(base["nominal_acc"] - refined["nominal_acc"]) < 1e-9, (
        f"nominal accuracy diverged at k={k}: {base['nominal_acc']} vs {refined['nominal_acc']}"
    )

    width_gain = 1 - refined["box_width"] / base["box_width"] if base["box_width"] else 0.0
    rows.append({"k": k, "baseline": base, "refined": refined, "width_gain": width_gain})
    print(
        f"  {k:>4d}  {base['box_width']:12.4e} {refined['box_width']:12.4e} {width_gain:6.1%}   "
        f"{base['cert_acc_worst']:9.4f} {refined['cert_acc_worst']:9.4f}  {base['nominal_acc']:8.4f}",
        file=sys.stderr,
    )

# %%
""" Plot: certified box width and certified (worst-case) Drusen-test accuracy against k. """

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

axs[1].plot(ks, [r["baseline"]["cert_acc_worst"] for r in rows], marker="o",
            color=script_utils.colours["grey"], label="baseline")
axs[1].plot(ks, [r["refined"]["cert_acc_worst"] for r in rows], marker="o",
            color=script_utils.colours["green"], label=f"refined ({N_SPLITS ** N_DIMS} leaves)")
axs[1].plot(ks, [r["baseline"]["nominal_acc"] for r in rows], linestyle="--",
            color=script_utils.colours["orange"], label="nominal")
axs[1].set_xlabel("attack size ($k_\\mathrm{poison}$)")
axs[1].set_ylabel("certified Drusen-test accuracy")
axs[1].set_ylim(0, 1.0)
axs[1].legend(fontsize="x-small")

fig.suptitle(rf"OCT-MNIST feature poisoning, $\epsilon={EPSILON}$", fontsize="small")
script_utils.apply_figure_size(fig, script_utils.set_size(0.9, subplots, shrink_height=1.6), dpi=300)
fig_dir = script_utils.make_dirs()[3]
path = f"{fig_dir}/octmnist_refinement_sweep.pdf"
plt.savefig(path, dpi=300)
print(f"\n  figure: {path}", file=sys.stderr)
