# %%
"""
Poisoning-radius experiment on OCT-MNIST PCA features: does refinement pay more as the eps-ball grows?

Hypothesis. The refinement gain grows with ``eps``. The feature-width sweep
(octmnist_pca_refinement_sweep.py) found the gain flat at 1.2-1.7% at eps=0.01. The proposed
reason is that at that radius the input ball contributes little of the certified box, so the radius,
not the width, is the axis the gain should respond to.

Mechanism. The feature width (d=16) and attack size (k=50) are fixed, and eps is swept over
{0.005, 0.01, 0.02, 0.05, 0.1, 0.2}. The scaled features have a standard deviation of about 0.09, so
0.2 is a ball comparable to the spread of the data. At each radius it compares:
    - an unrefined run;
    - the 12 most sensitive of the 16 coordinates bisected (4096 leaves). Bisecting all 16 was
      measured to add only a further 0.16% for 16x the compute.
The nominal model is the tuned one (nominal Drusen accuracy 0.928, against 0.932 for the pixel-space
convolutional pipeline). It does not depend on eps; the script asserts this at every radius. Each run
reports its worst bound violation, so an unsound result cannot be mistaken for a real one.

Output: a table on stderr (box width, certified accuracy and cross-entropy, their gains, violation),
and a figure of box width, box-width reduction and certified cross-entropy, each against eps.

Key external dependencies: torch, matplotlib, ``abstract_gradient_training``, and the sibling
modules ``octmnist_pca``, ``octmnist_train`` and ``script_utils``.
"""

import copy
import sys

import torch
import torch.utils.data
import tqdm
import matplotlib.pyplot as plt

import abstract_gradient_training as agt

import octmnist_pca
import script_utils

# %%
""" Script parameters. Everything below is shared by the refined and unrefined run at each epsilon. """

PCA_DIMS = 16  # tuned feature width (see octmnist_pca.FINETUNE_CONFIG)
K_POISON = 50  # fixed attack size, matching the sibling sweeps

# The sweep axis: l_inf radius on the scaled PCA features. 0.01 is the sibling scripts' value and
# the low end here; the scaled components have a standard deviation of ~0.09, so 0.2 is a ball
# comparable to the spread of the data itself.
EPSILONS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

STRATEGY = "sensitivity"  # the PCA projection is preprocessing, not a model transform
N_SPLITS = 2
# Splitting the 12 most sensitive of the 16 components captures essentially all of the available
# tightening: bisecting all 16 (65536 leaves, 798 s) moved the certified box from 1.9516e+00 to
# 1.9485e+00 against the top-12 split (4096 leaves, 50 s), a further 0.16% for 16x the compute,
# with identical certified accuracy. The priority is still to bisect before raising n_splits.
MAX_SPLIT_DIMS = 12
MAX_LEAVES = 10_000_000
LEAF_CHUNK = 64  # per-sample gradient tensors are [chunk * fragsize, hidden, d]; ~1 GB at these sizes

DEVICE = octmnist_pca.FINETUNE_CONFIG.device
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    DEVICE = "cpu"


def make_config(epsilon, refine):
    """Tuned fine-tuning config at the fixed attack size and the given ball radius, with refinement
    enabled when ``refine`` is a ``(n_splits, n_dims)`` pair (``None`` for the unrefined baseline)."""
    config = copy.deepcopy(octmnist_pca.FINETUNE_CONFIG)
    config.device = DEVICE
    config.k_poison = K_POISON
    config.epsilon = epsilon
    if refine is not None:
        n_splits, n_dims = refine
        config.input_refinement = agt.InputRefinementConfig(
            n_splits=n_splits, n_dims=n_dims, strategy=STRATEGY,
            max_leaves=MAX_LEAVES, leaf_chunk=LEAF_CHUNK,
        )
    return config


def certified_metrics(bounded_model, test_tensors):
    """Certified box width, Drusen accuracy and Drusen cross-entropy over the parameter box.

    The loss is the finer signal: accuracy only moves when a test point's bound crosses the
    decision boundary, so it can sit flat while the certificate genuinely tightens.
    """
    acc_worst, acc_nominal, _ = agt.test_metrics.test_accuracy(bounded_model, *test_tensors, epsilon=0)
    ce_worst, ce_nominal, _ = agt.test_metrics.test_cross_entropy(bounded_model, *test_tensors, epsilon=0)
    box_width = sum((u - l).sum().item() for l, u in zip(bounded_model.param_l, bounded_model.param_u))
    return {
        "cert_acc_worst": acc_worst, "nominal_acc": acc_nominal,
        "cert_ce": ce_worst, "nominal_ce": ce_nominal,
        "box_width": box_width,
    }


# %%
""" Build the shared dataset and the tuned nominal model. Both are epsilon-independent. """

PCA_STATE = octmnist_pca.fit_pca()
DRUSEN_TRAIN, DRUSEN_TEST, CLEAN_TRAIN, CLEAN_TEST = octmnist_pca.get_datasets(PCA_DIMS, PCA_STATE)
MODEL = octmnist_pca.get_pretrained_model(PCA_DIMS, PCA_STATE)

# %%
""" Sweep: at each epsilon, one unrefined run and one refined run. """

print(
    f"OCT-MNIST (PCA features, d={PCA_DIMS}) | k_poison={K_POISON} | strategy={STRATEGY} | "
    f"n_splits={N_SPLITS} bisecting {MAX_SPLIT_DIMS}/{PCA_DIMS} components "
    f"({N_SPLITS**MAX_SPLIT_DIMS} leaves)\n",
    file=sys.stderr,
)
print(
    f"  {'epsilon':>8s}  {'box width':>11s} {'box width':>11s} {'width':>7s}  "
    f"{'cert acc':>8s} {'cert acc':>8s} {'acc':>7s}  {'cert CE':>8s} {'cert CE':>8s} {'CE':>7s}  {'viol':>8s}",
    file=sys.stderr,
)
print(
    f"  {'':>8s}  {'(baseline)':>11s} {'(refined)':>11s} {'gain':>7s}  "
    f"{'(baseline)':>8s} {'(refined)':>8s} {'gain':>7s}  {'(baseline)':>8s} {'(refined)':>8s} "
    f"{'gain':>7s}  {'(max)':>8s}",
    file=sys.stderr,
)

rows = []
for epsilon in tqdm.tqdm(EPSILONS, desc="epsilon", file=sys.stderr):
    base_model, base_viol = octmnist_pca.run_certified(
        make_config(epsilon, None), PCA_DIMS, MODEL, DRUSEN_TRAIN, CLEAN_TRAIN, tag="eps"
    )
    base = certified_metrics(base_model, DRUSEN_TEST.tensors)

    refined_model, refined_viol = octmnist_pca.run_certified(
        make_config(epsilon, (N_SPLITS, MAX_SPLIT_DIMS)), PCA_DIMS, MODEL, DRUSEN_TRAIN, CLEAN_TRAIN,
        tag="eps"
    )
    refined = certified_metrics(refined_model, DRUSEN_TEST.tensors)

    # the nominal trajectory is plain SGD, so it is independent of epsilon and of refinement
    assert abs(base["nominal_acc"] - refined["nominal_acc"]) < 1e-9, (
        f"nominal accuracy diverged at {epsilon=}: {base['nominal_acc']} vs {refined['nominal_acc']}"
    )

    width_gain = 1 - refined["box_width"] / base["box_width"] if base["box_width"] else 0.0
    acc_gain = refined["cert_acc_worst"] - base["cert_acc_worst"]
    ce_gain = base["cert_ce"] - refined["cert_ce"]  # loss: lower is tighter
    violation = max(base_viol, refined_viol)
    rows.append({
        "epsilon": epsilon,
        "baseline": base,
        "refined": refined,
        "width_gain": width_gain,
        "acc_gain": acc_gain,
        "ce_gain": ce_gain,
        "violation": violation,
    })
    print(
        f"  {epsilon:>8}  {base['box_width']:11.4e} {refined['box_width']:11.4e} {width_gain:6.2%}  "
        f"{base['cert_acc_worst']:8.4f} {refined['cert_acc_worst']:8.4f} {acc_gain:+7.4f}  "
        f"{base['cert_ce']:8.4f} {refined['cert_ce']:8.4f} {ce_gain:+7.4f}  {violation:8.1e}",
        file=sys.stderr,
    )

# Drusen accuracy alone is degenerate: predicting "abnormal" everywhere scores 1.0 on the Drusen
# split and 0.667 on the clean split. The fine-tuned nominal model is epsilon-independent, so one
# clean-accuracy number covers the sweep.
nominal_clean = agt.test_metrics.test_accuracy(base_model, *CLEAN_TEST.tensors, epsilon=0)[1]
print(
    f"\n  nominal model: Drusen accuracy {rows[0]['baseline']['nominal_acc']:.4f}, clean "
    f"{nominal_clean:.4f} (0.667 = predict-abnormal-everywhere), cross-entropy "
    f"{rows[0]['baseline']['nominal_ce']:.4f} | pixel-space conv reference: 0.9320 / 0.8053",
    file=sys.stderr,
)

unsound = [r["epsilon"] for r in rows if r["violation"] > 1e-3]
if unsound:
    print(f"\n  WARNING: ERROR-level bound violations at epsilon={unsound}; those rungs are unsound.",
          file=sys.stderr)

# %%
""" Plot: box width, width reduction and certified accuracy against the ball radius. """

eps = [r["epsilon"] for r in rows]
subplots = (1, 3)
fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300)

axs[0].plot(eps, [r["baseline"]["box_width"] for r in rows], marker="s", linestyle=":",
            color=script_utils.colours["grey"], label="baseline (no refinement)")
axs[0].plot(eps, [r["refined"]["box_width"] for r in rows], marker="o",
            color=script_utils.colours["green"], label="refined")
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_xlabel(r"poisoning radius $\epsilon$")
axs[0].set_ylabel("certified box width")
axs[0].legend(fontsize="x-small")

axs[1].plot(eps, [r["width_gain"] for r in rows], marker="o", color=script_utils.colours["green"])
axs[1].set_xscale("log")
axs[1].set_ylim(0, None)
axs[1].set_xlabel(r"poisoning radius $\epsilon$")
axs[1].set_ylabel("box width reduction")

axs[2].plot(eps, [r["baseline"]["cert_ce"] for r in rows], marker="s", linestyle=":",
            color=script_utils.colours["grey"], label="baseline (no refinement)")
axs[2].plot(eps, [r["refined"]["cert_ce"] for r in rows], marker="o",
            color=script_utils.colours["green"], label="refined")
axs[2].axhline(rows[0]["baseline"]["nominal_ce"], linestyle="--",
               color=script_utils.colours["orange"], label="nominal (floor)")
axs[2].set_xscale("log")
axs[2].set_xlabel(r"poisoning radius $\epsilon$")
axs[2].set_ylabel("certified cross-entropy")
axs[2].legend(fontsize="x-small")

fig.suptitle(rf"OCT-MNIST on PCA features ($d={PCA_DIMS}$): refinement gain vs poisoning radius, "
             rf"$k={K_POISON}$", fontsize="small")
script_utils.apply_figure_size(fig, script_utils.set_size(1.0, subplots, shrink_height=2.4), dpi=300)
fig_dir = script_utils.make_dirs()[3]
path = f"{fig_dir}/octmnist_pca_epsilon_sweep.pdf"
plt.savefig(path, dpi=300)
print(f"\n  figure: {path}", file=sys.stderr)
