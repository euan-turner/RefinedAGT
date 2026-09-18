# %%
"""
Feature-width experiment on OCT-MNIST: does refinement help once the input is small enough to
partition, and does the gain grow as the partition covers more of the input?

Background. On raw 28x28 pixels, refinement could not move the certified metric: splitting a handful
of 784 coordinates buys nothing. This script replaces the pixels with a fixed PCA projection (see
octmnist_pca) so that most or all input coordinates can be split.

Hypothesis. With few enough input coordinates to bisect all of them, refinement measurably tightens
the certificate, and the gain grows with ``d`` for as long as every coordinate is still bisected.

Mechanism. A fixed attack (k=50, eps=0.01) at d in {4, 6, 8, 10, 12, 16, 20}. At each width it
compares an unrefined run with a refined one that bisects ``min(d, 12)`` coordinates:
    - every coordinate up to d=12;
    - above that, the 12 most sensitive, because the leaf count 2^d is capped at 2^12.
The reported split coverage says which regime each width is in. The PCA basis is nested, so the
features at one width are the leading features of every wider one. The training hyperparameters are
the pipeline's, tuned at d=16, and are held fixed across widths.

Recorded result (Isambard, 2026-09). The hypothesis does not hold. The gain is flat at 1.2-1.7%
whatever the width or coverage, and certified accuracy never moves: at eps=0.01 the input ball is a
small part of the certified box. Hence octmnist_pca_epsilon_sweep.py sweeps the radius instead. At
d=4 and d=6 the fixed schedule collapses towards the degenerate predictor (clean accuracy 0.684 and
0.668). The script flags this through clean accuracy, and the summary figure leaves this sweep out.

Output: a table on stderr, and a figure of box width, box-width reduction and certified Drusen
accuracy, each against ``d``.

Key external dependencies: torch, matplotlib, ``abstract_gradient_training``, and the sibling
modules ``octmnist_pca`` and ``script_utils``.
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
""" Script parameters. Everything below is shared by the refined and unrefined run at each d. """

EPSILON = 0.01  # l_inf feature-poisoning radius, now on the scaled PCA features (see octmnist_pca)
K_POISON = 50  # fixed attack size, matching the removed pixel-space sweep

# Number of retained PCA components, the sweep axis. Capped by octmnist_pca.PCA_MAX_DIMS; the basis
# is nested, so the d features of one rung are the leading features of every wider rung.
PCA_DIMS = [4, 6, 8, 10, 12, 16, 20]

STRATEGY = "sensitivity"  # available here: the PCA projection is preprocessing, not a model transform
N_SPLITS = 2  # bisect every split dimension before ever raising this; see the module docstring
MAX_SPLIT_DIMS = 12  # leaf-count cap: 2 ** 12 = 4096 leaves per fragment
MAX_LEAVES = 10_000_000  # raise the InputRefinementConfig cost guard for the larger rungs
LEAF_CHUNK = 64  # leaves stacked per bounding call; per-sample gradient tensors are
#                  [chunk * fragsize, hidden, d], so ~1 GB per bound at chunk=64, d=20, fragsize=2000

DEVICE = octmnist_pca.FINETUNE_CONFIG.device
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    DEVICE = "cpu"


def make_config(refine):
    """NOMINAL_CONFIG at the fixed attack size, with refinement enabled when ``refine`` is a
    ``(n_splits, n_dims)`` pair (``None`` for the unrefined baseline)."""
    config = copy.deepcopy(octmnist_pca.FINETUNE_CONFIG)
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


def certified_metrics(bounded_model, test_tensors):
    """Certified parameter-box width and (worst, nominal, best) Drusen-test accuracy over that box."""
    worst, nominal, best = agt.test_metrics.test_accuracy(bounded_model, *test_tensors, epsilon=0)
    box_width = sum((u - l).sum().item() for l, u in zip(bounded_model.param_l, bounded_model.param_u))
    return {"cert_acc_worst": worst, "nominal_acc": nominal, "cert_acc_best": best, "box_width": box_width}


# %%
""" Build the shared PCA basis. It is fit once and sliced, so the rungs are nested. """

PCA_STATE = octmnist_pca.fit_pca()
PCA, _, SCORE_RANGE = PCA_STATE

# %%
""" Sweep: at each PCA width, one unrefined run and one refined run. """

print(
    f"OCT-MNIST (PCA features) feature poisoning | eps={EPSILON} | k_poison={K_POISON} | "
    f"strategy={STRATEGY} | n_splits={N_SPLITS} | split-dim cap {MAX_SPLIT_DIMS}\n",
    file=sys.stderr,
)
print(
    f"  {'d':>3s} {'var':>6s} {'split':>6s} {'leaves':>7s}  {'box width':>12s} {'box width':>12s} "
    f"{'width':>7s}   {'cert acc':>9s} {'cert acc':>9s}  {'nominal':>8s} {'nominal':>8s}",
    file=sys.stderr,
)
print(
    f"  {'':>3s} {'expl.':>6s} {'dims':>6s} {'':>7s}  {'(baseline)':>12s} {'(refined)':>12s} "
    f"{'gain':>7s}   {'(baseline)':>9s} {'(refined)':>9s}  {'drusen':>8s} {'clean':>8s}",
    file=sys.stderr,
)

rows = []
for d in tqdm.tqdm(PCA_DIMS, desc="PCA width", file=sys.stderr):
    drusen_train, drusen_test, clean_train, clean_test = octmnist_pca.get_datasets(d, PCA_STATE)
    model = octmnist_pca.get_pretrained_model(d, PCA_STATE)

    base_model, base_viol = octmnist_pca.run_certified(
        make_config(None), d, model, drusen_train, clean_train
    )
    base = certified_metrics(base_model, drusen_test.tensors)

    n_dims = min(d, MAX_SPLIT_DIMS)
    refined_model, refined_viol = octmnist_pca.run_certified(
        make_config((N_SPLITS, n_dims)), d, model, drusen_train, clean_train
    )
    refined = certified_metrics(refined_model, drusen_test.tensors)
    # Drusen accuracy alone is degenerate -- predicting "abnormal" everywhere scores 1.0 on it and
    # 0.667 on the clean split. Clean accuracy is reported alongside so a collapsed rung is visible.
    clean_acc = agt.test_metrics.test_accuracy(base_model, *clean_test.tensors, epsilon=0)[1]

    # the nominal trajectory is refinement-independent: same nominal model at both ends of the pair
    assert abs(base["nominal_acc"] - refined["nominal_acc"]) < 1e-9, (
        f"nominal accuracy diverged at {d=}: {base['nominal_acc']} vs {refined['nominal_acc']}"
    )

    width_gain = 1 - refined["box_width"] / base["box_width"] if base["box_width"] else 0.0
    rows.append({
        "d": d,
        "n_dims": n_dims,
        "leaves": N_SPLITS**n_dims,
        "explained_variance": float(PCA.explained_variance_ratio_[:d].sum()),
        "baseline": base,
        "refined": refined,
        "width_gain": width_gain,
        "violation": max(base_viol, refined_viol),
        "clean_acc": clean_acc,
    })
    print(
        f"  {d:>3d} {rows[-1]['explained_variance']:>6.1%} {n_dims:>3d}/{d:<2d} "
        f"{rows[-1]['leaves']:>7d}  {base['box_width']:12.4e} {refined['box_width']:12.4e} "
        f"{width_gain:6.1%}   {base['cert_acc_worst']:9.4f} {refined['cert_acc_worst']:9.4f}  "
        f"{base['nominal_acc']:8.4f} {clean_acc:8.4f}",
        file=sys.stderr,
    )

degenerate = [r["d"] for r in rows if r["clean_acc"] < 0.70]
if degenerate:
    print(f"\n  NOTE: at d={degenerate} the model has collapsed towards predicting \"abnormal\" "
          f"everywhere (clean accuracy near the 0.667 trivial rate), so its Drusen accuracy is not "
          f"meaningful. FINETUNE_CONFIG is tuned at d=16.", file=sys.stderr)

unsound = [r["d"] for r in rows if r["violation"] > 1e-3]
if unsound:
    print(f"\n  WARNING: ERROR-level bound violations at d={unsound}; those rungs are unsound.",
          file=sys.stderr)

# %%
""" Plot: box width, width gain and certified accuracy against the number of PCA components. """

dims = [r["d"] for r in rows]
subplots = (1, 3)
fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300)

axs[0].plot(dims, [r["baseline"]["box_width"] for r in rows], marker="s", linestyle=":",
            color=script_utils.colours["grey"], label="baseline (no refinement)")
axs[0].plot(dims, [r["refined"]["box_width"] for r in rows], marker="o",
            color=script_utils.colours["green"], label="refined")
axs[0].set_yscale("log")
axs[0].set_xlabel("PCA components $d$")
axs[0].set_ylabel("certified box width")
axs[0].legend(fontsize="x-small")

axs[1].plot(dims, [r["width_gain"] for r in rows], marker="o", color=script_utils.colours["green"])
axs[1].axvline(MAX_SPLIT_DIMS + 0.5, linestyle="--", color=script_utils.colours["grey"])
axs[1].text(MAX_SPLIT_DIMS + 0.7, 0.85, "partial\ncoverage", fontsize="xx-small",
            color=script_utils.colours["grey"], transform=axs[1].get_xaxis_transform(), va="top")
# anchor at zero: the gain is flat at ~1%, and an autoscaled axis would render that noise as trend
axs[1].set_ylim(0, max(0.05, 1.35 * max(r["width_gain"] for r in rows)))
axs[1].set_xlabel("PCA components $d$")
axs[1].set_ylabel("box width reduction")

axs[2].plot(dims, [r["baseline"]["cert_acc_worst"] for r in rows], marker="s", linestyle=":",
            color=script_utils.colours["grey"], label="baseline (no refinement)")
axs[2].plot(dims, [r["refined"]["cert_acc_worst"] for r in rows], marker="o",
            color=script_utils.colours["green"], label="refined")
axs[2].plot(dims, [r["baseline"]["nominal_acc"] for r in rows], marker="^", linestyle="--",
            color=script_utils.colours["orange"], label="nominal")
axs[2].set_ylim(0, 1.0)
axs[2].set_xlabel("PCA components $d$")
axs[2].set_ylabel("certified Drusen accuracy")
axs[2].legend(fontsize="x-small")

fig.suptitle(rf"OCT-MNIST on PCA features: refinement gain vs input dimension, "
             rf"$k={K_POISON}$, $\epsilon={EPSILON}$", fontsize="small")
script_utils.apply_figure_size(fig, script_utils.set_size(1.0, subplots, shrink_height=2.4), dpi=300)
fig_dir = script_utils.make_dirs()[3]
path = f"{fig_dir}/octmnist_pca_refinement_sweep.pdf"
plt.savefig(path, dpi=300)
print(f"\n  figure: {path}", file=sys.stderr)
