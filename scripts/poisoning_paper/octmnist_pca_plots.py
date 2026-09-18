# %%
"""
Summary figure for the OCT-MNIST feature-poisoning experiments: one 3x2 panel of the results that
stand on their own. It tests no new hypothesis; the hypotheses are stated in the sweep scripts it
draws on.

    Row 1, model selection: how the pre-training radius (``PT_EPSILON`` in {0.01, 0.02, 0.05, 0.1})
        trades nominal and clean accuracy against certified box width. It shows why this is the
        hyperparameter that decides whether the certificate says anything.
    Row 2, the poisoning radius (octmnist_pca_epsilon_sweep.py): certified cross-entropy and
        box-width reduction against eps, unrefined against the top 12 coordinates bisected.
    Row 3, the refinement budget (octmnist_pca_leaf_sweep.py): the bisection ladder and the
        equal-cost alternatives at eps=0.1.

Every number is recomputed from the ``octmnist_pca`` cache rather than restated, so the figure cannot
drift from the runs. It shares cache entries with the sweeps, but it trains the model-selection
runs of row 1 itself when they are not cached, so it belongs in a GPU job. The feature-width sweep
is deliberately left out: its gain is flat, and at d=4 and d=6 the tuned schedule collapses to the
degenerate predictor, so octmnist_pca_refinement_sweep.py keeps it with that caveat attached.

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
""" Parameters. These mirror the three sweep scripts so the panels show the same runs. """

PCA_DIMS = 16
K_POISON = 50
CONV_REFERENCE = {"drusen": 0.9320, "clean": 0.8053}  # pixel-space pipeline, octmnist_train
TRIVIAL_CLEAN = 2 / 3  # clean-split accuracy of predicting "abnormal" everywhere

PT_EPSILONS = [0.01, 0.02, 0.05, 0.1]  # model-selection panel; 0.05 is the shipped value
EPSILONS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]  # threat-model panel
LEAF_SCHEDULE = [(2, 4), (2, 8), (2, 12), (2, 16), (8, 4), (4, 8)]  # refinement-budget panel
LEAF_EPSILON = 0.1  # radius the budget panel is measured at

REFINE_DIMS = 12  # partition used by the threat-model panel, matching the epsilon sweep
STRATEGY = "sensitivity"
N_SPLITS = 2
MAX_LEAVES = 10_000_000
LEAF_CHUNK = 64


def make_config(epsilon, refine):
    """Tuned config at the fixed attack size, refined when ``refine`` is an ``(n_splits, n_dims)``."""
    config = copy.deepcopy(octmnist_pca.FINETUNE_CONFIG)
    config.k_poison = K_POISON
    config.epsilon = epsilon
    if refine is not None:
        n_splits, n_dims = refine
        config.input_refinement = agt.InputRefinementConfig(
            n_splits=n_splits, n_dims=n_dims, strategy=STRATEGY,
            max_leaves=MAX_LEAVES, leaf_chunk=LEAF_CHUNK,
        )
    return config


def metrics(bounded_model, drusen_tensors):
    """Certified box width, Drusen accuracy and Drusen cross-entropy over the parameter box."""
    acc_w, acc_n, _ = agt.test_metrics.test_accuracy(bounded_model, *drusen_tensors, epsilon=0)
    ce_w, ce_n, _ = agt.test_metrics.test_cross_entropy(bounded_model, *drusen_tensors, epsilon=0)
    return {
        "cert_acc": acc_w, "nominal_acc": acc_n, "cert_ce": ce_w, "nominal_ce": ce_n,
        "box_width": sum((u - l).sum().item() for l, u in zip(bounded_model.param_l, bounded_model.param_u)),
    }


# %%
""" Shared data and the shipped nominal model. """

PCA_STATE = octmnist_pca.fit_pca()
DRUSEN_TRAIN, DRUSEN_TEST, CLEAN_TRAIN, CLEAN_TEST = octmnist_pca.get_datasets(PCA_DIMS, PCA_STATE)
MODEL = octmnist_pca.get_pretrained_model(PCA_DIMS, PCA_STATE)

# %%
""" Panel data 1: pre-training radius against nominal quality and certified tightness. """

pt_rows = []
for pt_eps in tqdm.tqdm(PT_EPSILONS, desc="pre-train radius", file=sys.stderr):
    model = octmnist_pca.get_pretrained_model(PCA_DIMS, PCA_STATE, epsilon=pt_eps)
    bounded, _ = octmnist_pca.run_certified(
        make_config(0.01, None), PCA_DIMS, model, DRUSEN_TRAIN, CLEAN_TRAIN,
        tag="ptsel", pretrain_id=octmnist_pca.pretrain_tag(epsilon=pt_eps),
    )
    row = metrics(bounded, DRUSEN_TEST.tensors)
    row["pt_eps"] = pt_eps
    row["clean_acc"] = agt.test_metrics.test_accuracy(bounded, *CLEAN_TEST.tensors, epsilon=0)[1]
    pt_rows.append(row)

print(f"\n  model selection at the shipped schedule (eps=0.01, k={K_POISON}):", file=sys.stderr)
print(f"  {'pt_eps':>7s} {'box width':>11s} {'cert acc':>9s} {'nom acc':>8s} {'clean':>7s}", file=sys.stderr)
for r in pt_rows:
    print(f"  {r['pt_eps']:>7} {r['box_width']:>11.4e} {r['cert_acc']:>9.4f} "
          f"{r['nominal_acc']:>8.4f} {r['clean_acc']:>7.4f}", file=sys.stderr)

# %%
""" Panel data 2: the poisoning radius, baseline against refined. """

eps_rows = []
for epsilon in tqdm.tqdm(EPSILONS, desc="poisoning radius", file=sys.stderr):
    base, _ = octmnist_pca.run_certified(
        make_config(epsilon, None), PCA_DIMS, MODEL, DRUSEN_TRAIN, CLEAN_TRAIN, tag="eps")
    refined, _ = octmnist_pca.run_certified(
        make_config(epsilon, (N_SPLITS, REFINE_DIMS)), PCA_DIMS, MODEL, DRUSEN_TRAIN, CLEAN_TRAIN,
        tag="eps")
    eps_rows.append({
        "epsilon": epsilon,
        "baseline": metrics(base, DRUSEN_TEST.tensors),
        "refined": metrics(refined, DRUSEN_TEST.tensors),
    })

# %%
""" Panel data 3: the refinement budget at a fixed radius. """

leaf_base, _ = octmnist_pca.run_certified(
    make_config(LEAF_EPSILON, None), PCA_DIMS, MODEL, DRUSEN_TRAIN, CLEAN_TRAIN, tag="leaf")
leaf_base_m = metrics(leaf_base, DRUSEN_TEST.tensors)

leaf_rows = []
for n_splits, n_dims in tqdm.tqdm(LEAF_SCHEDULE, desc="refinement budget", file=sys.stderr):
    refined, _ = octmnist_pca.run_certified(
        make_config(LEAF_EPSILON, (n_splits, n_dims)), PCA_DIMS, MODEL, DRUSEN_TRAIN, CLEAN_TRAIN,
        tag="leaf")
    row = metrics(refined, DRUSEN_TEST.tensors)
    row.update({"n_splits": n_splits, "n_dims": n_dims, "leaves": n_splits**n_dims})
    leaf_rows.append(row)

BISECT = [r for r in leaf_rows if r["n_splits"] == 2]
EQUAL_COST = [r for r in leaf_rows if r["n_splits"] != 2]

# %%
""" The figure. """

C = script_utils.colours
subplots = (3, 2)
fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300)
(ax_pt, ax_ptbox), (ax_eps, ax_gain), (ax_leaf, ax_leafce) = axs

# --- row 1: model selection -------------------------------------------------------------------
# Past some radius the regularizer zeroes the network: the box collapses and certified accuracy
# rises to 1.0 purely because the model predicts one class. Shade that region, or the two panels
# read as "more robust pre-training is strictly better".
pt = [r["pt_eps"] for r in pt_rows]
collapsed = [r["pt_eps"] for r in pt_rows if r["clean_acc"] < TRIVIAL_CLEAN + 0.02]
span_from = min(collapsed) / 1.4 if collapsed else None
ax_pt.plot(pt, [r["nominal_acc"] for r in pt_rows], marker="o", color=C["green"],
           label="nominal Drusen")
ax_pt.plot(pt, [r["cert_acc"] for r in pt_rows], marker="s", color=C["purple"],
           label="certified Drusen")
ax_pt.plot(pt, [r["clean_acc"] for r in pt_rows], marker="^", color=C["orange"], label="clean split")
ax_pt.axhline(TRIVIAL_CLEAN, linestyle=":", color=C["grey"], label="degenerate predictor")
ax_pt.set_xscale("log")
# start at zero: at small radii the certificate is vacuous (certified accuracy 0.0), which is as
# much of the story as the collapse at large radii
ax_pt.set_ylim(0, 1.08)
ax_pt.set_xlabel(r"pre-training radius $\epsilon_{pre}$")
ax_pt.set_ylabel("accuracy")
ax_pt.legend(fontsize="xx-small", ncol=2, loc="lower center")

ax_ptbox.plot(pt, [r["box_width"] for r in pt_rows], marker="o", color=C["green"])
ax_ptbox.set_xscale("log")
ax_ptbox.set_yscale("log")
ax_ptbox.set_xlabel(r"pre-training radius $\epsilon_{pre}$")
ax_ptbox.set_ylabel("certified box width")

for ax in (ax_pt, ax_ptbox):
    ax.axvline(octmnist_pca.PT_EPSILON, linestyle="--", lw=0.8, color=C["pink"])
    ax.annotate("shipped", (octmnist_pca.PT_EPSILON, 1.0), xycoords=("data", "axes fraction"),
                textcoords="offset points", xytext=(-2, -8), fontsize="xx-small",
                color=C["pink"], ha="right")
    if span_from is not None:
        ax.axvspan(span_from, max(pt) * 1.3, color=C["grey"], alpha=0.18, lw=0)
        ax.annotate("network\ncollapsed", (max(pt), 0.42), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(-2, 0), fontsize="xx-small",
                    color=C["grey"], ha="right", va="center")
    ax.set_xlim(min(pt) / 1.3, max(pt) * 1.3)

# --- row 2: the poisoning radius ---------------------------------------------------------------
eps = [r["epsilon"] for r in eps_rows]
ax_eps.plot(eps, [r["baseline"]["cert_ce"] for r in eps_rows], marker="s", linestyle=":",
            color=C["grey"], label="baseline")
ax_eps.plot(eps, [r["refined"]["cert_ce"] for r in eps_rows], marker="o", color=C["green"],
            label="refined")
ax_eps.axhline(eps_rows[0]["baseline"]["nominal_ce"], linestyle="--", color=C["orange"],
               label="nominal (floor)")
ax_eps.set_xscale("log")
ax_eps.set_xlabel(r"poisoning radius $\epsilon$")
ax_eps.set_ylabel("certified cross-entropy")
ax_eps.legend(fontsize="xx-small")

ax_gain.plot(eps, [1 - r["refined"]["box_width"] / r["baseline"]["box_width"] for r in eps_rows],
             marker="o", color=C["green"])
ax_gain.set_xscale("log")
ax_gain.set_ylim(0, None)
ax_gain.set_xlabel(r"poisoning radius $\epsilon$")
ax_gain.set_ylabel("box width reduction")

# --- row 3: the refinement budget --------------------------------------------------------------
for ax, key, ylabel in ((ax_leaf, "box_width", "certified box width"),
                        (ax_leafce, "cert_ce", "certified cross-entropy")):
    ax.axhline(leaf_base_m[key], linestyle=":", color=C["grey"], label="baseline")
    if key == "cert_ce":
        ax.axhline(leaf_base_m["nominal_ce"], linestyle="--", color=C["purple"], label="nominal")
    ax.plot([r["leaves"] for r in BISECT], [r[key] for r in BISECT], marker="o", color=C["green"],
            label=r"bisect ($n_{splits}=2$)")
    ax.scatter([r["leaves"] for r in EQUAL_COST], [r[key] for r in EQUAL_COST], marker="^",
               color=C["orange"], zorder=3, label="fewer, finer dims")
    for r in EQUAL_COST:  # label to the left of the rightmost point so it is not clipped
        rightmost = r["leaves"] == max(e["leaves"] for e in EQUAL_COST)
        ax.annotate(rf"${r['n_splits']}^{{{r['n_dims']}}}$", (r["leaves"], r[key]),
                    textcoords="offset points", xytext=(-6 if rightmost else 5, -9),
                    ha="right" if rightmost else "left", fontsize="xx-small", color=C["orange"])
    ax.set_xscale("log")
    ax.set_xlabel(rf"leaves per fragment ($\epsilon={LEAF_EPSILON}$)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize="xx-small")

fig.suptitle(
    rf"OCT-MNIST on PCA features ($d={PCA_DIMS}$, $k={K_POISON}$): model selection, "
    rf"threat model, refinement budget", fontsize="small")
script_utils.apply_figure_size(fig, script_utils.set_size(1.0, subplots, shrink_height=0.85), dpi=300)
fig_dir = script_utils.make_dirs()[3]
path = f"{fig_dir}/octmnist_pca_summary.pdf"
plt.savefig(path, dpi=300)

# %%
""" Console summary of what the figure shows. """

print(f"\n  nominal model (d={PCA_DIMS}): Drusen {eps_rows[0]['baseline']['nominal_acc']:.4f}, "
      f"cross-entropy {eps_rows[0]['baseline']['nominal_ce']:.4f} | conv reference "
      f"{CONV_REFERENCE['drusen']:.4f}", file=sys.stderr)
print(f"  refinement box-width reduction: "
      f"{1 - eps_rows[0]['refined']['box_width'] / eps_rows[0]['baseline']['box_width']:.2%} at "
      f"eps={EPSILONS[0]} rising to "
      f"{1 - eps_rows[-1]['refined']['box_width'] / eps_rows[-1]['baseline']['box_width']:.2%} at "
      f"eps={EPSILONS[-1]}", file=sys.stderr)
print(f"  refinement budget at eps={LEAF_EPSILON}: "
      f"{1 - BISECT[0]['box_width'] / leaf_base_m['box_width']:.2%} at {BISECT[0]['leaves']} leaves "
      f"to {1 - BISECT[-1]['box_width'] / leaf_base_m['box_width']:.2%} at {BISECT[-1]['leaves']}",
      file=sys.stderr)
print(f"\n  figure: {path}", file=sys.stderr)
