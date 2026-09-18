# %%
"""
Refinement-depth experiment on OCT-MNIST PCA features: how far refinement can be pushed, and how best
to spend a fixed number of leaves.

Hypotheses:
    H1  Returns diminish but do not vanish. The gain keeps improving up to the largest affordable
        partition, bisecting all 16 coordinates.
    H2  At equal leaf count, bisecting more coordinates beats cutting fewer coordinates more finely.
        This is the "bisect every dimension before raising n_splits" priority that the other sweeps
        assume.

Mechanism. One attack at d=16: k=50, eps=0.1, the radius at which the epsilon sweep found refinement
paying most. The unrefined run is compared against:
    - a bisection ladder: the 4, 8, 12 and 16 most sensitive coordinates bisected (2^4 to 2^16
      leaves);
    - equal-cost alternatives: 8^4 against 2^12 (4096 leaves), and 4^8 against 2^16 (65536 leaves).
Both certified accuracy and certified cross-entropy are reported. Cross-entropy is the finer signal:
accuracy only moves when a test point's bound crosses the decision boundary, so it can sit flat
while the certificate keeps tightening.

Recorded result (Isambard, 2026-09). H1 holds. The box narrows by 3.41% at 16 leaves, 7.67% at 256,
9.25% at 4096 and 10.74% at 65536. Certified accuracy barely moves (0.852 to 0.860) while the
certified loss keeps falling. H2 partly holds. At 4096 leaves, bisecting 12 coordinates beats an
8-way split of 4 (9.25% against 6.10%). At 65536 leaves, once every coordinate can be bisected, a
4-way split of the 8 most sensitive is ahead (11.37% against 10.74%) for the same certified loss.

Output: a table on stderr, and a figure of box width, certified cross-entropy and certified Drusen
accuracy against leaf count, with the ladder as a line and the alternatives as labelled points.

Key external dependencies: torch, matplotlib, ``abstract_gradient_training``, and the sibling
modules ``octmnist_pca`` and ``script_utils``.
"""

import copy
import sys
import time

import torch
import torch.utils.data
import tqdm
import matplotlib.pyplot as plt

import abstract_gradient_training as agt

import octmnist_pca
import script_utils

# %%
""" Script parameters. """

PCA_DIMS = 16
K_POISON = 50
EPSILON = 0.1  # the radius at which the epsilon sweep found refinement paying most (8.5% at d=16)

# (n_splits, n_dims) rungs. The first block is the stated priority -- bisect more dimensions, at
# n_splits=2 throughout -- up to bisecting all 16. The last two entries are equal-cost alternatives
# that spend the same leaves on fewer, more finely cut dimensions, to test that priority.
LEAF_SCHEDULE = [(2, 4), (2, 8), (2, 12), (2, 16), (8, 4), (4, 8)]
STRATEGY = "sensitivity"
MAX_LEAVES = 10_000_000
LEAF_CHUNK = 64

DEVICE = octmnist_pca.FINETUNE_CONFIG.device
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    DEVICE = "cpu"


def make_config(refine):
    """Tuned config at the fixed threat model, refined when ``refine`` is an ``(n_splits, n_dims)``
    pair (``None`` for the unrefined baseline)."""
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
    """Certified box width, Drusen accuracy and Drusen cross-entropy over the parameter box."""
    acc_worst, acc_nominal, _ = agt.test_metrics.test_accuracy(bounded_model, *test_tensors, epsilon=0)
    ce_worst, ce_nominal, _ = agt.test_metrics.test_cross_entropy(bounded_model, *test_tensors, epsilon=0)
    box_width = sum((u - l).sum().item() for l, u in zip(bounded_model.param_l, bounded_model.param_u))
    return {
        "cert_acc": acc_worst, "nominal_acc": acc_nominal,
        "cert_ce": ce_worst, "nominal_ce": ce_nominal,
        "box_width": box_width,
    }


# %%
""" Build the shared dataset and the tuned nominal model. Both are refinement-independent. """

PCA_STATE = octmnist_pca.fit_pca()
DRUSEN_TRAIN, DRUSEN_TEST, CLEAN_TRAIN, CLEAN_TEST = octmnist_pca.get_datasets(PCA_DIMS, PCA_STATE)
MODEL = octmnist_pca.get_pretrained_model(PCA_DIMS, PCA_STATE)

# %%
""" Sweep: the unrefined baseline, then each rung of the partition schedule. """

print(
    f"OCT-MNIST (PCA features, d={PCA_DIMS}) | eps={EPSILON} | k_poison={K_POISON} | "
    f"strategy={STRATEGY}\n",
    file=sys.stderr,
)
print(
    f"  {'split':>7s} {'leaves':>7s}  {'box width':>11s} {'width':>7s}  {'cert acc':>8s} {'acc':>7s}  "
    f"{'cert CE':>8s} {'CE':>7s}  {'time':>7s}  {'viol':>8s}",
    file=sys.stderr,
)
print(
    f"  {'':>7s} {'':>7s}  {'':>11s} {'gain':>7s}  {'':>8s} {'gain':>7s}  {'':>8s} {'gain':>7s}",
    file=sys.stderr,
)

start = time.time()
base_model, base_viol = octmnist_pca.run_certified(
    make_config(None), PCA_DIMS, MODEL, DRUSEN_TRAIN, CLEAN_TRAIN, tag="leaf"
)
base = certified_metrics(base_model, DRUSEN_TEST.tensors)
print(
    f"  {'none':>7s} {1:>7d}  {base['box_width']:11.4e} {'--':>7s}  {base['cert_acc']:8.4f} {'--':>7s}  "
    f"{base['cert_ce']:8.4f} {'--':>7s}  {time.time() - start:6.1f}s  {base_viol:8.1e}",
    file=sys.stderr,
)

rows = []
for n_splits, n_dims in tqdm.tqdm(LEAF_SCHEDULE, desc="partition", file=sys.stderr):
    start = time.time()
    refined_model, refined_viol = octmnist_pca.run_certified(
        make_config((n_splits, n_dims)), PCA_DIMS, MODEL, DRUSEN_TRAIN, CLEAN_TRAIN, tag="leaf"
    )
    elapsed = time.time() - start
    refined = certified_metrics(refined_model, DRUSEN_TEST.tensors)

    # the nominal trajectory is plain SGD, so refinement cannot move it
    assert abs(base["nominal_acc"] - refined["nominal_acc"]) < 1e-9, (
        f"nominal accuracy diverged at {n_splits=}, {n_dims=}"
    )

    leaves = n_splits**n_dims
    rows.append({
        "n_splits": n_splits, "n_dims": n_dims, "leaves": leaves,
        "baseline": base, "refined": refined, "elapsed": elapsed,
        "width_gain": 1 - refined["box_width"] / base["box_width"] if base["box_width"] else 0.0,
        "acc_gain": refined["cert_acc"] - base["cert_acc"],
        "ce_gain": base["cert_ce"] - refined["cert_ce"],  # loss: lower is tighter
        "violation": max(base_viol, refined_viol),
    })
    print(
        f"  {f'{n_splits}^{n_dims}':>7s} {leaves:>7d}  {refined['box_width']:11.4e} "
        f"{rows[-1]['width_gain']:6.2%}  {refined['cert_acc']:8.4f} {rows[-1]['acc_gain']:+7.4f}  "
        f"{refined['cert_ce']:8.4f} {rows[-1]['ce_gain']:+7.4f}  {elapsed:6.1f}s  "
        f"{rows[-1]['violation']:8.1e}",
        file=sys.stderr,
    )

print(
    f"\n  nominal model: Drusen accuracy {base['nominal_acc']:.4f}, cross-entropy "
    f"{base['nominal_ce']:.4f}; certified cross-entropy is bounded below by the nominal one.",
    file=sys.stderr,
)
unsound = [(r["n_splits"], r["n_dims"]) for r in rows if r["violation"] > 1e-3]
if unsound:
    print(f"  WARNING: ERROR-level bound violations at {unsound}; those rungs are unsound.",
          file=sys.stderr)

# %%
""" Plot: how the certified box and certified loss tighten as the partition grows. """

bisect = [r for r in rows if r["n_splits"] == 2]
equal_cost = [r for r in rows if r["n_splits"] != 2]
subplots = (1, 3)
fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300)

axs[0].axhline(base["box_width"], linestyle=":", color=script_utils.colours["grey"],
               label="baseline (no refinement)")
axs[0].plot([r["leaves"] for r in bisect], [r["refined"]["box_width"] for r in bisect],
            marker="o", color=script_utils.colours["green"], label=r"bisect ($n_{splits}=2$)")
axs[0].scatter([r["leaves"] for r in equal_cost], [r["refined"]["box_width"] for r in equal_cost],
               marker="^", color=script_utils.colours["orange"], zorder=3, label="fewer, finer dims")
axs[0].set_xscale("log")
axs[0].set_xlabel("leaves per fragment")
axs[0].set_ylabel("certified box width")
axs[0].legend(fontsize="xx-small")

axs[1].axhline(base["cert_ce"], linestyle=":", color=script_utils.colours["grey"],
               label="baseline (no refinement)")
axs[1].plot([r["leaves"] for r in bisect], [r["refined"]["cert_ce"] for r in bisect],
            marker="o", color=script_utils.colours["green"], label=r"bisect ($n_{splits}=2$)")
axs[1].scatter([r["leaves"] for r in equal_cost], [r["refined"]["cert_ce"] for r in equal_cost],
               marker="^", color=script_utils.colours["orange"], zorder=3, label="fewer, finer dims")
axs[1].axhline(base["nominal_ce"], linestyle="--", color=script_utils.colours["purple"],
               label="nominal (unreachable floor)")
axs[1].set_xscale("log")
axs[1].set_xlabel("leaves per fragment")
axs[1].set_ylabel("certified cross-entropy")
axs[1].legend(fontsize="xx-small")

axs[2].axhline(base["cert_acc"], linestyle=":", color=script_utils.colours["grey"],
               label="baseline (no refinement)")
axs[2].plot([r["leaves"] for r in bisect], [r["refined"]["cert_acc"] for r in bisect],
            marker="o", color=script_utils.colours["green"], label=r"bisect ($n_{splits}=2$)")
axs[2].scatter([r["leaves"] for r in equal_cost], [r["refined"]["cert_acc"] for r in equal_cost],
               marker="^", color=script_utils.colours["orange"], zorder=3, label="fewer, finer dims")
axs[2].axhline(base["nominal_acc"], linestyle="--", color=script_utils.colours["purple"],
               label="nominal")
axs[2].set_xscale("log")
axs[2].set_xlabel("leaves per fragment")
axs[2].set_ylabel("certified Drusen accuracy")
axs[2].legend(fontsize="xx-small")

fig.suptitle(rf"OCT-MNIST on PCA features ($d={PCA_DIMS}$): refinement depth, "
             rf"$k={K_POISON}$, $\epsilon={EPSILON}$", fontsize="small")
script_utils.apply_figure_size(fig, script_utils.set_size(1.0, subplots, shrink_height=2.4), dpi=300)
fig_dir = script_utils.make_dirs()[3]
path = f"{fig_dir}/octmnist_pca_leaf_sweep.pdf"
plt.savefig(path, dpi=300)
print(f"\n  figure: {path}", file=sys.stderr)
