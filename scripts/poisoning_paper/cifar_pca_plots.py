"""
Summary figure for the CIFAR-10 feature-poisoning experiments: one 2x3 panel drawn from the
threat-grid, refinement-depth and feature-width experiments. It tests no hypothesis of its own.
The hypotheses are stated in cifar_pca_threat_sweep.py, cifar_pca_leaf_sweep.py and
cifar_pca_dims_sweep.py; this script picks the views that together carry the overall conclusion.

    Row 1, the threat grid at d=20: how certified accuracy degrades with ``k`` and ``eps``, where
        refinement pays, and how far bisecting every coordinate falls short of the eps/2 reference.
    Row 2, the compute budget: how the gain grows with the number of leaves, how it holds as the
        feature width grows, and what each refined run bought for its GPU cost.

Every number is recomputed from the cached runs rather than restated, so the figure cannot drift from
the results. It never trains, and exits listing any missing runs. The feature-width panel carries the
pre-training-radius caveat described in cifar_pca_dims_sweep.py, together with the outcome of
cifar_pca_pt_epsilon_control.py, the control that resolves it.

Key external dependencies: matplotlib, and the sibling modules cifar_pca, cifar_pca_manifest and
script_utils.
"""

import sys

import matplotlib.pyplot as plt

import cifar_pca
import cifar_pca_manifest as manifest
import script_utils
from cifar_pca_manifest import (E2_D, E2_EPS, E2_K, E2_SCHEDULE, E3_DIMS, EPSILONS, K_POISONS,
                                SCREEN_SPLIT_DIMS, Run, collect)

D = E2_D
SCREEN, FULL = (2, SCREEN_SPLIT_DIMS), (2, D)

RESULTS = collect(manifest.all_runs())


def cell(d, k, eps, refine=None):
    """The reported result of one run, by its experiment coordinates."""
    return RESULTS[Run(d, k, eps, refine)]


def refined_reduction(d, k, eps, refine):
    """Relative box-width reduction of a refined run against the unrefined run at the same cell."""
    return 1 - cell(d, k, eps, refine)["test"]["box_width"] / cell(d, k, eps)["test"]["box_width"]


def half_eps_reduction(d, k, eps):
    """
    Relative width reduction of the unrefined eps/2 run against the unrefined run at eps. Each
    bisect-all leaf is an eps/2 cube, so this indicates how tight a single leaf is: a heuristic
    reference for what refinement could recover, not a bound (design §6, H2b).
    """
    return 1 - cell(d, k, eps / 2)["test"]["box_width"] / cell(d, k, eps)["test"]["box_width"]


def gpu_hours(result):
    return result["seconds"] * result["world_size"] / 3600


# %%
""" The figure. """

C = script_utils.colours
K_COLOURS = [script_utils.sequential_colours[i] for i in (4, 6, 8, 10, 11)]
subplots = (2, 3)
fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300)
(ax_acc, ax_gain, ax_ceiling), (ax_depth, ax_width, ax_cost) = axs

# --- row 1: the threat grid --------------------------------------------------------------------
nominal = cell(D, K_POISONS[0], EPSILONS[0])["test"]
for colour, k in zip(K_COLOURS, K_POISONS):
    ax_acc.plot(EPSILONS, [cell(D, k, e, FULL)["test"]["cert_acc"] for e in EPSILONS],
                marker="o", ms=2.5, color=colour, label=f"$k={k}$")
ax_acc.axhline(nominal["nominal_acc"], linestyle="--", color=C["orange"], label="nominal")
ax_acc.set_xscale("log")
ax_acc.set_ylim(0, 1)
ax_acc.set_xlabel(r"poisoning radius $\epsilon$")
ax_acc.set_ylabel("certified held-out accuracy")
ax_acc.legend(fontsize="xx-small", ncol=2)

for colour, k in zip(K_COLOURS, K_POISONS):
    ax_gain.plot(EPSILONS, [refined_reduction(D, k, e, FULL) for e in EPSILONS],
                 marker="o", ms=2.5, color=colour, label=f"$k={k}$")
    ax_gain.plot(EPSILONS, [refined_reduction(D, k, e, SCREEN) for e in EPSILONS],
                 marker="^", ms=2.5, linestyle="--", color=colour)
ax_gain.set_xscale("log")
ax_gain.set_ylim(0, None)
ax_gain.set_xlabel(r"poisoning radius $\epsilon$")
ax_gain.set_ylabel("box width reduction")
ax_gain.legend(fontsize="xx-small", title=rf"solid: all {D}, dashed: top {SCREEN_SPLIT_DIMS}",
               title_fontsize="xx-small")

ceiling = [half_eps_reduction(D, k, e) for k in K_POISONS for e in EPSILONS]
achieved = [refined_reduction(D, k, e, FULL) for k in K_POISONS for e in EPSILONS]
cell_colours = [K_COLOURS[K_POISONS.index(k)] for k in K_POISONS for _ in EPSILONS]
top = max(ceiling + achieved) * 1.05
ax_ceiling.scatter(ceiling, achieved, s=6, c=cell_colours, zorder=3)
ax_ceiling.plot([0, top], [0, top], linestyle=":", color=C["grey"], label="$y=x$")
ax_ceiling.set_xlim(0, top)
ax_ceiling.set_ylim(0, top)
ax_ceiling.set_xlabel(r"reduction of the unrefined $\epsilon/2$ run")
ax_ceiling.set_ylabel(rf"reduction bisecting all {D}")
ax_ceiling.legend(fontsize="xx-small")

# --- row 2: the budget axes --------------------------------------------------------------------
bisect = [(n, d) for n, d in sorted(E2_SCHEDULE, key=lambda r: r[0] ** r[1]) if n == 2]
equal_cost = [(n, d) for n, d in E2_SCHEDULE if n != 2]
ax_depth.plot([n**d for n, d in bisect], [refined_reduction(D, E2_K, E2_EPS, r) for r in bisect],
              marker="o", color=C["green"], label=r"bisect ($n_{splits}=2$)")
ax_depth.scatter([n**d for n, d in equal_cost],
                 [refined_reduction(D, E2_K, E2_EPS, r) for r in equal_cost],
                 marker="^", color=C["orange"], zorder=3, label="fewer, finer dims")
for n, d in equal_cost:
    ax_depth.annotate(rf"${n}^{{{d}}}$", (n**d, refined_reduction(D, E2_K, E2_EPS, (n, d))),
                      textcoords="offset points", xytext=(-5, 4), ha="right",
                      fontsize="xx-small", color=C["orange"])
ax_depth.set_xscale("log", base=2)
ax_depth.set_ylim(0, None)
ax_depth.set_xlabel("leaves per fragment")
ax_depth.set_ylabel("box width reduction")
ax_depth.set_title(rf"$k={E2_K}$, $\epsilon={E2_EPS}$", fontsize="xx-small")
ax_depth.legend(fontsize="xx-small")

ax_width.plot(E3_DIMS, [refined_reduction(d, E2_K, E2_EPS, (2, d)) for d in E3_DIMS],
              marker="o", color=C["green"], label="bisect all $d$")
ax_width.plot(E3_DIMS, [refined_reduction(d, E2_K, E2_EPS, SCREEN) for d in E3_DIMS],
              marker="^", linestyle="--", color=C["purple"], label=f"bisect top {SCREEN_SPLIT_DIMS}")
ax_width.plot(E3_DIMS, [half_eps_reduction(d, E2_K, E2_EPS) for d in E3_DIMS],
              marker="x", linestyle=":", color=C["grey"], label=r"unrefined at $\epsilon/2$")
ax_width.set_xticks(E3_DIMS)
ax_width.set_ylim(0, None)
ax_width.set_xlabel("PCA components $d$")
ax_width.set_ylabel("box width reduction")
ax_width.set_title("per-$d$ selection: see caveat", fontsize="xx-small")
ax_width.legend(fontsize="xx-small")

# every refined run in the campaign, against what it cost: the gain is small and bought dearly
for colour, d in zip((C["green"], C["purple"], C["pink"]), E3_DIMS):
    runs = [r for r in RESULTS if r.d == d and r.refine is not None]
    if not runs:
        continue
    ax_cost.scatter([gpu_hours(RESULTS[r]) for r in runs],
                    [refined_reduction(r.d, r.k, r.eps, r.refine) for r in runs],
                    s=6, color=colour, zorder=3, label=f"$d={d}$")
ax_cost.set_xscale("log")
ax_cost.set_ylim(0, None)
ax_cost.set_xlabel("GPU-hours of the refined run")
ax_cost.set_ylabel("box width reduction")
ax_cost.legend(fontsize="xx-small")

fig.suptitle(rf"CIFAR-10 on PCA features: threat grid, refinement depth and feature width "
             rf"($d={D}$ unless shown), test split", fontsize="small")
script_utils.apply_figure_size(fig, script_utils.set_size(1.0, subplots, shrink_height=1.35), dpi=300)
_, _, _, fig_dir = cifar_pca.dirs()
path = f"{fig_dir}/cifar_pca_summary.pdf"
plt.savefig(path, dpi=300)

# %%
""" Console summary of what the figure shows. """

worst = cell(D, K_POISONS[-1], EPSILONS[-1])["test"]
print(f"\n  nominal model (d={D}): held-out {nominal['nominal_acc']:.4f}, clean "
      f"{nominal['nominal_clean_acc']:.4f}, cross-entropy {nominal['nominal_ce']:.4f}", file=sys.stderr)
print(f"  certified held-out accuracy spans {cell(D, K_POISONS[0], EPSILONS[0], FULL)['test']['cert_acc']:.4f} "
      f"(k={K_POISONS[0]}, eps={EPSILONS[0]}) to {cell(D, K_POISONS[-1], EPSILONS[-1], FULL)['test']['cert_acc']:.4f} "
      f"(k={K_POISONS[-1]}, eps={EPSILONS[-1]}); certified clean falls to {worst['cert_clean_acc']:.4f} there",
      file=sys.stderr)
print(f"  bisect-all width reduction spans {min(achieved):.2%} to {max(achieved):.2%}, against an "
      f"eps/2 reference of {min(ceiling):.2%} to {max(ceiling):.2%}", file=sys.stderr)
print(f"  refinement depth at k={E2_K}, eps={E2_EPS}: "
      f"{refined_reduction(D, E2_K, E2_EPS, bisect[0]):.2%} at {bisect[0][0]**bisect[0][1]} leaves to "
      f"{refined_reduction(D, E2_K, E2_EPS, bisect[-1]):.2%} at {bisect[-1][0]**bisect[-1][1]}", file=sys.stderr)
for pair in ((16, 3), (2, 12)), ((4, 8), (2, 16)), ((4, 10), (2, 20)):
    finer, wider = pair
    if finer in E2_SCHEDULE and wider in E2_SCHEDULE:
        f_red, w_red = (refined_reduction(D, E2_K, E2_EPS, r) for r in (finer, wider))
        print(f"    equal cost {finer[0]}^{finer[1]} {f_red:.2%} vs {wider[0]}^{wider[1]} {w_red:.2%}: "
              f"{'wider' if w_red > f_red else 'finer'} wins by {abs(w_red - f_red) * 100:.2f} points",
              file=sys.stderr)
print(f"  feature width: bisect-all reduction "
      + ", ".join(f"d={d} {refined_reduction(d, E2_K, E2_EPS, (2, d)):.2%}" for d in E3_DIMS), file=sys.stderr)
print("  CAVEAT: E0 selected pt_epsilon="
      + ", ".join(f"{cifar_pca.selected_hyperparameters(d).pt_epsilon} at d={d}" for d in E3_DIMS)
      + ", so the baseline widths are not comparable across d. At a fixed pt_epsilon, the baseline "
      "widens with d and the bisect-all gain holds; see cifar_pca_pt_epsilon_control.py",
      file=sys.stderr)
print(f"\n  figure: {path}", file=sys.stderr)
