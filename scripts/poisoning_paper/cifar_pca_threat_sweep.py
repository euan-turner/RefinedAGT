"""
Threat-grid experiment: how the certificate depends on the strength of the attack, and where input
refinement helps.

Hypotheses:
    H1  The certificate stays informative across the grid (certified held-out accuracy well above
        chance), and the number of poisoned samples ``k`` degrades it more than the radius ``eps``.
    H2  Refinement tightens the certificate only modestly. Bisecting every feature coordinate reduces
        box width by roughly 5-7% and raises certified accuracy by at most about 0.02.
    H2b Bisecting every coordinate still falls well short of an unrefined run at eps/2. Each leaf of
        the full bisection is an eps/2-radius cube, so the eps/2 run is a rough reference for how tight
        a single leaf is. The shortfall is what bounding leaves separately and combining them loses.
    H3  The refinement gain peaks at moderate ``eps`` (0.02-0.05) and falls at 0.1.
    H4  The refinement gain grows with ``k``.

Mechanism. At d=20, for every cell of k in {10, 25, 50, 100, 200} x eps in {0.005, 0.01, 0.02, 0.05,
0.1}, it compares four cached runs:
    - unrefined (the baseline);
    - the 12 most sensitive coordinates bisected (2^12 leaves);
    - all 20 coordinates bisected (2^20 leaves);
    - unrefined at eps/2 (the H2b reference).
Every run uses the same selected hyperparameters, so the nominal model is identical across the grid;
the script asserts this. The refinement gain at a cell is the relative reduction in box width against
the baseline at that cell. All metrics are on the test split.

Output:
    - a per-cell table on stderr;
    - ``cifar_pca_threat_sweep.json``;
    - a four-panel figure:
        - baseline certified accuracy over the grid (H1);
        - the accuracy gained by bisecting all 20 coordinates (H2);
        - box-width reduction against ``eps``, one line per ``k`` (H2-H4);
        - bisect-all reduction against the eps/2 reduction (H2b: points below y = x fall short).

Reads cached runs only, and exits listing any that are missing. The runs are produced by
cifar_pca_run.py through the Slurm jobs.

Key external dependencies: matplotlib, and the sibling modules cifar_pca, cifar_pca_manifest and
script_utils.
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

import cifar_pca
import script_utils
from cifar_pca_manifest import EPSILONS, K_POISONS, SCREEN_SPLIT_DIMS, Run, collect, e1_runs

D = 20
SCREEN, FULL = (2, SCREEN_SPLIT_DIMS), (2, D)

results = collect(e1_runs())


def test(k, eps, refine=None):
    return results[Run(D, k, eps, refine)]["test"]


def width_gain(k, eps, refined):
    """Relative box-width reduction of ``refined`` against the unrefined run at the same cell."""
    return 1 - refined["box_width"] / test(k, eps)["box_width"]


# the nominal trajectory is plain SGD, independent of the attack and of refinement
nominal = test(K_POISONS[0], EPSILONS[0])
for run, result in results.items():
    assert abs(result["test"]["nominal_acc"] - nominal["nominal_acc"]) < 1e-9, f"nominal model moved at {run}"

# %%
print(f"\nE1 | CIFAR-10 PCA d={D} | nominal held-out test acc {nominal['nominal_acc']:.4f}, clean "
      f"{nominal['nominal_clean_acc']:.4f} | test split", file=sys.stderr)
print(f"  {'k':>4s} {'eps':>6s}  {'box':>9s}  {'width reduction':>23s}  {'certified held-out acc':>26s}  "
      f"{'certified CE':>17s}  {'cert':>6s}  {'full':>6s} {'viol':>7s}", file=sys.stderr)
print(f"  {'':>4s} {'':>6s}  {'(base)':>9s}  {'top12':>7s} {'all20':>7s} {'eps/2':>7s}  {'base':>8s} {'top12':>8s} "
      f"{'all20':>8s}  {'base':>8s} {'all20':>8s}  {'clean':>6s}  {'hours':>6s}", file=sys.stderr)
rows = []
for k in K_POISONS:
    for eps in EPSILONS:
        base, screen, full, half = test(k, eps), test(k, eps, SCREEN), test(k, eps, FULL), test(k, eps / 2)
        full_record = results[Run(D, k, eps, FULL)]
        violation = max(results[Run(D, k, e, r)]["violation"] for e, r in ((eps, None), (eps, SCREEN), (eps, FULL)))
        row = {"k": k, "eps": eps, "base": base, "screen": screen, "full": full, "half_eps": half,
               "full_seconds": full_record["seconds"], "full_world_size": full_record["world_size"],
               "violation": violation}
        rows.append(row)
        print(f"  {k:>4d} {eps:>6}  {base['box_width']:9.3e}  {width_gain(k, eps, screen):7.2%} "
              f"{width_gain(k, eps, full):7.2%} {width_gain(k, eps, half):7.2%}  {base['cert_acc']:8.4f} "
              f"{screen['cert_acc']:8.4f} {full['cert_acc']:8.4f}  {base['cert_ce']:8.4f} {full['cert_ce']:8.4f}  "
              f"{base['cert_clean_acc']:6.4f}  {full_record['seconds'] / 3600:6.2f} {violation:7.1e}",
              file=sys.stderr)

results_dir, _, _, fig_dir = cifar_pca.dirs()
with open(f"{results_dir}/cifar_pca_threat_sweep.json", "w") as file:
    json.dump(rows, file, indent=1)

# %%
C = script_utils.colours
K_COLOURS = [script_utils.sequential_colours[i] for i in (4, 6, 8, 10, 11)]
subplots = (2, 2)
fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300)
(ax_base, ax_delta), (ax_gain, ax_ceiling) = axs


def heatmap(ax, values, title, fmt):
    """``values[i][j]`` at ``K_POISONS[i]``, ``EPSILONS[j]``, annotated."""
    image = ax.imshow(values, cmap="Greens", aspect="auto", origin="lower")
    for i, j in np.ndindex(values.shape):
        ax.text(j, i, format(values[i, j], fmt), ha="center", va="center", fontsize="xx-small")
    ax.set_xticks(range(len(EPSILONS)), EPSILONS, fontsize="x-small")
    ax.set_yticks(range(len(K_POISONS)), K_POISONS, fontsize="x-small")
    ax.set_xlabel(r"poisoning radius $\epsilon$")
    ax.set_ylabel("poisoned per batch $k$")
    ax.set_title(title, fontsize="small")
    return image


heatmap(ax_base, np.array([[test(k, e)["cert_acc"] for e in EPSILONS] for k in K_POISONS]),
        "certified held-out accuracy (baseline)", ".2f")
heatmap(ax_delta, np.array([[test(k, e, FULL)["cert_acc"] - test(k, e)["cert_acc"] for e in EPSILONS] for k in K_POISONS]),
        rf"accuracy gained by bisecting all {D}", "+.3f")

for colour, k in zip(K_COLOURS, K_POISONS):
    ax_gain.plot(EPSILONS, [width_gain(k, e, test(k, e, FULL)) for e in EPSILONS], marker="o", ms=2.5,
                 color=colour, label=f"$k={k}$")
    ax_gain.plot(EPSILONS, [width_gain(k, e, test(k, e, SCREEN)) for e in EPSILONS], marker="^", ms=2.5,
                 linestyle="--", color=colour)
ax_gain.set_xscale("log")
ax_gain.set_ylim(0, None)
ax_gain.set_xlabel(r"poisoning radius $\epsilon$")
ax_gain.set_ylabel("box width reduction")
ax_gain.legend(fontsize="xx-small", title=rf"solid: all {D}, dashed: top {SCREEN_SPLIT_DIMS}", title_fontsize="xx-small")

# each bisect-all leaf is an eps/2 cube, so the unrefined eps/2 run indicates how tight one leaf is
ceiling = [width_gain(r["k"], r["eps"], r["half_eps"]) for r in rows]
achieved = [width_gain(r["k"], r["eps"], r["full"]) for r in rows]
ax_ceiling.scatter(ceiling, achieved, s=6, c=[K_COLOURS[K_POISONS.index(r["k"])] for r in rows], zorder=3)
top = max(ceiling + achieved) * 1.05
ax_ceiling.plot([0, top], [0, top], linestyle=":", color=C["grey"], label="$y=x$")
ax_ceiling.set_xlim(0, top)
ax_ceiling.set_ylim(0, top)
ax_ceiling.set_xlabel(r"reduction of the unrefined $\epsilon/2$ run")
ax_ceiling.set_ylabel(rf"reduction bisecting all {D}")
ax_ceiling.legend(fontsize="xx-small")

fig.suptitle(rf"CIFAR-10 on PCA features ($d={D}$): threat grid, test split", fontsize="small")
script_utils.apply_figure_size(fig, script_utils.set_size(1.0, subplots, shrink_height=0.9), dpi=300)
path = f"{fig_dir}/cifar_pca_threat_sweep.pdf"
plt.savefig(path, dpi=300)
print(f"\n  figure: {path}", file=sys.stderr)
