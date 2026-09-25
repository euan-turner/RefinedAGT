"""
Paper figure for the refinement-ladder campaign (scripts/isambard/README.md section 5): certified
accuracy at each partition of the eps-ball, against the nominal model and the unrefined certificate.

One panel per ladder, in a 2x2 grid: CIFAR-10 (d=20, k=200, eps=0.02), OCT-MNIST (d=15, k=200,
eps=0.1), half-moons (k=200, eps=0.01) and MAGIC (k=200, eps=0.01). The x axis is the leaf count per eps-ball on a log scale, and every point is labelled with
its partition: n_splits^n_dims, or a product of two tiers for the MAGIC rungs that cut the most
sensitive features finer than the rest (3^5.2^5). Bisection ladders (n_splits = 2) and finer
cuts (n_splits > 2) are separate series. Open markers are runs whose leaves were sharded over several
GPUs (4 on Isambard, 2 for the local MAGIC runs); filled markers ran on one GPU. Both come from each
run's cached ``world_size``.

Reads only the aggregators' results files (cifar_pca_leaf_sweep.py --cell e4,
octmnist_pca_ladder_sweep.py, halfmoons_refinement_sweep.py, magic_refinement_sweep.py), never the
parameter boxes, so it runs anywhere those files have been copied. The MAGIC ladder was run locally,
not on Isambard, so its results file is read from ``--magic-results``.

Usage: python refinement_ladder_plots.py [--results isambard4/.results] [--magic-results .results]
           [--out <figures dir>]

Key external dependencies: matplotlib, and the sibling module script_utils (house style and sizing).
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import script_utils

# Two categorical slots, validated together (CVD dE 24.7, normal-vision dE 33.6, >= 3:1 contrast).
BISECT = "#2a78d6"
FINER = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "white"

parser = argparse.ArgumentParser()
parser.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "isambard4", ".results"))
parser.add_argument("--magic-results", default=None, help="directory holding magic_refinement_sweep.json; "
                    "default the local results directory ($AGT_ROOT/.results)")
parser.add_argument("--out", default=None, help="figures directory; default $AGT_ROOT/.figures")
args = parser.parse_args()


def load(name, results_dir=None):
    with open(os.path.join(results_dir or args.results, name)) as file:
        return json.load(file)


def rung(n_splits, n_dims, cert_acc, world_size, tiers=None):
    """One ladder point. ``tiers`` lists every (n_splits, n_dims) tier when the partition has more than
    one, the finest first; n_splits and n_dims are then the finest tier, which picks the series."""
    tiers = tiers or [(n_splits, n_dims)]
    leaves = 1
    for n, d in tiers:
        leaves *= n**d
    label = r"\cdot".join(f"{n}^{{{d}}}" for n, d in tiers)
    return {"n_splits": n_splits, "n_dims": n_dims, "leaves": leaves, "label": label,
            "cert_acc": cert_acc, "world_size": world_size}


def cifar_panel():
    data = load("cifar_pca_leaf_sweep_e4.json")
    base = data["baseline"]["test"]
    return {
        "title": "CIFAR-10 (PCA, $d=20$), $k=200$, $\\epsilon=0.02$",
        "nominal": base["nominal_acc"], "baseline": base["cert_acc"], "finer_is_ladder": False,
        "rungs": [rung(r["n_splits"], r["n_dims"], r["test"]["cert_acc"], r["world_size"]) for r in data["rungs"]],
    }


def octmnist_panel(k=200):
    (cell,) = [cell for cell in load("octmnist_pca_ladder_sweep.json") if cell["k"] == k]
    base = cell["baseline"]
    return {
        "title": f"OCT-MNIST (PCA, $d=15$), $k={k}$, $\\epsilon=0.1$",
        "nominal": base["nominal_acc"], "baseline": base["cert_acc"], "finer_is_ladder": False,
        "rungs": [rung(r["n_splits"], r["n_dims"], r["cert_acc"], r["cost"]["world_size"]) for r in cell["rungs"]],
    }


def halfmoons_panel():
    data = load("halfmoons_refinement_sweep.json")
    base = data["baseline"]
    return {
        "title": "half-moons ($d=6$), $k=200$, $\\epsilon=0.01$",
        "nominal": base["nominal_acc"], "baseline": base["cert_acc_worst"], "finer_is_ladder": True,
        "rungs": [rung(r["n_splits"], r["n_dims"], r["refined"]["cert_acc_worst"], r["refined"]["world_size"])
                  for r in data["rungs"]],
    }


def magic_panel():
    data = load("magic_refinement_sweep.json", args.magic_results or script_utils.make_dirs()[0])
    base = data["baseline"]
    return {
        "title": "MAGIC ($d=10$), $k=200$, $\\epsilon=0.01$",
        "nominal": base["nominal_acc"], "baseline": base["cert_acc_worst"], "finer_is_ladder": True,
        "rungs": [rung(r["n_splits"], r["n_dims"], r["refined"]["cert_acc_worst"], r["refined"]["world_size"],
                       tiers=r["tiers"]) for r in data["rungs"]],
    }


def draw(ax, panel):
    rungs = sorted(panel["rungs"], key=lambda r: r["leaves"])
    bisect = [r for r in rungs if r["n_splits"] == 2]
    finer = [r for r in rungs if r["n_splits"] > 2]
    # on half-moons and MAGIC the finer cuts continue the ladder from the last bisection rung
    finer_line = (bisect[-1:] + finer) if panel["finer_is_ladder"] else []

    ax.axhline(panel["nominal"], linestyle="--", linewidth=1.0, color=INK_MUTED, zorder=1)
    ax.axhline(panel["baseline"], linestyle=":", linewidth=1.2, color=INK_MUTED, zorder=1)
    ax.plot([r["leaves"] for r in bisect], [r["cert_acc"] for r in bisect], color=BISECT, linewidth=1.5, zorder=2)
    if finer_line:
        ax.plot([r["leaves"] for r in finer_line], [r["cert_acc"] for r in finer_line], color=FINER,
                linewidth=1.5, zorder=2)
    for r in rungs:
        colour = BISECT if r["n_splits"] == 2 else FINER
        sharded = r["world_size"] > 1
        ax.plot(r["leaves"], r["cert_acc"], marker="o", markersize=4, linestyle="none", zorder=3,
                markerfacecolor=SURFACE if sharded else colour, markeredgecolor=colour, markeredgewidth=1.2)

    # partition labels: where both families share the panel, bisection above its line and finer cuts
    # below; a single rising ladder is labelled above-left of each point, clear of the curve
    for r in rungs:
        if panel["finer_is_ladder"]:
            offset, ha, va = (-3, 3), "right", "bottom"
        elif r["n_splits"] == 2:
            offset, ha, va = (0, 5), "center", "bottom"
        else:
            offset, ha, va = (0, -6), "center", "top"
        ax.annotate(f"${r['label']}$", (r["leaves"], r["cert_acc"]),
                    textcoords="offset points", xytext=offset, ha=ha, va=va, fontsize=6, color=INK)

    ax.set_xscale("log")
    ax.set_title(panel["title"], fontsize=8)
    ax.set_xlabel("leaves per $\\epsilon$-ball", fontsize=8)
    lo = min(panel["baseline"], *(r["cert_acc"] for r in rungs))
    hi = panel["nominal"]
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(axis="y", linewidth=0.3, color="#d0d0cc")
    ax.set_axisbelow(True)


panels = [cifar_panel(), octmnist_panel(k=200), halfmoons_panel(), magic_panel()]
subplots = (2, 2)
fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300)
for ax, panel in zip(axs.flat, panels):
    draw(ax, panel)
for ax in axs[:, 0]:
    ax.set_ylabel("certified test accuracy", fontsize=8)

legend = [
    Line2D([], [], linestyle="--", color=INK_MUTED, label="nominal (no attack)"),
    Line2D([], [], linestyle=":", color=INK_MUTED, linewidth=1.2, label="certified, unrefined"),
    Line2D([], [], color=BISECT, marker="o", markersize=4.5, label="certified, bisection ($2^{d'}$)"),
    Line2D([], [], color=FINER, marker="o", markersize=4.5, label="certified, finer cuts ($s^{d'}$, $s>2$)"),
    Line2D([], [], linestyle="none", marker="o", markersize=4.5, color=INK_MUTED, label="1 GPU"),
    Line2D([], [], linestyle="none", marker="o", markersize=4.5, markerfacecolor=SURFACE,
           markeredgecolor=INK_MUTED, markeredgewidth=1.2, label="multi-GPU data parallel"),
]
fig.legend(handles=legend, loc="outside lower center", ncol=3, fontsize=7, frameon=False,
           title="$d'$: number of input features cut", title_fontsize=6.5)

script_utils.apply_figure_size(fig, script_utils.set_size(1.0, subplots, shrink_height=1.35), dpi=300)
out_dir = args.out or script_utils.make_dirs()[3]
os.makedirs(out_dir, exist_ok=True)
for ext in ("pdf", "png"):
    path = os.path.join(out_dir, f"refinement_ladders.{ext}")
    fig.savefig(path, dpi=300)
    print(f"figure: {path}")
