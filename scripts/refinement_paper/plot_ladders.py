"""
Paper figure of the refinement ladders (refinement_ladders.pdf/.png): certified test accuracy at each
partition of the eps-ball, against the nominal model and the unrefined certificate.

One panel per task, in a 2x2 grid, each at the attack of the task's ladder (tasks.<task>.K_POISON,
EPSILON). The x axis is the leaf count per eps-ball on a log scale, and every point is labelled with
its partition: N^D, or a product of tiers for rungs that cut the most sensitive features finer than
the rest (3^5.2^5). Bisection rungs (every cut feature halved) and finer cuts are separate series.
Open markers are runs whose leaves were sharded over several GPUs; filled markers ran on one GPU.

Reads only the results files aggregate.py writes (<task>.json), never the parameter boxes, so it runs
anywhere those files have been copied.

Usage: python plot_ladders.py [--results DIR] [--out DIR]
``--results`` defaults to ``.results``; ``--results reference`` plots the results shipped with the
repository. ``--out`` defaults to ``.figures``.

Key external dependencies: matplotlib, the sibling modules harness and tasks, and
scripts/poisoning_paper/script_utils.py (the upstream house style and figure sizing).
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import harness
import tasks
from tasks import upstream_scripts  # noqa: F401 - makes the upstream poisoning_paper modules importable

import script_utils

# Two categorical slots, validated together (CVD dE 24.7, normal-vision dE 33.6, >= 3:1 contrast).
BISECT = "#2a78d6"
FINER = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "white"

# tasks whose finer cuts continue the ladder from the last bisection rung, and so are joined to it;
# on the others they are alternatives to bisection at similar cost
FINER_CUTS_CONTINUE_LADDER = {"halfmoons", "magic"}


def load_panel(results_dir, name):
    """The ladder of task ``name``: the cell of its results file at the task's ladder attack."""
    task = tasks.get(name)
    with open(os.path.join(results_dir, f"{name}.json")) as file:
        results = json.load(file)
    (cell,) = [c for c in results["cells"] if c["k"] == task.K_POISON and c["epsilon"] == task.EPSILON]
    return {
        "title": f"{results['title']}, $k={cell['k']}$, $\\epsilon={cell['epsilon']:g}$",
        "nominal": cell["baseline"]["metrics"]["nominal_acc"],
        "baseline": cell["baseline"]["metrics"]["cert_acc"],
        "finer_is_ladder": name in FINER_CUTS_CONTINUE_LADDER,
        "rungs": [
            {
                "leaves": r["leaves"],
                "bisection": max(r["splits"]) == 2,
                "label": r"\cdot".join(f"{n}^{{{d}}}" for n, d in harness.rung_tiers(r["splits"])),
                "cert_acc": r["metrics"]["cert_acc"],
                "sharded": r["world_size"] > 1,
            }
            for r in cell["rungs"]
        ],
    }


def draw(ax, panel):
    rungs = sorted(panel["rungs"], key=lambda r: r["leaves"])
    bisect = [r for r in rungs if r["bisection"]]
    finer = [r for r in rungs if not r["bisection"]]
    finer_line = (bisect[-1:] + finer) if panel["finer_is_ladder"] else []

    ax.axhline(panel["nominal"], linestyle="--", linewidth=1.0, color=INK_MUTED, zorder=1)
    ax.axhline(panel["baseline"], linestyle=":", linewidth=1.2, color=INK_MUTED, zorder=1)
    ax.plot([r["leaves"] for r in bisect], [r["cert_acc"] for r in bisect], color=BISECT, linewidth=1.5, zorder=2)
    if finer_line:
        ax.plot([r["leaves"] for r in finer_line], [r["cert_acc"] for r in finer_line], color=FINER,
                linewidth=1.5, zorder=2)
    for r in rungs:
        colour = BISECT if r["bisection"] else FINER
        ax.plot(r["leaves"], r["cert_acc"], marker="o", markersize=4, linestyle="none", zorder=3,
                markerfacecolor=SURFACE if r["sharded"] else colour, markeredgecolor=colour, markeredgewidth=1.2)

    # partition labels: where both families share the panel, bisection above its line and finer cuts
    # below; a single rising ladder is labelled above-left of each point, clear of the curve
    for r in rungs:
        if panel["finer_is_ladder"]:
            offset, ha, va = (-3, 3), "right", "bottom"
        elif r["bisection"]:
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


def make_figure(results_dir):
    panels = [load_panel(results_dir, name) for name in tasks.NAMES]
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
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None, help="directory of <task>.json files; default .results")
    parser.add_argument("--out", default=None, help="figures directory; default .figures")
    args = parser.parse_args()

    figure = make_figure(args.results or harness.directory(".results"))
    out_dir = args.out or harness.directory(".figures")
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"refinement_ladders.{ext}")
        figure.savefig(path, dpi=300)
        print(f"figure: {path}")
