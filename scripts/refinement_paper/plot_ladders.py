"""
Paper figure of the refinement ladders (refinement_ladders.pdf/.png): certified test accuracy at each
partition of the eps-ball, against the nominal model and the unrefined certificate.

One panel per task, in a 2x2 grid, each at the attack of the task's ladder (tasks.<task>.K_POISON,
EPSILON). The x axis is the leaf count per eps-ball on a log scale, and every point is labelled with
its partition: N^D, or a product of tiers for rungs that cut the most sensitive features finer than
the rest (3^5.2^5). Bisection rungs (every cut feature halved) and finer cuts are separate series.
Open markers are runs whose leaves were sharded over several GPUs; filled markers ran on one GPU.

The half-moons and MAGIC panels also carry the task's small-k, large-eps ladder (<task>_smallk.json):
the same rungs, drawn dashed with square markers, with its own unrefined certificate. Both attacks share
the task's nominal model. The image tasks' small-k runs are controls with one rung and are not drawn.

Reads only the results files aggregate.py writes (<task>.json, <task>_smallk.json), never the parameter
boxes, so it runs anywhere those files have been copied.

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
# finer cuts that, on a task where they are alternatives to bisection, form a series of their own: the
# same cut cutting ever more features
FINER_SERIES = {"octmnist": ["3^10", "3^12", "3^15"]}
# tasks whose small-k ladder (<task>_smallk.json) is drawn over their k=200 ladder
SMALL_K_OVERLAY = {"halfmoons", "magic"}


def _ladder(cell):
    """The unrefined certificate and the rungs of one results-file cell."""
    return {
        "k": cell["k"],
        "epsilon": cell["epsilon"],
        "nominal": cell["baseline"]["metrics"]["nominal_acc"],
        "baseline": cell["baseline"]["metrics"]["cert_acc"],
        "rungs": [
            {
                "rung": r["rung"],
                "leaves": r["leaves"],
                "bisection": max(r["splits"]) == 2,
                "label": r"\cdot".join(f"{n}^{{{d}}}" for n, d in harness.rung_tiers(r["splits"])),
                "cert_acc": r["metrics"]["cert_acc"],
                "sharded": r["world_size"] > 1,
            }
            for r in sorted(cell["rungs"], key=lambda r: r["leaves"])
        ],
    }


def load_panel(results_dir, name):
    """The ladder of task ``name`` at the task's ladder attack, and its small-k ladder where one is drawn."""
    task = tasks.get(name)
    with open(os.path.join(results_dir, f"{name}.json")) as file:
        results = json.load(file)
    (cell,) = [c for c in results["cells"] if c["k"] == task.K_POISON and c["epsilon"] == task.EPSILON]
    panel = _ladder(cell)
    panel["title"] = f"{results['title']}, $k={cell['k']}$, $\\epsilon={cell['epsilon']:g}$"
    panel["finer_is_ladder"] = name in FINER_CUTS_CONTINUE_LADDER
    panel["finer_series"] = FINER_SERIES.get(name, [])
    panel["small_k"] = None
    if name in SMALL_K_OVERLAY:
        with open(os.path.join(results_dir, f"{name}_smallk.json")) as file:
            (small_cell,) = json.load(file)["cells"]
        small = _ladder(small_cell)
        assert round(small["nominal"], 9) == round(panel["nominal"], 9), f"{name}: the nominal model differs"
        panel["small_k"] = small
        panel["title"] = (f"{results['title']}: $k={cell['k']}$, $\\epsilon={cell['epsilon']:g}$ and "
                          f"$k={small['k']}$, $\\epsilon={small['epsilon']:g}$")
    return panel


def _draw_series(ax, panel, ladder, linestyle, marker, markersize, linewidth):
    """The bisection and finer-cut lines and markers of one ladder."""
    rungs = ladder["rungs"]
    bisect = [r for r in rungs if r["bisection"]]
    finer = [r for r in rungs if not r["bisection"]]
    if panel["finer_is_ladder"]:
        finer_line = bisect[-1:] + finer
    else:
        finer_line = [r for r in finer if r["rung"] in panel["finer_series"]]

    ax.plot([r["leaves"] for r in bisect], [r["cert_acc"] for r in bisect], color=BISECT, linewidth=linewidth,
            linestyle=linestyle, zorder=2)
    if finer_line:
        ax.plot([r["leaves"] for r in finer_line], [r["cert_acc"] for r in finer_line], color=FINER,
                linewidth=linewidth, linestyle=linestyle, zorder=2)
    for r in rungs:
        colour = BISECT if r["bisection"] else FINER
        ax.plot(r["leaves"], r["cert_acc"], marker=marker, markersize=markersize, linestyle="none", zorder=3,
                markerfacecolor=SURFACE if r["sharded"] else colour, markeredgecolor=colour,
                markeredgewidth=1.2 if marker == "o" else 1.0)


def draw(ax, panel):
    rungs = panel["rungs"]
    small = panel["small_k"]

    ax.axhline(panel["nominal"], linestyle="--", linewidth=1.0, color=INK_MUTED, zorder=1)
    ax.axhline(panel["baseline"], linestyle=":", linewidth=1.2, color=INK_MUTED, zorder=1)
    _draw_series(ax, panel, panel, "-", "o", 4, 1.5)
    if small is not None:
        ax.axhline(small["baseline"], linestyle=(0, (1, 3)), linewidth=1.0, color=INK_MUTED, zorder=1)
        _draw_series(ax, panel, small, "--", "s", 3.5, 1.2)
        # which unrefined line is which, at the right edge, each on its outer side
        above = panel["baseline"] > small["baseline"]
        for value, k, up in ((panel["baseline"], panel["k"], above), (small["baseline"], small["k"], not above)):
            ax.annotate(f"$k={k}$", (1.0, value), xycoords=("axes fraction", "data"), textcoords="offset points",
                        xytext=(-2, 2 if up else -2), ha="right", va="bottom" if up else "top",
                        fontsize=6, color=INK_MUTED)

    # partition labels, once per rung (a small-k ladder has the same rungs): where both families share
    # the panel, bisection above its line and finer cuts below, except a finer cut with no bisection
    # rung of its leaf count, which is labelled above; a single rising ladder is labelled above-left
    # of each point, clear of the curve
    bisection_leaves = {r["leaves"] for r in rungs if r["bisection"]}
    for r in rungs:
        if panel["finer_is_ladder"]:
            offset, ha, va = (-3, 3), "right", "bottom"
        elif r["bisection"] or (r["rung"] not in panel["finer_series"] and r["leaves"] not in bisection_leaves):
            offset, ha, va = (0, 5), "center", "bottom"
        else:
            offset, ha, va = (0, -6), "center", "top"
        ax.annotate(f"${r['label']}$", (r["leaves"], r["cert_acc"]),
                    textcoords="offset points", xytext=offset, ha=ha, va=va, fontsize=6, color=INK)

    ax.set_xscale("log")
    ax.set_title(panel["title"], fontsize=8)
    ax.set_xlabel("leaves per $\\epsilon$-ball", fontsize=8)
    values = [panel["baseline"]] + [r["cert_acc"] for r in rungs]
    if small is not None:
        values += [small["baseline"]] + [r["cert_acc"] for r in small["rungs"]]
    lo = min(values)
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
        Line2D([], [], color=INK_MUTED, marker="o", markersize=4, label="$k=200$, small $\\epsilon$"),
        Line2D([], [], color=INK_MUTED, marker="s", markersize=3.5, linestyle="--", label="$k=25$, large $\\epsilon$"),
    ]
    fig.legend(handles=legend, loc="outside lower center", ncol=3, fontsize=7, frameon=False,
               title="$d'$: number of input features cut", title_fontsize=6.5)
    script_utils.apply_figure_size(fig, script_utils.set_size(1.0, subplots, shrink_height=1.45), dpi=300)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None,
                        help="directory of <task>.json and <task>_smallk.json files; default .results")
    parser.add_argument("--out", default=None, help="figures directory; default .figures")
    args = parser.parse_args()

    figure = make_figure(args.results or harness.directory(".results"))
    out_dir = args.out or harness.directory(".figures")
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"refinement_ladders.{ext}")
        figure.savefig(path, dpi=300)
        print(f"figure: {path}")
