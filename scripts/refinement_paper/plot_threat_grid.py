"""
Figure of the unrefined certificate over the threat-model grid (threat_grid.pdf/.png): for each task, a
heatmap over (k, epsilon) of the gap between nominal and certified test accuracy.

One panel per task, in the order of tasks.NAMES, all on one colour scale: green where the certificate
is close to nominal, purple where it is far. The scale is square-root, so the small gaps of the image
tasks are resolved without saturating the large gaps of the tabular ones. Each cell is labelled with
its gap in accuracy points, and the cell of the task's ladder attack (K_POISON, EPSILON) is outlined.

Reads only the results files aggregate.py writes for the grid (<task>_threat.json; reproduce.sh names
them), never the parameter boxes, so it runs anywhere those files have been copied.

Usage: python plot_threat_grid.py [--results DIR] [--out DIR]
``--results`` defaults to ``.results`` and ``--out`` to ``.figures``.

Key external dependencies: matplotlib, the sibling modules harness and tasks, and
scripts/poisoning_paper/script_utils.py (the upstream house style and figure sizing).
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Rectangle

import harness
import tasks
from tasks import upstream_scripts  # noqa: F401 - makes the upstream poisoning_paper modules importable

import script_utils

GAP_CMAP = LinearSegmentedColormap.from_list("gap", ["#1b7837", "#7fbf7b", "#f7f7f7", "#af8dc3", "#762a83"])
GAP_NORM = PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0)  # a gap is at most the nominal accuracy, at most 1
INK = "#0b0b0b"


def load_panel(results_dir, name):
    """The grid of task ``name``: k and epsilon values and the nominal - certified gap at each cell."""
    task = tasks.get(name)
    with open(os.path.join(results_dir, f"{name}_threat.json")) as file:
        results = json.load(file)
    cells = {(c["k"], c["epsilon"]): c["baseline"]["metrics"] for c in results["cells"]}
    ks = sorted({k for k, _ in cells})
    epsilons = sorted({e for _, e in cells})
    gap = np.array([[cells[k, e]["nominal_acc"] - cells[k, e]["cert_acc"] for e in epsilons] for k in ks])
    nominal = {round(m["nominal_acc"], 9) for m in cells.values()}
    assert len(nominal) == 1, f"{name}: the nominal model differs between attacks"
    return {
        "title": results["title"],
        "nominal": nominal.pop(),
        "ks": ks,
        "epsilons": epsilons,
        "gap": gap,
        "ladder": (ks.index(task.K_POISON), epsilons.index(task.EPSILON))
        if (task.K_POISON, task.EPSILON) in cells else None,
    }


def draw(ax, panel):
    image = ax.imshow(panel["gap"], cmap=GAP_CMAP, norm=GAP_NORM, origin="lower", aspect="auto")
    for i in range(len(panel["ks"])):
        for j in range(len(panel["epsilons"])):
            gap = panel["gap"][i, j]
            on_dark = not 0.2 <= GAP_NORM(gap) <= 0.8
            ax.text(j, i, f"{100 * gap:.1f}", ha="center", va="center", fontsize=5.5,
                    color="white" if on_dark else INK)
    if panel["ladder"] is not None:
        i, j = panel["ladder"]
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.5))
    ax.set_xticks(range(len(panel["epsilons"])), [f"{e:g}" for e in panel["epsilons"]], fontsize=6)
    ax.set_yticks(range(len(panel["ks"])), panel["ks"], fontsize=6)
    ax.set_xlabel("$\\epsilon$", fontsize=8)
    ax.set_title(f"{panel['title']}\nnominal acc {panel['nominal']:.3f}", fontsize=7)
    ax.tick_params(length=0, which="both")
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def make_figure(results_dir):
    panels = [load_panel(results_dir, name) for name in tasks.NAMES]
    subplots = (1, len(panels))
    fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300)
    for ax, panel in zip(axs, panels):
        image = draw(ax, panel)
    axs[0].set_ylabel("$k$", fontsize=8)
    bar = fig.colorbar(image, ax=axs, shrink=0.9, pad=0.01)
    bar.set_ticks([0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    bar.ax.tick_params(labelsize=6)
    bar.ax.minorticks_off()
    bar.set_label("nominal $-$ certified accuracy, unrefined", fontsize=7)
    script_utils.apply_figure_size(fig, script_utils.set_size(1.0, (1, 1), shrink_height=0.65), dpi=300)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None, help="directory of <task>_threat.json files; default .results")
    parser.add_argument("--out", default=None, help="figures directory; default .figures")
    args = parser.parse_args()

    figure = make_figure(args.results or harness.directory(".results"))
    out_dir = args.out or harness.directory(".figures")
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"threat_grid.{ext}")
        figure.savefig(path, dpi=300)
        print(f"figure: {path}")
