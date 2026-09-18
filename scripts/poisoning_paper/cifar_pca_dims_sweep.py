"""
Feature-width experiment: whether the certificate, and refinement's benefit, change as more PCA
components are kept.

Hypotheses:
    H6a The relative gain from bisecting all ``d`` coordinates barely depends on ``d`` (within about
        1 point). At any ``d``, each leaf is an eps/2 cube, so the tightening per leaf is the same,
        while the cost grows as 2^d.
    H6b The unrefined certificate loosens as ``d`` grows, because the eps-ball's contribution through
        the first layer, ``eps * sum_i |W_ji|``, grows with the number of inputs. Nominal accuracy rises
        slightly with the extra explained variance.

Mechanism. The refinement-depth attack (k=100, eps=0.05) at each d in {20, 22, 24}, with four runs
per width:
    - unrefined;
    - the 12 most sensitive coordinates bisected;
    - all ``d`` bisected (2^20, 2^22 and 2^24 leaves, sharded across 4 GPUs);
    - unrefined at eps/2.
Each ``d`` uses the hyperparameters selected for that width (cifar_pca_selection.py), because a
schedule tuned at one width can collapse at another.

Caveat. Selection chose a different pre-training radius at d=20 (``pt_epsilon`` = 0.01) than at
d=22 and 24 (0.02). Going from 0.01 to 0.02 shrinks box width by 22-24%, more than ``d`` does, so the
baselines here mix the effect of ``d`` with the effect of ``pt_epsilon``, and this script alone
cannot test H6b. The control, cifar_pca_pt_epsilon_control.py, fixes ``pt_epsilon`` at 0.01. There
the baseline widens with ``d`` (1.03x at d=22, 1.14x at d=24), so H6b's box-width claim holds, and
the bisect-all gain stays within 0.5 points (H6a). H6b's nominal-accuracy claim is not supported:
0.795, 0.802 and 0.791, within sampling noise.

Output:
    - a table on stderr;
    - ``cifar_pca_dims_sweep.json``;
    - a figure of certified accuracy, box-width reduction and bisect-all GPU-hours, each against ``d``.

Reads cached runs only, and exits listing any that are missing.

Key external dependencies: matplotlib, and the sibling modules cifar_pca, cifar_pca_manifest and
script_utils.
"""

import dataclasses
import json
import sys

import matplotlib.pyplot as plt

import cifar_pca
import script_utils
from cifar_pca_manifest import E2_EPS, E2_K, E3_DIMS, SCREEN_SPLIT_DIMS, Run, collect, e3_runs

results = collect(e3_runs())
explained = cifar_pca.fit_pca()[0].explained_variance_ratio_

rows = []
for d in E3_DIMS:
    base, screen, full, half = (results[Run(d, E2_K, eps, refine)]
                                for eps, refine in ((E2_EPS, None), (E2_EPS, (2, SCREEN_SPLIT_DIMS)),
                                                    (E2_EPS, (2, d)), (E2_EPS / 2, None)))
    rows.append({"d": d, "explained_variance": float(explained[:d].sum()),
                 "hyperparameters": dataclasses.asdict(cifar_pca.selected_hyperparameters(d)),
                 "base": base, "screen": screen, "full": full, "half_eps": half})


def gain(row, key):
    return 1 - row[key]["test"]["box_width"] / row["base"]["test"]["box_width"]


# %%
print(f"\nE3 | CIFAR-10 PCA k={E2_K} eps={E2_EPS} | per-d selected hyperparameters | test split", file=sys.stderr)
print(f"  {'d':>3s} {'var':>6s}  {'nominal':>15s}  {'certified held-out acc':>26s}  {'width reduction':>23s}  "
      f"{'full run':>13s} {'viol':>7s}", file=sys.stderr)
print(f"  {'':>3s} {'':>6s}  {'held':>7s} {'clean':>7s}  {'base':>8s} {'top12':>8s} {'all d':>8s}  "
      f"{'top12':>7s} {'all d':>7s} {'eps/2':>7s}  {'hours':>7s} {'ranks':>5s}", file=sys.stderr)
for r in rows:
    t = {key: r[key]["test"] for key in ("base", "screen", "full")}
    violation = max(r[key]["violation"] for key in ("base", "screen", "full"))
    print(f"  {r['d']:>3d} {r['explained_variance']:6.1%}  {t['base']['nominal_acc']:7.4f} "
          f"{t['base']['nominal_clean_acc']:7.4f}  {t['base']['cert_acc']:8.4f} {t['screen']['cert_acc']:8.4f} "
          f"{t['full']['cert_acc']:8.4f}  {gain(r, 'screen'):7.2%} {gain(r, 'full'):7.2%} {gain(r, 'half_eps'):7.2%}  "
          f"{r['full']['seconds'] / 3600:7.2f} {r['full']['world_size']:>5d} {violation:7.1e}", file=sys.stderr)

results_dir, _, _, fig_dir = cifar_pca.dirs()
with open(f"{results_dir}/cifar_pca_dims_sweep.json", "w") as file:
    json.dump(rows, file, indent=1)

# %%
C = script_utils.colours
dims = [r["d"] for r in rows]
subplots = (1, 3)
fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300)

for key, style, label in (("base", dict(marker="s", linestyle=":", color=C["grey"]), "baseline"),
                          ("screen", dict(marker="^", linestyle="--", color=C["purple"]), f"bisect top {SCREEN_SPLIT_DIMS}"),
                          ("full", dict(marker="o", color=C["green"]), "bisect all $d$")):
    axs[0].plot(dims, [r[key]["test"]["cert_acc"] for r in rows], label=label, **style)
axs[0].plot(dims, [r["base"]["test"]["nominal_acc"] for r in rows], marker="v", linestyle="--", color=C["orange"],
            label="nominal")
axs[0].set_ylim(0, 1)
axs[0].set_ylabel("certified held-out accuracy")
axs[0].legend(fontsize="xx-small")

axs[1].plot(dims, [gain(r, "full") for r in rows], marker="o", color=C["green"], label="bisect all $d$")
axs[1].plot(dims, [gain(r, "screen") for r in rows], marker="^", linestyle="--", color=C["purple"],
            label=f"bisect top {SCREEN_SPLIT_DIMS}")
axs[1].plot(dims, [gain(r, "half_eps") for r in rows], marker="x", linestyle=":", color=C["grey"],
            label=r"unrefined at $\epsilon/2$")
axs[1].set_ylim(0, None)
axs[1].set_ylabel("box width reduction")
axs[1].legend(fontsize="xx-small")

axs[2].plot(dims, [r["full"]["seconds"] * r["full"]["world_size"] / 3600 for r in rows], marker="o", color=C["green"])
axs[2].set_yscale("log")
axs[2].set_ylabel("GPU-hours, bisect all $d$")

for ax in axs:
    ax.set_xticks(dims)
    ax.set_xlabel("PCA components $d$")
fig.suptitle(rf"CIFAR-10 on PCA features: feature width, $k={E2_K}$, $\epsilon={E2_EPS}$, test split", fontsize="small")
script_utils.apply_figure_size(fig, script_utils.set_size(1.0, subplots, shrink_height=2.4), dpi=300)
path = f"{fig_dir}/cifar_pca_dims_sweep.pdf"
plt.savefig(path, dpi=300)
print(f"\n  figure: {path}", file=sys.stderr)
