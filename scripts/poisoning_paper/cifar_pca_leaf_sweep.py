"""
Refinement-depth experiment: how the certificate tightens as the partition of the eps-ball grows, and
how best to spend a fixed number of leaves.

Hypotheses:
    H2  The gain grows steadily but modestly with the number of bisected coordinates, reaching roughly
        5-7% box-width reduction when all 20 are bisected.
    H5  At equal leaf count, bisecting more coordinates beats cutting fewer coordinates more finely.
        This is the least certain hypothesis: on OCT-MNIST, the finer cut 4^8 slightly beat 2^16.

Mechanism. One attack at d=20: k=100, eps=0.05, chosen so the unrefined certificate is informative
but not saturated. The unrefined baseline is compared against two families of partitions:
    - a bisection ladder: 2 pieces on each of the 4, 8, 12, 16 and 20 most sensitive coordinates
      (2^4 to 2^20 leaves);
    - equal-cost alternatives that cut fewer coordinates more finely: 16^3 (against 2^12), 4^8
      (against 2^16) and 4^10 (against 2^20).
Every run shares one nominal model; the script asserts this. All metrics are on the test split.

Output:
    - a table on stderr: box width, certified accuracy and cross-entropy, their gains over the
      baseline, and run time;
    - ``cifar_pca_leaf_sweep.json``;
    - a figure of box width, certified cross-entropy and certified accuracy against leaf count, with
      the ladder drawn as a line and the alternatives as labelled points.

``--cell e4`` runs the same analysis on the refinement ladder at k=200, eps=0.02 (the E1 cell with the
largest certified-accuracy gain; see cifar_pca_manifest), whose equal-cost alternatives are the
quadrisections 4^8 and 4^10. Its results file and figure are suffixed ``_e4``.

Reads cached runs only, and exits listing any that are missing.

Key external dependencies: matplotlib, and the sibling modules cifar_pca, cifar_pca_manifest and
script_utils.
"""

import argparse
import json
import sys

import matplotlib.pyplot as plt

import cifar_pca
import script_utils
import cifar_pca_manifest as manifest
from cifar_pca_manifest import Run, collect

parser = argparse.ArgumentParser()
parser.add_argument("--cell", choices=("e2", "e4"), default="e2")
args = parser.parse_args()
CELL, SCHEDULE, RUNS = {
    "e2": ((manifest.E2_D, manifest.E2_K, manifest.E2_EPS), manifest.E2_SCHEDULE, manifest.e2_runs()),
    "e4": ((manifest.E4_D, manifest.E4_K, manifest.E4_EPS), manifest.E4_SCHEDULE, manifest.e4_runs()),
}[args.cell]
D, K, EPS = CELL
SUFFIX = "" if args.cell == "e2" else f"_{args.cell}"

results = collect(RUNS)
base = results[Run(D, K, EPS, None)]
rows = []
for n_splits, n_dims in sorted(SCHEDULE, key=lambda r: r[0] ** r[1]):
    result = results[Run(D, K, EPS, (n_splits, n_dims))]
    assert abs(result["test"]["nominal_acc"] - base["test"]["nominal_acc"]) < 1e-9, "refinement moved the nominal model"
    rows.append({"n_splits": n_splits, "n_dims": n_dims, "leaves": n_splits**n_dims, **result})

# %%
b = base["test"]
print(f"\n{args.cell.upper()} | CIFAR-10 PCA d={D} k={K} eps={EPS} | nominal held-out test acc {b['nominal_acc']:.4f}, "
      f"CE {b['nominal_ce']:.4f} | test split", file=sys.stderr)
print(f"  {'split':>6s} {'leaves':>9s}  {'box':>9s} {'width':>7s}  {'cert acc':>8s} {'gain':>7s}  "
      f"{'cert CE':>8s} {'gain':>7s}  {'clean':>6s}  {'time':>9s} {'ranks':>5s} {'viol':>7s}", file=sys.stderr)
print(f"  {'none':>6s} {1:>9d}  {b['box_width']:9.3e} {'--':>7s}  {b['cert_acc']:8.4f} {'--':>7s}  "
      f"{b['cert_ce']:8.4f} {'--':>7s}  {b['cert_clean_acc']:6.4f}  {base['seconds']:8.0f}s {base['world_size']:>5d} "
      f"{base['violation']:7.1e}", file=sys.stderr)
for r in rows:
    t = r["test"]
    print(f"  {f'{r['n_splits']}^{r['n_dims']}':>6s} {r['leaves']:>9d}  {t['box_width']:9.3e} "
          f"{1 - t['box_width'] / b['box_width']:7.2%}  {t['cert_acc']:8.4f} {t['cert_acc'] - b['cert_acc']:+7.4f}  "
          f"{t['cert_ce']:8.4f} {b['cert_ce'] - t['cert_ce']:+7.4f}  {t['cert_clean_acc']:6.4f}  "
          f"{r['seconds']:8.0f}s {r['world_size']:>5d} {r['violation']:7.1e}", file=sys.stderr)

results_dir, _, _, fig_dir = cifar_pca.dirs()
with open(f"{results_dir}/cifar_pca_leaf_sweep{SUFFIX}.json", "w") as file:
    json.dump({"baseline": base, "rungs": rows}, file, indent=1)

# %%
C = script_utils.colours
bisect = [r for r in rows if r["n_splits"] == 2]
equal_cost = [r for r in rows if r["n_splits"] != 2]
subplots = (1, 3)
fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300)

for ax, key, ylabel in ((axs[0], "box_width", "certified box width"),
                        (axs[1], "cert_ce", "certified held-out cross-entropy"),
                        (axs[2], "cert_acc", "certified held-out accuracy")):
    ax.axhline(b[key], linestyle=":", color=C["grey"], label="baseline")
    if key != "box_width":
        ax.axhline(b["nominal_ce" if key == "cert_ce" else "nominal_acc"], linestyle="--", color=C["purple"],
                   label="nominal")
    ax.plot([r["leaves"] for r in bisect], [r["test"][key] for r in bisect], marker="o", color=C["green"],
            label=r"bisect ($n_{splits}=2$)")
    if equal_cost:
        ax.scatter([r["leaves"] for r in equal_cost], [r["test"][key] for r in equal_cost], marker="^",
                   color=C["orange"], zorder=3, label="fewer, finer dims")
    for r in equal_cost:
        ax.annotate(rf"${r['n_splits']}^{{{r['n_dims']}}}$", (r["leaves"], r["test"][key]),
                    textcoords="offset points", xytext=(-5, 4), ha="right", fontsize="xx-small", color=C["orange"])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("leaves per fragment")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize="xx-small")

fig.suptitle(rf"CIFAR-10 on PCA features ($d={D}$): refinement depth, $k={K}$, $\epsilon={EPS}$, "
             "test split", fontsize="small")
script_utils.apply_figure_size(fig, script_utils.set_size(1.0, subplots, shrink_height=2.4), dpi=300)
path = f"{fig_dir}/cifar_pca_leaf_sweep{SUFFIX}.pdf"
plt.savefig(path, dpi=300)
print(f"\n  figure: {path}", file=sys.stderr)
