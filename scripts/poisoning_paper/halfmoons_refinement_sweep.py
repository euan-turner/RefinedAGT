# %%
"""
Leaf-count scaling of input-ball refinement for half-moons feature poisoning.

Hypothesis. Once every one of the 6 features is split, cutting each one more finely keeps tightening
the certificate, with diminishing returns. The first campaign (n_splits 2 to 6, one GPU) found
certified accuracy still rising at 6^6 (0.736 to 0.762), so the ladder continues on 4 GPUs.

Mechanism. One attack (k=200, eps=0.01) and one nominal model; the script asserts that the nominal
model is the same at every rung. The unrefined run is compared with runs at each ``n_splits`` in
``--rungs``. The gain is the relative reduction in box width against the unrefined run. See
halfmoons_refinement for the task and the metrics.

Output: a table on stdout (box width and certified accuracy, unrefined and refined, width gain, run
time and number of GPUs), ``halfmoons_refinement_sweep.json``, and a figure of box width and
certified accuracy against leaf count.

Reads cached runs only (written by halfmoons_run.py through the Slurm jobs), and fails on any missing
rung.

Usage: python halfmoons_refinement_sweep.py [--rungs 2 3 4 5 6 8 10 12 16]

Key external dependencies: matplotlib, and the sibling modules halfmoons_refinement and script_utils.
"""

import argparse
import json

import matplotlib.pyplot as plt

import halfmoons_refinement
import script_utils

parser = argparse.ArgumentParser()
parser.add_argument("--rungs", type=int, nargs="+", default=[2, 3, 4, 5, 6],
                    help="n_splits values to read, e.g. --rungs 2 3 4 5 6 8 10 12 16")
args = parser.parse_args()

# %%
""" Read the cached baseline and refined runs. Never trains: require_cached=True. """

base_model, base_record = halfmoons_refinement.run(None, require_cached=True)
base = {**base_record, **halfmoons_refinement.certified_metrics(base_model)}

rows = []
for n_splits in args.rungs:
    refined_model, refined_record = halfmoons_refinement.run(n_splits, require_cached=True)
    refined = {**refined_record, **halfmoons_refinement.certified_metrics(refined_model)}

    # the nominal trajectory is refinement-independent: same nominal model at every rung
    assert abs(base["nominal_acc"] - refined["nominal_acc"]) < 1e-9, (
        f"nominal accuracy diverged at {n_splits=}: {base['nominal_acc']} vs {refined['nominal_acc']}"
    )

    leaves = n_splits ** halfmoons_refinement.SPLIT_DIMS
    width_gain = 1 - refined["box_width"] / base["box_width"] if base["box_width"] else 0.0
    rows.append({"n_splits": n_splits, "n_dims": halfmoons_refinement.SPLIT_DIMS, "leaves": leaves,
                 "refined": refined, "width_gain": width_gain})

# %%
""" Table and results file. """

print(
    f"halfmoons feature poisoning | eps={halfmoons_refinement.EPSILON} | k_poison={halfmoons_refinement.K_POISON} | "
    f"strategy={halfmoons_refinement.STRATEGY} | rungs {args.rungs}\n"
)
print(
    f"  {'split':>5s} {'leaves':>9s}  {'box width':>12s} {'box width':>12s} {'width':>7s}   "
    f"{'cert acc':>9s} {'cert acc':>9s}  {'nominal':>8s}  {'time':>8s} {'ranks':>5s} {'viol':>7s}"
)
print(f"  {'':>5s} {'':>9s}  {'(baseline)':>12s} {'(refined)':>12s} {'gain':>7s}   {'(baseline)':>9s} {'(refined)':>9s}")
for r in rows:
    refined = r["refined"]
    print(
        f"  {f'{r['n_splits']}^{r['n_dims']}':>5s} {r['leaves']:>9d}  {base['box_width']:12.4e} "
        f"{refined['box_width']:12.4e} {r['width_gain']:6.1%}   {base['cert_acc_worst']:9.4f} "
        f"{refined['cert_acc_worst']:9.4f}  {base['nominal_acc']:8.4f}  {refined['seconds']:7.0f}s "
        f"{refined['world_size']:>5d} {refined['violation']:7.1e}"
    )

results_dir, _, _, fig_dir = script_utils.make_dirs()
with open(f"{results_dir}/halfmoons_refinement_sweep.json", "w") as file:
    json.dump({"baseline": base, "rungs": rows}, file, indent=1)

# %%
""" Plot: certified box width and certified (worst-case) test accuracy against leaf count. """

leaves = [r["leaves"] for r in rows]
subplots = (1, 2)
fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300)

axs[0].axhline(base["box_width"], linestyle=":",
               color=script_utils.colours["grey"], label="baseline (no refinement)")
axs[0].plot(leaves, [r["refined"]["box_width"] for r in rows], marker="o",
            color=script_utils.colours["green"], label="refined")
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_xlabel("refinement leaves per fragment")
axs[0].set_ylabel("certified parameter box width")
axs[0].legend(fontsize="x-small")

axs[1].axhline(base["cert_acc_worst"], linestyle=":",
               color=script_utils.colours["grey"], label="baseline (no refinement)")
axs[1].plot(leaves, [r["refined"]["cert_acc_worst"] for r in rows], marker="o",
            color=script_utils.colours["green"], label="refined")
axs[1].axhline(base["nominal_acc"], linestyle="--",
               color=script_utils.colours["orange"], label="nominal")
axs[1].set_xscale("log")
axs[1].set_xlabel("refinement leaves per fragment")
axs[1].set_ylabel("certified worst-case test accuracy")
axs[1].legend(fontsize="x-small")

fig.suptitle(rf"half-moons: certified tightness vs refinement leaves, $k={halfmoons_refinement.K_POISON}$, "
             rf"$\epsilon={halfmoons_refinement.EPSILON}$", fontsize="small")
script_utils.apply_figure_size(fig, script_utils.set_size(0.9, subplots, shrink_height=1.6), dpi=300)
path = f"{fig_dir}/halfmoons_refinement_sweep.pdf"
plt.savefig(path, dpi=300)
print(f"\n  figure: {path}")
