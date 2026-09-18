# %%
"""
Refinement-depth experiment on UCI house-electric: once every input coordinate is split, does cutting
each one more finely keep tightening the certificate?

Hypothesis. The certificate tightens as ``n_splits`` rises from 2 to 3 to 4 on all 11 features, with
diminishing returns. The other experiments mostly bisect a subset of coordinates. Here every
coordinate is already split at the first rung, so this tests the next lever: finer cuts on every
dimension, the step CIFAR-10 cannot afford at d=20.

Mechanism. One attack (k=200, eps=0.01) and one nominal model; the script asserts that the nominal
model is the same at every rung. The unrefined run is compared with runs at each ``n_splits`` in
``--rungs`` (2^11, 3^11 and 4^11 leaves). The gain is the relative reduction in box width against the
unrefined run. See uci_refinement for the task and the metrics.

Output: a table on stdout (box width and certified worst-case test MSE, unrefined and refined, and
the width gain), and a figure of box width and certified test MSE against leaf count.

Reads cached runs only (written by uci_run.py through the Slurm jobs), and fails on any missing rung.

Usage: python uci_refinement_sweep.py [--rungs 2 3 4]

Key external dependencies: matplotlib, and the sibling modules uci_refinement and script_utils.
"""

import argparse

import matplotlib.pyplot as plt

import script_utils
import uci_refinement

parser = argparse.ArgumentParser()
parser.add_argument("--rungs", type=int, nargs="+", default=[2, 3],
                    help="n_splits values to read, e.g. --rungs 2 3 4")
args = parser.parse_args()

# %%
""" Read the cached baseline and refined runs. Never trains: require_cached=True. """

base_model, base_record = uci_refinement.run(None, require_cached=True)
base = {**base_record, **uci_refinement.certified_metrics(base_model)}

rows = []
for n_splits in args.rungs:
    refined_model, refined_record = uci_refinement.run(n_splits, require_cached=True)
    refined = {**refined_record, **uci_refinement.certified_metrics(refined_model)}

    # the nominal trajectory is refinement-independent: same nominal model at every rung
    assert abs(base["mse_nominal"] - refined["mse_nominal"]) < 1e-9, (
        f"nominal MSE diverged at {n_splits=}: {base['mse_nominal']} vs {refined['mse_nominal']}"
    )

    leaves = n_splits ** uci_refinement.SPLIT_DIMS
    width_gain = 1 - refined["box_width"] / base["box_width"] if base["box_width"] else 0.0
    rows.append({"n_splits": n_splits, "leaves": leaves, "refined": refined, "width_gain": width_gain})

# %%
""" Table. """

print(
    f"UCI house-electric feature poisoning | eps={uci_refinement.EPSILON} | k_poison={uci_refinement.K_POISON} | "
    f"matmul={uci_refinement.INTERVAL_MATMUL} | strategy={uci_refinement.STRATEGY} | rungs {args.rungs}\n"
)
print(
    f"  {'leaves':>9s}  {'box width':>12s} {'box width':>12s} {'width':>7s}   "
    f"{'cert MSE':>10s} {'cert MSE':>10s}  {'nominal':>8s}"
)
print(
    f"  {'':>9s}  {'(baseline)':>12s} {'(refined)':>12s} {'gain':>7s}   "
    f"{'(baseline)':>10s} {'(refined)':>10s}"
)

for r in rows:
    refined = r["refined"]
    print(
        f"  {r['leaves']:>9d}  {base['box_width']:12.4e} {refined['box_width']:12.4e} {r['width_gain']:6.1%}   "
        f"{base['mse_worst']:10.4e} {refined['mse_worst']:10.4e}  {base['mse_nominal']:8.4e}"
    )

# %%
""" Plot: certified box width and certified worst-case test MSE against leaf count. """

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

axs[1].axhline(base["mse_worst"], linestyle=":",
               color=script_utils.colours["grey"], label="baseline (no refinement)")
axs[1].plot(leaves, [r["refined"]["mse_worst"] for r in rows], marker="o",
            color=script_utils.colours["green"], label="refined")
axs[1].axhline(base["mse_nominal"], linestyle="--",
               color=script_utils.colours["orange"], label="nominal")
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].set_xlabel("refinement leaves per fragment")
axs[1].set_ylabel("certified worst-case test MSE")
axs[1].legend(fontsize="x-small")

fig.suptitle(rf"UCI house-electric: certified tightness vs refinement leaves, "
             rf"$k={uci_refinement.K_POISON}$, $\epsilon={uci_refinement.EPSILON}$", fontsize="small")
script_utils.apply_figure_size(fig, script_utils.set_size(0.9, subplots, shrink_height=1.6), dpi=300)
fig_dir = script_utils.make_dirs()[3]
path = f"{fig_dir}/uci_refinement_sweep.pdf"
plt.savefig(path, dpi=300)
print(f"\n  figure: {path}")
