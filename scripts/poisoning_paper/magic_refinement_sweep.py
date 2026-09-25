# %%
"""
Leaf-count scaling of input-ball refinement for MAGIC gamma-telescope feature poisoning.

Hypothesis. As on half-moons, splitting the feature-poisoning ball more finely keeps tightening the
certificate, with diminishing returns. MAGIC has 10 features, too many to cut every one finely, so
the ladder refines in half-steps: bisect the 5 most sensitive features (2^5), then all 10 (2^10), then
cut the sensitive half one step finer (3^5x2^5), then the rest (3^10), then the sensitive half again
(4^5x3^5) and the rest (4^10). Those rungs were computed (locally, on two GPUs) and are the default.
The pattern continues to 6^5x5^5 (24.3M leaves, ~12 h on 4 Isambard GPUs; magic_ladder.sh), but those
rungs were never run.

Mechanism. One attack (k=200, eps=0.01) and one nominal model; the script asserts that the nominal
model is the same at every rung. The unrefined run is compared with runs at each rung in
``--rungs`` (``N^D`` or ``N^DxM^E``, see magic_refinement.parse_rung). The gain is the relative reduction in box width against the unrefined run. See
magic_refinement for the task and the metrics.

Output: a table on stdout (box width and certified accuracy, unrefined and refined, width gain, run
time and number of GPUs), ``magic_refinement_sweep.json``, and a figure of box width and certified
accuracy against leaf count.

Reads cached runs only (written by magic_run.py), and fails on any missing rung.

Usage: python magic_refinement_sweep.py [--rungs 2^5 2^10 3^5x2^5 3^10 4^5x3^5 ...]

Key external dependencies: matplotlib, and the sibling modules magic_refinement and script_utils.
"""

import argparse
import json

import matplotlib.pyplot as plt

import magic_refinement
import script_utils

# the rungs computed (locally, on two GPUs); the rest of the ladder, 5^5x4^5 to 6^5x5^5, was never run
LADDER = ["2^5", "2^10", "3^5x2^5", "3^10", "4^5x3^5", "4^10"]


parser = argparse.ArgumentParser()
parser.add_argument("--rungs", type=magic_refinement.parse_rung, nargs="+",
                    default=[magic_refinement.parse_rung(r) for r in LADDER],
                    help="rungs to read, e.g. --rungs 2^5 2^10 3^5x2^5; default the computed rungs (LADDER)")
args = parser.parse_args()

# %%
""" Read the cached baseline and refined runs. Never trains: require_cached=True. """

base_model, base_record = magic_refinement.run(None, require_cached=True)
base = {**base_record, **magic_refinement.certified_metrics(base_model)}
majority = magic_refinement.majority_accuracy()

rows = []
for refine in args.rungs:
    label = magic_refinement.rung_label(refine)
    refined_model, refined_record = magic_refinement.run(refine, require_cached=True)
    refined = {**refined_record, **magic_refinement.certified_metrics(refined_model)}

    # the nominal trajectory is refinement-independent: same nominal model at every rung
    assert abs(base["nominal_acc"] - refined["nominal_acc"]) < 1e-9, (
        f"nominal accuracy diverged at {label}: {base['nominal_acc']} vs {refined['nominal_acc']}"
    )

    width_gain = 1 - refined["box_width"] / base["box_width"] if base["box_width"] else 0.0
    # n_splits and n_dims are the primary (finest, most sensitive) tier; "tiers" holds every tier
    (n_splits, n_dims), *_ = refine
    rows.append({"label": label, "tiers": refine, "n_splits": n_splits, "n_dims": n_dims,
                 "leaves": magic_refinement.n_leaves(refine), "refined": refined, "width_gain": width_gain})

# %%
""" Table and results file. """

print(
    f"MAGIC feature poisoning | eps={magic_refinement.EPSILON} | k_poison={magic_refinement.K_POISON} | "
    f"strategy={magic_refinement.STRATEGY} | majority acc {majority:.4f}\n"
)
print(
    f"  {'split':>9s} {'leaves':>9s}  {'box width':>12s} {'box width':>12s} {'width':>7s}   "
    f"{'cert acc':>9s} {'cert acc':>9s}  {'nominal':>8s}  {'time':>8s} {'ranks':>5s} {'viol':>7s}"
)
print(f"  {'':>9s} {'':>9s}  {'(baseline)':>12s} {'(refined)':>12s} {'gain':>7s}   {'(baseline)':>9s} {'(refined)':>9s}")
for r in rows:
    refined = r["refined"]
    print(
        f"  {r['label']:>9s} {r['leaves']:>9d}  {base['box_width']:12.4e} "
        f"{refined['box_width']:12.4e} {r['width_gain']:6.1%}   {base['cert_acc_worst']:9.4f} "
        f"{refined['cert_acc_worst']:9.4f}  {base['nominal_acc']:8.4f}  {refined['seconds']:7.0f}s "
        f"{refined['world_size']:>5d} {refined['violation']:7.1e}"
    )

results_dir, _, _, fig_dir = script_utils.make_dirs()
with open(f"{results_dir}/magic_refinement_sweep.json", "w") as file:
    json.dump({"baseline": base, "majority_acc": majority, "rungs": rows}, file, indent=1)

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

fig.suptitle(rf"MAGIC: certified tightness vs refinement leaves, $k={magic_refinement.K_POISON}$, "
             rf"$\epsilon={magic_refinement.EPSILON}$", fontsize="small")
script_utils.apply_figure_size(fig, script_utils.set_size(0.9, subplots, shrink_height=1.6), dpi=300)
path = f"{fig_dir}/magic_refinement_sweep.pdf"
plt.savefig(path, dpi=300)
print(f"\n  figure: {path}")
