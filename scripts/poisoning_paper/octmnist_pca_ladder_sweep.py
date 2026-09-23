"""
Refinement-ladder experiment on OCT-MNIST PCA features: at the two threat-grid cells where refinement
moved certified accuracy most (d=15, eps=0.1, k=200 and 400), how the certificate tightens across a
ladder of partitions of the eps-ball. The run list and its rationale are in octmnist_pca_ladder.

Hypotheses:
    H1  Certified accuracy rises monotonically along the bisection ladder (2^5, 2^10, 2^15), with the
        largest gain at k=400.
    H2  Cutting every feature more finely (3^15) buys further certified accuracy beyond bisecting all
        15, which only a 4-GPU sharded run can afford.
    H3  At similar leaf counts, bisecting more features beats cutting fewer more finely (2^15 against
        3^10), as on CIFAR-10; the d=16 depth sweep found it only partly holds on OCT-MNIST.

Output: a table on stderr per k (box width, certified accuracy and cross-entropy, their gains over the
baseline, run time and number of GPUs) and ``octmnist_pca_ladder_sweep.json``. The figures are made
from that file.

Reads cached runs only, and exits listing any that are missing.

Key external dependencies: the sibling modules octmnist_pca, octmnist_pca_ladder and script_utils.
"""

import json
import os
import sys

import octmnist_pca
import octmnist_pca_ladder as ladder
import script_utils

missing = [ladder.run_args(k, refine) for k, refine in ladder.all_runs()
           if not os.path.isfile(ladder.cache_path(k, refine))]
if missing:
    sys.exit(f"{len(missing)} of {len(ladder.all_runs())} ladder runs not cached:\n  " + "\n  ".join(missing))

pca_state = octmnist_pca.fit_pca()
drusen_train, drusen_test, clean_train, clean_test = octmnist_pca.get_datasets(ladder.PCA_DIMS, pca_state)
model = octmnist_pca.get_pretrained_model(ladder.PCA_DIMS, pca_state)

results = []
for k in ladder.LADDER_KS:
    base_model, base_viol = ladder.run(k, None, model, drusen_train, clean_train)
    base = {**ladder.certified_metrics(base_model, drusen_test, clean_test), "violation": base_viol,
            "cost": ladder.cost_record(k, None)}
    rows = []
    for n_splits, n_dims in sorted(ladder.LADDER_SCHEDULE, key=ladder.leaves):
        refined_model, viol = ladder.run(k, (n_splits, n_dims), model, drusen_train, clean_train)
        refined = ladder.certified_metrics(refined_model, drusen_test, clean_test)
        assert abs(refined["nominal_acc"] - base["nominal_acc"]) < 1e-9, f"refinement moved the nominal model at {k=}"
        rows.append({"n_splits": n_splits, "n_dims": n_dims, "leaves": n_splits**n_dims, **refined,
                     "violation": viol, "cost": ladder.cost_record(k, (n_splits, n_dims))})
    results.append({"k": k, "baseline": base, "rungs": rows})

    print(f"\nOCT-MNIST PCA d={ladder.PCA_DIMS} k={k} eps={ladder.EPSILON} | nominal Drusen acc "
          f"{base['nominal_acc']:.4f}, CE {base['nominal_ce']:.4f}", file=sys.stderr)
    print(f"  {'split':>6s} {'leaves':>9s}  {'box':>9s} {'width':>7s}  {'cert acc':>8s} {'gain':>7s}  "
          f"{'cert CE':>8s} {'gain':>7s}  {'clean':>6s}  {'time':>9s} {'ranks':>5s} {'viol':>7s}", file=sys.stderr)
    for r in [{"n_splits": None, "leaves": 1, **base}] + rows:
        split = "none" if r["n_splits"] is None else f"{r['n_splits']}^{r['n_dims']}"
        cost = r["cost"] or {}
        seconds = f"{cost['seconds']:8.0f}s" if cost else f"{'--':>9s}"
        ranks = f"{cost['world_size']:>5d}" if cost else f"{'--':>5s}"
        print(f"  {split:>6s} {r['leaves']:>9d}  {r['box_width']:9.3e} {1 - r['box_width'] / base['box_width']:7.2%}  "
              f"{r['cert_acc']:8.4f} {r['cert_acc'] - base['cert_acc']:+7.4f}  {r['cert_ce']:8.4f} "
              f"{base['cert_ce'] - r['cert_ce']:+7.4f}  {r['cert_clean_acc']:6.4f}  {seconds} {ranks} "
              f"{r['violation']:7.1e}", file=sys.stderr)

unsound = [(res["k"], r["n_splits"], r["n_dims"]) for res in results for r in res["rungs"]
           if r["violation"] > ladder.VIOLATION_TOLERANCE]
if unsound:
    print(f"  WARNING: bound violations above {ladder.VIOLATION_TOLERANCE} at (k, n_splits, n_dims) = {unsound}",
          file=sys.stderr)

results_dir = script_utils.make_dirs()[0]
with open(f"{results_dir}/octmnist_pca_ladder_sweep.json", "w") as file:
    json.dump(results, file, indent=1)
print(f"\n  results: {results_dir}/octmnist_pca_ladder_sweep.json", file=sys.stderr)
