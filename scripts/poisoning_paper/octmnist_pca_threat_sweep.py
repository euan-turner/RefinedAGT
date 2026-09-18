# %%
"""
Threat-grid experiment on OCT-MNIST PCA features: how the certificate, and the gain from bisecting
every feature, depend on the attack size ``k`` and the poisoning radius ``eps``.

Hypotheses:
    H1  The certificate degrades as ``k`` and ``eps`` grow. The figure uses the pre-trained model's
        accuracy as the reference: a certificate below it cannot show that fine-tuning on
        poisonable data helped.
    H2  The gain from bisecting every feature grows with ``eps``, as the epsilon sweep found at a fixed
        ``k`` (octmnist_pca_epsilon_sweep.py), and varies smoothly with ``k``. An earlier run over the
        full ``k`` grid at eps=0.01 moved 1.5-2.4% across ``k``.

Mechanism. At d=15, for every cell of k in {20, 50, 100, 200, 400} x eps in {0.01, 0.02, 0.1}
(a subset of the original pixel-space grid of ``octmnist_sweep.py``), it compares two runs:
    - unrefined;
    - every one of the 15 features bisected (2^15 = 32768 leaves).
Hyperparameters are the pipeline's tuned point (selected at d=16, applied at d=15 so that bisecting
every feature stays affordable). The nominal model is identical across the grid; the script asserts
this. The gain at a cell is the relative reduction in box width against the unrefined run.

Threat model: as octmnist_pca. The ball is on the scaled PCA features, not the pixels, so these
``eps`` values are not the pixel-space radii of ``octmnist_sweep.py``, even where the numbers match.

Output:
    - a per-cell table on stderr;
    - ``octmnist_pca_threat_sweep.json``;
    - a figure with three panels, each against ``k`` with one line per ``eps``:
        - certified accuracy, unrefined and refined, with the nominal and pre-trained references
          (H1);
        - box-width reduction (H2);
        - certified accuracy gained.

Launch: ``torchrun --nproc-per-node 2 octmnist_pca_threat_sweep.py``. Under torchrun, the refined
runs' leaves are sharded across the GPUs, and the cheap unrefined runs are repeated on every rank.
With plain ``python`` it runs in one process and reads the same cache, so a finished sweep can be
re-plotted without spare GPUs.

Key external dependencies: torch (torch.distributed), matplotlib, ``abstract_gradient_training``,
and the sibling modules ``octmnist_pca`` and ``script_utils``.
"""

import copy
import json
import os
import sys

import torch
import torch.distributed as dist
import matplotlib.pyplot as plt

import abstract_gradient_training as agt

import octmnist_pca
import script_utils

# %%
""" Script parameters. """

PCA_DIMS = 15
# A subset of octmnist_sweep.py's adversary-1 grid (0..600 in steps of 20): on the full eps=0.01 column
# the refinement gain moved smoothly with k (1.5-2.4% box width), so five points cover its shape.
K_POISONS = [20, 50, 100, 200, 400]
EPSILONS = [0.01, 0.02, 0.1]  # octmnist_sweep.py, adversary 1 -- here in scaled PCA feature units

N_SPLITS = 2
SPLIT_DIMS = PCA_DIMS  # bisect every feature: 2 ** 15 leaves
STRATEGY = "sensitivity"  # immaterial when every feature is split, kept to match the sibling sweeps
MAX_LEAVES = N_SPLITS**SPLIT_DIMS
# Peak memory scales with the chunk, not with the leaf count, and throughput does not: one 2000-sample
# fragment measured 0.69 / 0.72 / 0.74 ms per leaf at chunks of 16 / 32 / 64, peaking at 2.3 / 4.4 /
# 8.7 GiB. 16 is as fast and leaves room on a GPU shared with other jobs.
LEAF_CHUNK = 16

VIOLATION_TOLERANCE = 1e-3  # above this a run's certificate is unsound
TAG = "threat"


def make_config(k_poison, epsilon, refine):
    """Tuned config at the given attack, bisecting every feature when ``refine`` is true. The leaves are
    sharded across ranks whenever the script is launched with more than one process."""
    config = copy.deepcopy(octmnist_pca.FINETUNE_CONFIG)
    config.k_poison = k_poison
    config.epsilon = epsilon
    if refine:
        config.input_refinement = agt.InputRefinementConfig(
            n_splits=N_SPLITS, n_dims=SPLIT_DIMS, strategy=STRATEGY, max_leaves=MAX_LEAVES,
            leaf_chunk=LEAF_CHUNK, shard_leaves=octmnist_pca.world_size() > 1,
        )
    return config


def cost_record(config):
    """The ``{"seconds", "world_size", "peak_gib"}`` record of a cached run, or None if it has none."""
    path = f"{octmnist_pca.cache_path(config, PCA_DIMS, TAG)}.json"
    if not os.path.isfile(path):
        return None
    with open(path) as file:
        return json.load(file)


def certified_metrics(bounded_model, drusen_test, clean_test):
    """Certified box width and, over the box, worst-case and nominal Drusen accuracy and cross-entropy,
    and clean-split accuracy -- the column that exposes a collapsed predict-abnormal model."""
    acc_w, acc_n, _ = agt.test_metrics.test_accuracy(bounded_model, *drusen_test.tensors, epsilon=0)
    ce_w, ce_n, _ = agt.test_metrics.test_cross_entropy(bounded_model, *drusen_test.tensors, epsilon=0)
    clean_w, clean_n, _ = agt.test_metrics.test_accuracy(bounded_model, *clean_test.tensors, epsilon=0)
    box_width = sum((u - l).sum().item() for l, u in zip(bounded_model.param_l, bounded_model.param_u))
    return {
        "cert_acc": acc_w, "nominal_acc": acc_n, "cert_ce": ce_w, "nominal_ce": ce_n,
        "cert_clean_acc": clean_w, "nominal_clean_acc": clean_n, "box_width": box_width,
    }


def log(message):
    if octmnist_pca.rank() == 0:
        print(message, file=sys.stderr, flush=True)


# %%
""" Shared data and the pre-trained model. Both are independent of the attack and of refinement. """

octmnist_pca.init_distributed()
PCA_STATE = octmnist_pca.fit_pca()
DRUSEN_TRAIN, DRUSEN_TEST, CLEAN_TRAIN, CLEAN_TEST = octmnist_pca.get_datasets(PCA_DIMS, PCA_STATE)
MODEL = octmnist_pca.get_pretrained_model(PCA_DIMS, PCA_STATE)
PRETRAINED_ACC = agt.test_metrics.test_accuracy(
    agt.bounded_models.IntervalBoundedModel(copy.deepcopy(MODEL)), *DRUSEN_TEST.tensors, epsilon=0
)[1]

# %%
""" Sweep: at every (eps, k), the unrefined run and the bisect-all run. """

log(f"OCT-MNIST (PCA features, d={PCA_DIMS}) adversary 1 | bisect all {SPLIT_DIMS} features, "
    f"{MAX_LEAVES} leaves | leaf_chunk={LEAF_CHUNK} | world size {octmnist_pca.world_size()}\n")
log(f"  {'eps':>5s} {'k':>4s}  {'box width':>10s} {'box width':>10s} {'width':>7s}  {'cert acc':>8s} "
    f"{'cert acc':>8s} {'acc':>7s}  {'cert CE':>8s} {'cert CE':>8s}  {'clean':>6s}  {'time':>6s} "
    f"{'peak':>6s}  {'viol':>7s}")
log(f"  {'':>5s} {'':>4s}  {'(base)':>10s} {'(refined)':>10s} {'gain':>7s}  {'(base)':>8s} "
    f"{'(refined)':>8s} {'gain':>7s}  {'(base)':>8s} {'(refined)':>8s}  {'cert':>6s}  {'(s)':>6s} "
    f"{'(GiB)':>6s}  {'(max)':>7s}")

rows = []
for epsilon in EPSILONS:
    for k in K_POISONS:
        base_config = make_config(k, epsilon, refine=False)
        base_model, base_viol = octmnist_pca.run_certified(
            base_config, PCA_DIMS, MODEL, DRUSEN_TRAIN, CLEAN_TRAIN, tag=TAG
        )
        base = certified_metrics(base_model, DRUSEN_TEST, CLEAN_TEST)

        refined_config = make_config(k, epsilon, refine=True)
        refined_model, refined_viol = octmnist_pca.run_certified(
            refined_config, PCA_DIMS, MODEL, DRUSEN_TRAIN, CLEAN_TRAIN, tag=TAG
        )
        refined = certified_metrics(refined_model, DRUSEN_TEST, CLEAN_TEST)
        cost = cost_record(refined_config)

        # the nominal trajectory is plain SGD, independent of the attack and of refinement
        assert abs(base["nominal_acc"] - refined["nominal_acc"]) < 1e-9, f"nominal model moved at {epsilon=}, {k=}"

        row = {
            "epsilon": epsilon, "k": k, "baseline": base, "refined": refined, "cost": cost,
            "width_gain": 1 - refined["box_width"] / base["box_width"] if base["box_width"] else 0.0,
            "acc_gain": refined["cert_acc"] - base["cert_acc"],
            "violation": max(base_viol, refined_viol),
        }
        rows.append(row)
        seconds = f"{cost['seconds']:6.0f}" if cost else f"{'--':>6s}"
        peak = f"{cost['peak_gib']:6.1f}" if cost and cost["peak_gib"] is not None else f"{'--':>6s}"
        log(f"  {epsilon:>5} {k:>4d}  {base['box_width']:10.3e} {refined['box_width']:10.3e} "
            f"{row['width_gain']:7.2%}  {base['cert_acc']:8.4f} {refined['cert_acc']:8.4f} "
            f"{row['acc_gain']:+7.4f}  {base['cert_ce']:8.4f} {refined['cert_ce']:8.4f}  "
            f"{refined['cert_clean_acc']:6.4f}  {seconds} {peak}  {row['violation']:7.1e}")

nominal = rows[0]["baseline"]
log(f"\n  pre-trained Drusen accuracy {PRETRAINED_ACC:.4f} | fine-tuned nominal Drusen {nominal['nominal_acc']:.4f}, "
    f"clean {nominal['nominal_clean_acc']:.4f} (0.667 = predict-abnormal-everywhere)")
unsound = [(r["epsilon"], r["k"]) for r in rows if r["violation"] > VIOLATION_TOLERANCE]
if unsound:
    log(f"  WARNING: bound violations above {VIOLATION_TOLERANCE} at (eps, k) = {unsound}; those cells are unsound.")

# %%
""" Results table and plot, written by rank 0. """

if octmnist_pca.rank() == 0:
    results_dir, _, _, fig_dir = script_utils.make_dirs()
    with open(f"{results_dir}/octmnist_pca_threat_sweep.json", "w") as file:
        json.dump(rows, file, indent=1)

    subplots = (1, 3)
    fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300)
    colours = list(script_utils.colours.values())
    for colour, epsilon in zip(colours, EPSILONS):
        cell = [r for r in rows if r["epsilon"] == epsilon]
        ks = [r["k"] for r in cell]
        axs[0].plot(ks, [r["baseline"]["cert_acc"] for r in cell], marker="s", ms=2.5, linestyle="--", color=colour)
        axs[0].plot(ks, [r["refined"]["cert_acc"] for r in cell], marker="o", ms=2.5, color=colour,
                    label=rf"$\epsilon={epsilon}$")
        axs[1].plot(ks, [r["width_gain"] for r in cell], marker="o", ms=2.5, color=colour, label=rf"$\epsilon={epsilon}$")
        axs[2].plot(ks, [r["acc_gain"] for r in cell], marker="o", ms=2.5, color=colour, label=rf"$\epsilon={epsilon}$")

    axs[0].axhline(nominal["nominal_acc"], linestyle=":", color=script_utils.colours["grey"], label="fine-tuned nominal")
    axs[0].axhline(PRETRAINED_ACC, linestyle="-.", color=script_utils.lb_color, label="pre-trained")
    axs[0].plot([], [], linestyle="--", color=script_utils.colours["grey"], label="unrefined")
    axs[0].set_ylim(0, 1.0)
    axs[0].set_ylabel("certified Drusen accuracy")
    axs[0].legend(fontsize="xx-small")
    axs[1].set_ylim(0, None)
    axs[1].set_ylabel("box width reduction")
    axs[2].axhline(0, linewidth=0.5, color=script_utils.colours["grey"])
    axs[2].set_ylabel("certified accuracy gained")
    for ax in axs:
        ax.set_xlabel("attack size $k$")
        ax.set_xlim(0, K_POISONS[-1])

    fig.suptitle(rf"OCT-MNIST on PCA features ($d={PCA_DIMS}$), adversary 1: bisecting all {SPLIT_DIMS} "
                 rf"features ($2^{{{SPLIT_DIMS}}}$ leaves)", fontsize="small")
    script_utils.apply_figure_size(fig, script_utils.set_size(1.0, subplots, shrink_height=2.4), dpi=300)
    path = f"{fig_dir}/octmnist_pca_threat_sweep.pdf"
    plt.savefig(path, dpi=300)
    log(f"\n  figure: {path}")

if dist.is_initialized():
    dist.destroy_process_group()
