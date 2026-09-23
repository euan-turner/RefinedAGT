# Input-ball refinement for poison-certified training: hypotheses, results, reproduction

Purpose: the hypotheses tested by the poisoning-paper refinement experiments, the result for each one
as run on Isambard-AI in September 2026, and how to rerun them. The experiments cover CIFAR-10 on PCA
features, OCT-MNIST on PCA features, UCI house-electric and half-moons.

Key external dependencies: Slurm, uv, torch 2.13.0 (`+cu126` on aarch64, from `uv.lock`), NCCL via
`torchrun --standalone`.

All experiment scripts are in [`../poisoning_paper/`](../poisoning_paper/), and all job scripts are in
this directory. Every script's header docstring states its hypothesis and mechanism in full. This file
summarises them.

---

## 1. Terms

- **Threat model.** In every training batch, up to `k` samples may have their input features moved
  anywhere within an l_inf ball of radius `eps`. The attacker has full knowledge and can adapt.
- **Certificate.** Abstract Gradient Training returns a parameter box that contains every model any
  such attack could produce. It reports:
  - **Certified accuracy / cross-entropy / MSE:** the worst case over the box.
  - **Certified clean accuracy:** exposes a model that has collapsed onto one class.
  - **Box width:** the sum of the per-parameter interval widths. Smaller means tighter.
- **Refinement.** Each poisoned sample's eps-ball is cut along its `n_dims` most sensitive input
  coordinates into `n_splits` pieces each. That gives `n_splits^n_dims` **leaves**, which are bounded
  separately and then combined. **Bisecting** means `n_splits = 2`. The **gain** is the relative
  reduction in box width against the unrefined run at the same attack.
- The nominal model (trained with no attack) is identical across attacks and refinement settings;
  every sweep asserts this.
- **Soundness.** Every run reported a floating-point bound violation of 0.0, so no float64 reruns
  were needed.

The dataset-specific setup (splits, features, model, hyperparameter selection) is in the docstrings of
[`cifar_pca.py`](../poisoning_paper/cifar_pca.py), [`octmnist_pca.py`](../poisoning_paper/octmnist_pca.py)
and [`uci_refinement.py`](../poisoning_paper/uci_refinement.py).

Figures are written to `$AGT_ROOT/.figures/`. The local copy of this campaign's outputs is in
`scripts/poisoning_paper/.isambard3/` (gitignored): figures in `isambard-figures/`, job logs and
result records in `isambard-logs3/`.

---

## 2. Hypotheses and results

Verdicts: **holds**, **partly holds** or **does not hold**. Unless stated otherwise, numbers are on the
test split. "(inferred)" marks a hypothesis that was not written down before the runs; it was
reconstructed from the script's design.

### 2.1 CIFAR-10 on PCA features

Binary vehicle-vs-animal classification. Automobile images are the poisonable data. The model is
`d -> H ReLU -> 1`, with hyperparameters selected per `d` on a validation split by
[`cifar_pca_selection.py`](../poisoning_paper/cifar_pca_selection.py). Every run is launched through
[`cifar_pca_run.py`](../poisoning_paper/cifar_pca_run.py). The run list is
[`cifar_pca_manifest.py`](../poisoning_paper/cifar_pca_manifest.py). Nominal model at d=20: held-out
accuracy 0.795, clean 0.789.

**Threat grid:** d=20, k in {10, 25, 50, 100, 200} x eps in {0.005, 0.01, 0.02, 0.05, 0.1}. Each cell
has four runs: unrefined; top 12 bisected (2^12 leaves); all 20 bisected (2^20); and unrefined at eps/2.
Script: [`cifar_pca_threat_sweep.py`](../poisoning_paper/cifar_pca_threat_sweep.py). Figure:
`cifar_pca_threat_sweep.pdf`.

| | Hypothesis | Verdict | Result |
|---|---|---|---|
| H1 | The certificate is informative across the grid, and `k` degrades it more than `eps` | partly holds | `k` dominates: at eps=0.05 the box grows 15x from k=10 to 200 (1.61 to 24.7), while at k=50 it grows 2.3x across a 20x range of eps (3.55 to 8.25). At k=10, eps barely matters (certified acc 0.786 to 0.777). Informative up to k=100: certified acc is at least 0.646. It is not informative at k=200, eps of 0.05 or more: certified acc 0.51-0.54 and certified clean acc 0.42-0.47. |
| H2 | Bisecting every coordinate gives a modest gain: about 5-7% width and at most +0.02 certified acc | partly holds | Up to k=100: 2.2-7.2% width, at most +0.010 acc. At k=200 the gain is larger: 5.2-11.1% width, up to +0.027 acc (k=200, eps=0.05: 0.536 to 0.563). |
| H2b | Bisect-all falls well short of the unrefined eps/2 run | holds | In every cell. The eps/2 reduction is 11.5-46.4%, against 2.2-11.1% for bisect-all. At k=100, eps=0.05: 16.6% against 6.6%. |
| H3 | The gain peaks at eps 0.02-0.05 and falls at 0.1 | partly holds | Holds up to k=100: the peak is at 0.05 for k of 50 or less and at 0.02 for k=100, and every k falls at 0.1. At k=200 the gain is largest at the smallest eps (11.1% at 0.005) and falls monotonically. |
| H4 | The gain grows with `k` | holds | Monotone in `k` at every eps. Strongest at eps=0.005 (2.2% to 11.1%), weakest at eps=0.1 (4.6% to 5.2%). |

**Refinement depth:** d=20, k=100, eps=0.05. A bisection ladder (2^4 to 2^20 leaves) against
equal-cost alternatives that cut fewer coordinates more finely. Script:
[`cifar_pca_leaf_sweep.py`](../poisoning_paper/cifar_pca_leaf_sweep.py). Figure:
`cifar_pca_leaf_sweep.pdf`.

| | Hypothesis | Verdict | Result |
|---|---|---|---|
| H2 | The gain grows steadily and modestly with bisected coordinates, to about 6% at all 20 | holds | 1.09% (2^4), 2.41% (2^8), 3.69% (2^12), 5.09% (2^16), 6.64% (2^20): 1.1-1.6 points per 4 more coordinates. Certified acc 0.660 to 0.668; certified CE 0.639 to 0.624. |
| H5 | At equal leaf count, bisecting more coordinates beats cutting fewer more finely | holds | At all three costs: 2^12 3.69% against 16^3 1.55%; 2^16 5.09% against 4^8 3.85%; 2^20 6.64% against 4^10 4.79%. |

**Feature width:** k=100, eps=0.05 at d in {20, 22, 24}, each width using its own selected
hyperparameters. Four runs per width: unrefined, top 12, all d, and eps/2. Script:
[`cifar_pca_dims_sweep.py`](../poisoning_paper/cifar_pca_dims_sweep.py). Figure:
`cifar_pca_dims_sweep.pdf`.

Selection chose pre-training radius `pt_epsilon` = 0.01 at d=20 but 0.02 at d=22 and 24, and that
radius moves box width by more than `d` does. The **pre-training-radius control**
([`cifar_pca_pt_epsilon_control.py`](../poisoning_paper/cifar_pca_pt_epsilon_control.py)) reruns d=22
and 24 with `pt_epsilon` = 0.01. Its results are printed as JSON lines in the `agt-pt-control-*` job
logs; no script aggregates them.

| d | `pt_epsilon` | base box | vs d=20 | top-12 gain | all-d gain | eps/2 gain | cert acc (base / all d) | nominal acc | bisect-all hours (4 GPUs) |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 0.01 | 13.56 | 1.00 | 3.69% | 6.64% | 16.59% | 0.660 / 0.668 | 0.795 | 0.25 |
| 22 | 0.01 | 13.92 | 1.03 | 3.27% | 6.59% | 16.78% | 0.666 / 0.671 | 0.802 | 1.08 |
| 24 | 0.01 | 15.51 | 1.14 | 2.81% | 6.21% | 15.93% | 0.653 / 0.658 | 0.791 | 4.48 |
| 22 | 0.02 (selected) | 10.55 | 0.78 | 3.68% | 7.05% | 18.85% | 0.691 / 0.695 | 0.786 | 1.08 |
| 24 | 0.02 (selected) | 12.10 | 0.89 | 3.20% | 6.71% | 17.87% | 0.678 / 0.690 | 0.780 | 4.48 |

| | Hypothesis | Verdict | Result |
|---|---|---|---|
| H6a | The bisect-all gain is roughly flat in `d` (within about 1 point) | holds | 6.64 / 6.59 / 6.21% at fixed `pt_epsilon`, a spread of 0.43 points; 6.64 / 7.05 / 6.71% at the selected radii. The top-12 gain falls with `d` (3.69 to 2.81%), because 12 of `d` coordinates is a shrinking share. |
| H6b | The unrefined certificate loosens with `d` | holds, once `pt_epsilon` is fixed | +2.7% box width at d=22 and +14.4% at d=24. At the selected radii the effect is hidden: d=22 comes out 22% *tighter* than d=20, because raising `pt_epsilon` from 0.01 to 0.02 shrinks the box by 22-24%. |
| H6b' | Nominal accuracy rises slightly with `d` | does not hold | 0.795 / 0.802 / 0.791 is not monotone. Differences of about 0.01 on 1000 test points are within sampling noise. |

Summary figure for all three CIFAR experiments:
[`cifar_pca_plots.py`](../poisoning_paper/cifar_pca_plots.py), writing `cifar_pca_summary.pdf`.

### 2.2 OCT-MNIST on PCA features

Normal vs abnormal retinal scans. Drusen scans are the poisonable data. A fixed PCA projection replaces
the pixel-space convolutional features, so that the input is small enough to partition. The model is
`d -> 256 ReLU -> 1`, tuned at d=16. The pipeline is
[`octmnist_pca.py`](../poisoning_paper/octmnist_pca.py). Nominal model at d=16: Drusen accuracy
0.932 (matching the pixel-space convolutional model), clean 0.727.

| Experiment and script | Hypothesis | Verdict | Result | Figure |
|---|---|---|---|---|
| Feature width: k=50, eps=0.01, d in {4, ..., 20}, bisecting min(d, 12) coordinates. [`octmnist_pca_refinement_sweep.py`](../poisoning_paper/octmnist_pca_refinement_sweep.py) | With few enough coordinates to bisect all of them, refinement measurably tightens the certificate, and the gain grows with `d` (inferred) | does not hold | The gain is flat at 1.2-1.7% for every `d`, whether every coordinate is bisected or only 12 are. Certified accuracy never moves. At eps=0.01 the input ball is a small part of the box. At d=4 and 6 the fixed schedule collapses towards "abnormal everywhere" (clean acc 0.684, 0.668). | `octmnist_pca_refinement_sweep.pdf` |
| Poisoning radius: d=16, k=50, eps in {0.005, ..., 0.2}, top 12 of 16 bisected (4096 leaves). [`octmnist_pca_epsilon_sweep.py`](../poisoning_paper/octmnist_pca_epsilon_sweep.py) | The gain grows with `eps` | holds | 0.61, 1.17, 2.30, 5.30, 9.25 and 9.66% across eps = 0.005 to 0.2, levelling off above 0.1. Certified accuracy improves only at eps of 0.1 or more (+0.008, +0.012). | `octmnist_pca_epsilon_sweep.pdf` |
| Refinement depth: d=16, k=50, eps=0.1. Ladder 2^4 to 2^16; alternatives 8^4 and 4^8. [`octmnist_pca_leaf_sweep.py`](../poisoning_paper/octmnist_pca_leaf_sweep.py) | H1: returns diminish but do not vanish, up to bisecting all 16 | holds | 3.41% (2^4), 7.67% (2^8), 9.25% (2^12), 10.74% (2^16). Certified acc 0.852 / 0.856 / 0.856 / 0.860; certified CE keeps falling (0.5409 to 0.5375). | `octmnist_pca_leaf_sweep.pdf` |
| (same) | H2: at equal leaf count, bisecting more coordinates beats cutting fewer more finely | partly holds | At 4096 leaves, yes: 2^12 9.25% against 8^4 6.10%. At 65536 leaves, no: 4^8 11.37% against 2^16 10.74%, at the same certified CE. | (same) |
| Threat grid: d=15, k in {20, ..., 400} x eps in {0.01, 0.02, 0.1}, unrefined against all 15 bisected (2^15 leaves). [`octmnist_pca_threat_sweep.py`](../poisoning_paper/octmnist_pca_threat_sweep.py) | H1: the certificate degrades with `k` and `eps`, judged against the pre-trained model's Drusen accuracy of 0.316 (inferred) | holds | Unrefined certified acc falls from 0.844 to 0.676 across k at eps=0.01, and from 0.828 to 0.312 at eps=0.1. It stays above the pre-trained 0.316 in every cell except k=400, eps=0.1 (0.312 unrefined, 0.364 refined). | `octmnist_pca_threat_sweep.pdf` |
| (same) | H2: the gain grows with `eps` and varies smoothly with `k` (inferred) | holds | The gain is set by eps and nearly constant in `k`: 1.5-2.3% at 0.01, 2.8-3.0% at 0.02, 10.2-10.5% at 0.1. The accuracy gain is largest at large `k` and `eps` (+0.040 at k=200, +0.052 at k=400, both at eps=0.1). | (same) |

Model selection, from the summary figure: the pre-training radius decides whether the certificate
means anything. At k=50, eps=0.01:

| `pt_epsilon` | box width | certified acc | nominal acc | clean acc |
|---|---|---|---|---|
| 0.01 | 16.6 | 0.536 | 0.912 | 0.760 |
| 0.02 | 7.80 | 0.788 | 0.908 | 0.763 |
| 0.05 (used) | 1.06 | 0.900 | 0.932 | 0.727 |
| 0.1 | 0.028 | 1.000 | 1.000 | 0.667 (collapsed: "abnormal everywhere") |

Summary figure: [`octmnist_pca_plots.py`](../poisoning_paper/octmnist_pca_plots.py), writing
`octmnist_pca_summary.pdf`. It leaves out the feature-width sweep.

### 2.3 UCI house-electric

Regression on 11 continuous features with a one-layer MLP (64 units) and MSE loss, at k=200 of each
10000-sample batch and eps=0.01. Every one of the 11 features is cut into `n_splits` pieces. The module
is [`uci_refinement.py`](../poisoning_paper/uci_refinement.py), runs go through
[`uci_run.py`](../poisoning_paper/uci_run.py), and the aggregation script is
[`uci_refinement_sweep.py`](../poisoning_paper/uci_refinement_sweep.py). Figure:
`uci_refinement_sweep.pdf`.

**Hypothesis (inferred):** once every coordinate is split, cutting each one more finely keeps
tightening the certificate, with diminishing returns. **Holds.**

| `n_splits` | leaves | box width | gain | certified MSE (nominal 0.0529) | wall time |
|---|---|---|---|---|---|
| unrefined | 1 | 0.444 | | 0.0649 | 18 s, 1 GPU |
| 2 | 2048 | 0.327 | 26.3% | 0.0611 | 154 s, 1 GPU |
| 3 | 177147 | 0.290 | 34.6% | 0.0599 | 46 min, 4 GPUs |
| 4 | 4194304 | 0.272 | 38.7% | 0.0593 | 17.6 h, 4 GPUs |

### 2.4 Half-moons

Binary classification on 2-D half-moons with quadratic and cubic features appended (6 inputs), a
`6 -> 128 ReLU -> 2` model, k=200, eps=0.01. Every one of the 6 features is cut into `n_splits` pieces.
Script: [`halfmoons_refinement_sweep.py`](../poisoning_paper/halfmoons_refinement_sweep.py). This
campaign's version trained in the job and had no cache, so its job log and
`halfmoons_refinement_sweep.pdf` are the result. The script now reads cached runs written by
[`halfmoons_run.py`](../poisoning_paper/halfmoons_run.py) (section 5).

**Hypothesis (inferred):** as UCI, finer cuts on every coordinate keep tightening the certificate,
with diminishing returns. **Holds**, with by far the largest gains of any dataset.

| `n_splits` | leaves | box width (unrefined 45.2) | gain | certified acc (unrefined 0.386, nominal 0.956) |
|---|---|---|---|---|
| 2 | 64 | 30.0 | 33.8% | 0.554 |
| 3 | 729 | 25.5 | 43.6% | 0.670 |
| 4 | 4096 | 23.4 | 48.2% | 0.706 |
| 5 | 15625 | 22.2 | 50.9% | 0.736 |
| 6 | 46656 | 21.4 | 52.6% | 0.762 |

### 2.5 Across datasets

Refinement pays most where the input ball is a large share of the certificate and every coordinate
can be split finely: half-moons (6 inputs, 34-53%) and UCI (11 inputs, 26-39%). On the PCA pipelines
(15-24 inputs), bisecting every coordinate buys 2-11% of box width and at most about 0.05 of
certified accuracy, for 10^3-10^4 times the GPU time of an unrefined run (CIFAR d=20: 2 s unrefined
against 913 s on 4 GPUs). There, the attack size and the
pre-training radius move the certificate far more than refinement does.

---

## 3. Reproducing the results

Legend: **[login]** Isambard login node, **[job]** Slurm batch job.

### 3.1 Platform facts

| | |
|---|---|
| Node | 4x GH200 (aarch64 Grace + 96 GB H100-class GPU); `--gpus=1` allocates one whole superchip |
| Driver | 565.57 (CUDA 12.7 native), hence torch `+cu126` on aarch64 |
| Slurm | `--nodes=1 --gpus=N`, default partition, no `--account`, max `--time` 24 h; independent jobs run on separate nodes at the same time |
| Login nodes | 1 core, 4 GiB per session: enough for `uv sync`, downloads and manifests; not for training or aggregation |
| Storage | `$HOME` 100 GiB (over quota blocks SSH); `$PROJECTDIR` persistent, not backed up; `$SCRATCHDIR` 5 TiB |
| Etiquette | no `squeue`/`sinfo` polling loops (acceptable-use policy); use `sacct` after the fact |

### 3.2 One-time setup [login]

```bash
# 1. uv (installs to ~/.local/bin)
curl --location --silent --show-error --fail https://astral.sh/uv/install.sh | sh

# 2. repository on project storage (GitHub needs an SSH key on Isambard, or clone over HTTPS with a token)
mkdir -p "$PROJECTDIR/$USER"
git clone -b refine git@github.com:euan-turner/RefinedAGT.git "$PROJECTDIR/$USER/AbstractGradientTraining"

# 3. environment: repository, cache root ($AGT_ROOT), uv locations, venv
source "$PROJECTDIR/$USER/AbstractGradientTraining/scripts/isambard/env.sh"

# 4. python + locked environment (aarch64 resolves torch 2.13.0+cu126)
cd "$AGT_REPO"
uv python install 3.12
uv sync --frozen --extra experiments       # if the login node stalls: srun --nodes=1 --gpus=1 --time=00:30:00 uv sync ...
source scripts/isambard/env.sh             # re-source: activates the venv now that it exists

# 5. stage datasets (or: cd "$AGT_ROOT/logs" && sbatch "$AGT_REPO/scripts/isambard/stage.sbatch")
mkdir -p "$AGT_ROOT/.data" "$AGT_ROOT/logs" "$AGT_ROOT/manifests"
python -c "import os, torchvision; [torchvision.datasets.CIFAR10(root=os.environ['AGT_ROOT'] + '/.data', train=t, download=True) for t in (True, False)]"
python -c "from medmnist import OCTMNIST; OCTMNIST(split='train', download=True)"   # -> ~/.medmnist/octmnist.npz
python -c "import uci_datasets; uci_datasets.Dataset('houseelectric')"              # bundled: confirms it loads
```

In every later session, source [`env.sh`](env.sh) before `sbatch`. Jobs inherit its variables and
re-source it to activate the venv. Update code with `git pull && uv sync --frozen --extra
experiments`, and never while jobs are running from the checkout: a change to any config default or
`LEAF_CHUNK` changes cache keys.

### 3.3 Run order

Submit every job from `$AGT_ROOT/logs`, because job logs (`%x_%j.out`) are written to the submit
directory:

```bash
cd "$AGT_ROOT/logs"
S="$AGT_REPO/scripts/isambard"
sbatch "$S/smoke.sbatch"        # 1. check the node; wait for it to pass
sbatch "$S/prepare.sbatch"      # 2. shared prerequisites; wait for it to finish
bash   "$S/production.sh"       # 3. every experiment at once (prints each job id)
sbatch "$S/aggregate.sbatch"    # 4. once production has finished: CIFAR and UCI tables and figures
cd "$AGT_REPO/scripts/poisoning_paper" && \
  srun --nodes=1 --gpus=1 --time=00:30:00 python cifar_pca_plots.py   # 5. CIFAR summary figure
```

1. **[`smoke.sbatch`](smoke.sbatch)** (4 GPUs, 30 min). Pass: 4 GH200s listed, torch
   `2.13.0+cu126`, the [`nccl_check.py`](nccl_check.py) all-reduce prints OK, and pytest reports
   `1243 passed, 1 failed, 24 skipped`. The one failure, `test_bounded_sgd[ce-10-1-1.0-0.0-0.0]`, is
   expected: a float32 rounding difference between the Grace-CPU BLAS and x86. Investigate any other
   failure.
2. **[`prepare.sbatch`](prepare.sbatch)** (4 GPUs, up to 6 h). It builds the files that parallel jobs
   would otherwise race to write: the CIFAR and OCT-MNIST PCA bases, the UCI initial model, and the
   CIFAR hyperparameter selection at d = 20, 22 and 24 (a 324-point grid per width, sharded over the
   4 GPUs). Check that `$AGT_ROOT/.results/cifar_pca_selected_*_d{20,22,24}.json` exist.
3. **[`production.sh`](production.sh)** writes the CIFAR run lists (from
   `cifar_pca_manifest.py --missing`, frozen at submit time) and submits:

   | Job | GPUs x `--time` | Runs | Measured |
   |---|---|---|---|
   | [`cifar_cheap.sbatch`](cifar_cheap.sbatch) | 1 x 2 h | every CIFAR run under 2^16 leaves, in sequence | |
   | [`cifar_sharded.sbatch`](cifar_sharded.sbatch) arrays | 4 x 45 min / 3 h / 12 h (d = 20 / 22 / 24) | one CIFAR run of at least 2^16 leaves per task | 15 min / 65 min / 4.5 h |
   | [`octmnist_threat.sbatch`](octmnist_threat.sbatch) | 4 x 3 h | OCT-MNIST threat grid | |
   | [`octmnist_single.sbatch`](octmnist_single.sbatch) | 1 x 8 h | OCT-MNIST width, radius and depth sweeps, then the summary figure | |
   | [`uci_single.sbatch`](uci_single.sbatch) | 1 x 30 min | UCI unrefined and `n_splits`=2 | 3 min |
   | [`uci_sharded.sbatch`](uci_sharded.sbatch) `3` / `4` | 4 x 6 h / 20 h | UCI `n_splits` = 3, 4 | 46 min / 17.6 h |
   | [`halfmoons.sbatch`](halfmoons.sbatch) | 1 x 1 h | half-moons unrefined and 2^6..6^6 (cached; aggregated in section 5) | |
   | [`pt_control_cheap.sbatch`](pt_control_cheap.sbatch) | 1 x 30 min | CIFAR control at d = 22, 24: unrefined, top 12, eps/2 | |
   | [`pt_control_sharded.sbatch`](pt_control_sharded.sbatch) `22` / `24` | 4 x 3 h / 12 h | CIFAR control, all d bisected | 65 min / 4.5 h |

   The OCT-MNIST sweeps train and plot in the same job. `production.sh` is safe to
   re-run. The CIFAR lists then contain only runs that failed or timed out, plus a `--float64` rerun
   for any run with a bound violation above 1e-3; every other job reads its finished runs from the
   cache. The `--time` values came from [`benchmark.sbatch`](benchmark.sbatch) (job 6560456), which
   only needs re-running on different hardware.
4. **[`aggregate.sbatch`](aggregate.sbatch)** (1 GPU, 2 h) runs the CIFAR threat, depth and width
   scripts and `uci_refinement_sweep.py --rungs 2 3 4`. They read cached runs only, and exit listing
   any that are missing.
5. **`cifar_pca_plots.py`** is in no job; it reads the same cache.

Check job outcomes with `sacct -X -S <date> --format=JobID%20,JobName%22,State,Elapsed,Timelimit,ExitCode`.
If NCCL hangs, add `export NCCL_DEBUG=INFO` to the job and rerun `nccl_check.py`.

### 3.4 Outputs and bringing them back

Everything lives under `$AGT_ROOT` ([`env.sh`](env.sh)):
- `.results/`: cached run records (`*.json`, `*.violation`) and parameter boxes;
- `.models/`: pre-trained models;
- `.data/`: datasets and PCA bases;
- `.figures/`: every figure;
- `logs/`: job logs, holding the per-experiment tables;
- `manifests/`: the frozen CIFAR run lists.

```bash
DEST=scripts/poisoning_paper/.isambard     # [local]; gitignored
rsync -av "<user>@<isambard-login>:<AGT_ROOT>/.figures/" "$DEST/figures/"
rsync -av "<user>@<isambard-login>:<AGT_ROOT>/logs/" "$DEST/logs/"
rsync -av --include='*/' --include='*.json' --include='*.violation' --exclude='*' \
    "<user>@<isambard-login>:<AGT_ROOT>/.results/" "$DEST/results/"
```

---

## 4. Open items

- Refactor, not part of this work: `init_distributed`, `rank` and `world_size` have three copies
  (CIFAR, OCT-MNIST, UCI), and CIFAR and UCI import `count_violations` from `octmnist_pca`. Both belong
  in `script_utils`.
- The pre-training-radius control has no aggregation script; its numbers above were read from the job
  logs.

---

## 5. Refinement-ladder campaign

Purpose: the paper figures of nominal, unrefined and refined certified accuracy against the partition
of the eps-ball (`n_splits^n_dims`), marking the rungs that needed 4-GPU sharding. Section 2 left
two gaps. On CIFAR-10 and OCT-MNIST the depth sweeps sit where refinement moves certified accuracy by
only a few test points. On half-moons, certified accuracy was still rising at 6^6. Status: submitted
scripts only; no results yet.

| Dataset | Cell | Rungs (new in **bold**) | Run through |
|---|---|---|---|
| CIFAR-10, d=20 | k=200, eps=0.02 (E4) | **2^4**, **2^8**, 2^12, **2^16**, 2^20, **4^8**, **4^10** | `cifar_pca_run.py`, listed by `cifar_pca_manifest.py` |
| OCT-MNIST, d=15 | eps=0.1, k in {200, 400} | **2^5**, **2^10**, 2^15, **3^10**, **3^12**, **4^10**, **3^15** | [`octmnist_pca_ladder_run.py`](../poisoning_paper/octmnist_pca_ladder_run.py), listed by [`octmnist_pca_ladder.py`](../poisoning_paper/octmnist_pca_ladder.py) |
| Half-moons | k=200, eps=0.01 | 2^6..6^6 (rerun into the cache), **8^6**, **10^6**, **12^6**, **16^6** | [`halfmoons_run.py`](../poisoning_paper/halfmoons_run.py) |

The CIFAR-10 and OCT-MNIST ladders reuse cached runs: the E1 runs, and the threat-grid baseline and
2^15 runs. The OCT-MNIST ladder builds its configurations exactly as the threat grid does (tag,
`leaf_chunk`, and `max_leaves` equal to the leaf count), because both of those fields are in the
cache key. Rungs of at least 2^16 leaves run on 4 GPUs. Every run record stores `world_size`, and the
aggregators print it in a `ranks` column.

```bash
cd "$AGT_ROOT/logs"
bash "$AGT_REPO/scripts/isambard/ladders.sh"                # every ladder job at once (prints each job id)
sbatch "$AGT_REPO/scripts/isambard/ladders_aggregate.sbatch"  # once they have all finished
```

| Job | GPUs x `--time` | Runs | Estimate |
|---|---|---|---|
| [`cifar_cheap.sbatch`](cifar_cheap.sbatch) | 1 x 2 h | CIFAR 2^4, 2^8 | seconds |
| [`cifar_sharded.sbatch`](cifar_sharded.sbatch) array | 4 x 45 min | CIFAR 2^16, 4^8, 4^10 | 62 s / 61 s / 911 s (measured, E2) |
| [`octmnist_ladder_cheap.sbatch`](octmnist_ladder_cheap.sbatch) | 1 x 1 h | OCT 2^5, 2^10, 3^10 at both k | ~15 min |
| [`octmnist_ladder_sharded.sbatch`](octmnist_ladder_sharded.sbatch) array | 4 x 1.5 h | OCT 3^12, 4^10 at both k | 13 / 26 min each |
| (same) array | 4 x 12 h | OCT 3^15 at both k | ~6 h each |
| [`halfmoons.sbatch`](halfmoons.sbatch) | 1 x 1 h | half-moons unrefined, 2^6..6^6 | ~5 min |
| [`halfmoons_sharded.sbatch`](halfmoons_sharded.sbatch) `8 10 12` | 4 x 4 h | half-moons 8^6, 10^6, 12^6 | ~1.2 h |
| [`halfmoons_sharded.sbatch`](halfmoons_sharded.sbatch) `16` | 4 x 12 h | half-moons 16^6 (16.8M leaves) | ~5 h |
| [`ladders_aggregate.sbatch`](ladders_aggregate.sbatch) | 1 x 1 h | tables and `cifar_pca_leaf_sweep_e4.json`, `octmnist_pca_ladder_sweep.json`, `halfmoons_refinement_sweep.json` | |

The estimates are extrapolations and have not been measured on Isambard. The OCT-MNIST figures scale
the threat grid's 2^15 run (49 s on 4 GPUs) by leaf count. The half-moons figures scale 3.9 ms per
leaf, measured on one RTX 5090 in float64, over 4 GPUs. Check `sacct` elapsed times against them.

Aggregate on Isambard, not locally. The pre-trained OCT-MNIST d=15 model retrained on x86 differs
slightly (unrefined k=200 certified accuracy 0.528, against 0.524 on Isambard), so local runs would
miss the cache and give different numbers. The rsync in section 3.4 also brings back the aggregators'
`*.json` results files.
