# Input-ball refinement experiments

Experiments and figures for input-ball refinement of poison-certified training
(`abstract_gradient_training.input_refinement`; soundness argument and limitations in
[docs/input_refinement_soundness.md](../../docs/input_refinement_soundness.md)).

The unit of every experiment is a **refinement ladder**: one task, one feature-poisoning attack (`k`
poisoned samples per batch, each moved within an l_inf ball of radius `epsilon`), and the certified
test accuracy at a sequence of ever finer partitions of that ball, against the unrefined certificate
and the nominal (unattacked) model. [`reproduce.sh`](reproduce.sh) lists every experiment and is the
single entrypoint that runs, aggregates and plots them.

A rung `N^D` cuts the `D` most sensitive input features into `N` pieces each, giving `N^D` leaves;
`N^DxM^E` also cuts the next `E` features into `M`. Each task's docstring (`tasks/<task>.py`) describes
its data, model, schedule and what its metrics mean.

## Experiments

| Results file | Task | Attack | Rungs | Est. GPU-h |
|---|---|---|---|---|
| **Large k, small eps** (paper figure, `refinement_ladders.pdf`) | | | | |
| `cifar` | CIFAR-10, 20 PCA features, float32 | k=200, eps=0.02 | 2^4 2^8 2^12 2^16 2^20 4^8 4^10 | 2.2 |
| `octmnist` | OCT-MNIST, 15 PCA features, float32 | k=200, eps=0.02 | 2^5 2^10 2^15 3^10 3^12 4^10 3^15 | 26.7 |
| `halfmoons` | half-moons, 6 features, float64 | k=200, eps=0.01 | 2^6 3^6 4^6 5^6 6^6 8^6 10^6 12^6 16^6 | 8.8 |
| `magic` | MAGIC gamma telescope, 10 features, float64 | k=200, eps=0.01 | 2^5 2^10 3^5x2^5 3^10 4^5x3^5 4^10 | 6.8 |
| **Small k, large eps** (half-moons and MAGIC drawn over their k=200 ladders in `refinement_ladders.pdf`) | | | | |
| `halfmoons_smallk` | half-moons | k=25, eps=0.05 | the half-moons ladder | 8.8 |
| `magic_smallk` | MAGIC | k=25, eps=0.1 | the MAGIC ladder | 6.8 |
| `cifar_smallk` | CIFAR-10 | k=25, eps=0.1 | 2^20 | 1.0 |
| `octmnist_smallk` | OCT-MNIST | k=25, eps=0.1 | 2^15 | 0.1 |
| **Threat-model grid** (`threat_grid.pdf`) | | | | |
| `<task>_threat` | each task | k ∈ {5, 10, 25, 50, 100, 200} × eps ∈ {0.005, 0.01, 0.02, 0.05, 0.1, 0.2} | unrefined only | 0.1 |

Every ladder includes its unrefined run. Estimates are leaves times each task's measured
`SECONDS_PER_LEAF`; `bash reproduce.sh list` prints every run with its estimate (61 GPU-hours in all).

### Choice of attacks

`epsilon` is in each task's own feature units, which differ: raw `x, x^2, x^3` for half-moons,
standardised features for MAGIC, and PCA features scaled to [0, 1] for the two image tasks. The scale
that matters for a feature-space attack is how far a training point is from the other class: the median
l_inf distance from a poisonable training point to its nearest training point of the other class is
0.66 (half-moons), 0.42 (MAGIC), 0.126 (CIFAR-10) and 0.044 (OCT-MNIST).

- **k is shared across tasks** (200 and 25), and **epsilon is shared within a feature space**: 0.01 for
  the tabular tasks and 0.02 for the image tasks at k=200. At k=200 every radius is under half of the
  class distance above, so a poisoned point cannot be moved onto a typical point of the other class.
- **Small k, large eps.** k=25 with a radius at which the unrefined certificate leaves at least as
  large a gap to nominal as the task's k=200 ladder, so refinement has as much to close: half-moons
  eps=0.05 (a gap of 55 points, against 57 at its ladder) and MAGIC eps=0.1 (29 points, against 19).
- **The image tasks at small k need no refinement.** They fine-tune a robustly pre-trained model at a
  low learning rate, with half or more of every batch trusted clean data. At k ≤ 50 their unrefined
  certificate stays within 10 points of nominal over the whole grid, up to eps=0.2 (1.6 and 4.5 times
  the class distance), and the margin it loses grows almost linearly in k but only weakly in eps. One
  rung that bisects every feature, at k=25 and eps=0.1, shows how much of that small gap refinement
  closes, in place of a full ladder.

## Layout

| File | Role |
|---|---|
| [`reproduce.sh`](reproduce.sh) | every experiment, and the single entrypoint: list, train, submit, aggregate and plot them |
| [`tasks/`](tasks) | one module per task, all with the interface in [`tasks/__init__.py`](tasks/__init__.py) |
| [`harness.py`](harness.py) | one cached certified run; the rung grammar; which record is reported |
| [`run.py`](run.py) | launches runs. The only script that trains |
| [`aggregate.py`](aggregate.py) | cached runs of one task into a table and a results file |
| [`plot_ladders.py`](plot_ladders.py) | the paper figure, from the four ladder results files and the half-moons and MAGIC small-k files |
| [`plot_threat_grid.py`](plot_threat_grid.py) | the threat-model grid figure, from the four `<task>_threat.json` files |
| [`reference/`](reference) | the results file of every experiment, and the figures drawn from them (`reference/figures/`) |
| [`isambard/`](isambard) | Slurm launchers for Isambard-AI |

Outputs go under `$AGT_ROOT` when it is set, otherwise next to these scripts: `.results/` (runs and
results files), `.models/`, `.data/`, `.figures/`. A run is `.results/runs/<task>/k<k>_eps<eps>_<dtype>/<rung>`:
`.json` is its record (certified metrics, wall-clock time of certified training in `seconds`, GPUs, peak
memory, worst bound violation) and `.pt` its certified parameter box.

## Setup

```bash
uv sync --frozen --extra experiments     # from the repository root; Linux x86_64 or aarch64
source .venv/bin/activate
cd scripts/refinement_paper
```

## Reproducing the results

The figures from the shipped results, with no training (the copies in `reference/figures/` were drawn
this way):

```bash
python plot_ladders.py --results reference         # -> .figures/refinement_ladders.{pdf,png}
python plot_threat_grid.py --results reference     # -> .figures/threat_grid.{pdf,png}
```

Everything from scratch:

```bash
bash reproduce.sh list        # every run and its estimated GPU-seconds; --missing for what is left
bash reproduce.sh local       # train every run on this machine (or: reproduce.sh isambard, below)
bash reproduce.sh results     # .results/<name>.json per experiment, .figures/{refinement_ladders,threat_grid}.*
```

Every mode skips cached runs, so each is safe to run again and an interrupted `local` resumes at the
first unfinished run. `results` never trains: if a run is missing it lists the `run.py` commands that
compute it. Each experiment is one line of `EXPERIMENTS` in `reproduce.sh`, holding the same `run.py`
options (`--k`, `--eps`, `--rungs`) that the individual scripts take:

```bash
python run.py halfmoons --k 25 --eps 0.05 --rungs unrefined 2^6 3^6      # chosen runs of one experiment
python aggregate.py halfmoons --k 25 --eps 0.05 --out .results/halfmoons_smallk.json
```

`--k` and `--eps` take several values and run every combination, one results-file cell per attack.

On one GPU the cheap rungs take seconds; the largest do not (OCT-MNIST 3^15 is about 24 GPU-hours,
half-moons 16^6 about 7). `local` runs everything on one GPU in sequence. To shard a large run's leaves
over several GPUs instead, which gives the same result, run it with torchrun before `local` reaches it:

```bash
torchrun --standalone --nproc-per-node 4 run.py octmnist --rungs 3^15
```

### Precision

Bounds are computed without outward rounding
([limitation](../../docs/input_refinement_soundness.md#7-floating-point-the-limitation-and-where-outward-rounding-is-needed)),
so every run records the worst violation of its own bounds that AGT logged. A run above `1e-3` is not
reported: `aggregate.py` asks for its float64 rerun (`run.py ... --dtype float64`) and reports that in
its place. `reproduce.sh list --missing` and `isambard` include those reruns.

### Reproducibility across machines

Half-moons and MAGIC start from a seeded initialisation. CIFAR-10 and OCT-MNIST start from a PCA basis
and a model pre-trained on the machine, cached under `.data/` and `.models/` so that all runs on one
machine share one starting point. Between Isambard-AI (aarch64) and an x86_64 workstation, 173 runs of
these experiments gave identical certified and nominal accuracy on all four tasks; box widths differed
by up to 3e-4 relative on the float32 image tasks and 6e-16 on the float64 tabular ones.

## Reference results

[`reference/`](reference) holds the results file of every experiment in `reproduce.sh`, named as
`reproduce.sh results` names them, and in `reference/figures/` the figures drawn from them. The CIFAR-10,
half-moons and MAGIC ladders at k=200 were produced by the code before it was restructured into this directory (CIFAR-10
and half-moons on Isambard-AI, GH200 with 1 or 4 GPUs per run; MAGIC on a workstation with two RTX
5090s) and converted to the current file format; the restructured code reproduces their values, and the
test suite checks the cheapest half-moons rungs against them. The OCT-MNIST ladder and every small-k and
threat-grid file were run on Isambard-AI with the restructured code.

## Running on Isambard-AI

One node is 4 GH200 GPUs; `--gpus=1` allocates one. Jobs are limited to 24 h. Login nodes are enough
for `uv sync`, run lists, aggregation and plotting, and not for training.

One-time setup, on a login node:

```bash
curl --location --silent --show-error --fail https://astral.sh/uv/install.sh | sh
mkdir -p "$PROJECTDIR/$USER"
git clone <repository url> "$PROJECTDIR/$USER/AbstractGradientTraining"
source "$PROJECTDIR/$USER/AbstractGradientTraining/scripts/refinement_paper/isambard/env.sh"
cd "$AGT_REPO"
uv python install 3.12
uv sync --frozen --extra experiments
source scripts/refinement_paper/isambard/env.sh      # again: activates the venv now that it exists
```

In every later session, source [`isambard/env.sh`](isambard/env.sh) first. It sets `AGT_REPO` (the
checkout) and `AGT_ROOT` (`$PROJECTDIR/$USER/agt`, where all outputs go), and jobs inherit both.

```bash
R="$AGT_REPO/scripts/refinement_paper"
mkdir -p "$AGT_ROOT/logs" && (cd "$AGT_ROOT/logs" && sbatch "$R/isambard/smoke.sbatch")   # 1. wait for it to pass
bash "$R/reproduce.sh" isambard    # 2. data preparation, then every uncached run, held until preparation succeeds
bash "$R/reproduce.sh" results     # 3. once the jobs have finished: tables, results files, figures
```

1. [`smoke.sbatch`](isambard/smoke.sbatch) (4 GPUs) checks the GPUs, the torch build, the NCCL
   all-reduce and runs the test suite. On aarch64 one upstream test,
   `test_bounded_sgd[ce-10-1-1.0-0.0-0.0]`, has failed from a float32 rounding difference between the
   Grace CPU's BLAS and x86; investigate any other failure. It is judged by hand, so nothing waits on it.
2. `reproduce.sh isambard` submits [`prepare.sbatch`](isambard/prepare.sbatch) (1 GPU), which builds
   the files that parallel jobs would otherwise race to write, and then, for each experiment,
   [`submit.sh`](isambard/submit.sh) `<task> [run.py options]` with every job held until preparation
   succeeds. `submit.sh` submits one 1-GPU job running the experiment's uncached runs under 2^16 leaves
   in sequence, and one 4-GPU job per larger run. Time limits are twice the estimate from the task's
   `SECONDS_PER_LEAF` plus 30 minutes, capped at 24 h. Run it again once the jobs have left the queue
   to resubmit what failed or timed out and the float64 rerun of any run whose bounds were violated.
3. `reproduce.sh results` runs on a login node.

Check job outcomes with `sacct -X -S <date> --format=JobID%20,JobName%22,State,Elapsed,Timelimit,ExitCode`
rather than polling `squeue`. To bring results back, copy `$AGT_ROOT/.results/*.json` (enough for the
figures) or all of `$AGT_ROOT/.results/` (records and parameter boxes of every run).
