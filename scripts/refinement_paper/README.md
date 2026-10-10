# Input-ball refinement experiments

Experiments and figure for input-ball refinement of poison-certified training
(`abstract_gradient_training.input_refinement`; soundness argument and limitations in
[docs/input_refinement_soundness.md](../../docs/input_refinement_soundness.md)).

Each experiment is a **refinement ladder**: one task, one feature-poisoning attack (`k` poisoned
samples per batch, each moved within an l_inf ball of radius `epsilon`), and the certified test
accuracy at a sequence of ever finer partitions of that ball, against the unrefined certificate and
the nominal (unattacked) model. The four ladders make up the paper figure, `refinement_ladders.pdf`.

| Task | Input | Attack of the ladder | Rungs | Precision |
|---|---|---|---|---|
| `cifar` | CIFAR-10, 20 PCA features | k=200, eps=0.02 | 2^4 2^8 2^12 2^16 2^20 4^8 4^10 | float32 |
| `octmnist` | OCT-MNIST, 15 PCA features | k=200, eps=0.1 | 2^5 2^10 2^15 3^10 3^12 4^10 3^15 | float32 |
| `halfmoons` | half-moons, 6 features | k=200, eps=0.01 | 2^6 3^6 4^6 5^6 6^6 8^6 10^6 12^6 16^6 | float64 |
| `magic` | MAGIC gamma telescope, 10 features | k=200, eps=0.01 | 2^5 2^10 3^5x2^5 3^10 4^5x3^5 4^10 | float64 |

A rung `N^D` cuts the `D` most sensitive input features into `N` pieces each, giving `N^D` leaves;
`N^DxM^E` also cuts the next `E` features into `M`. Each task's docstring (`tasks/<task>.py`) describes
its data, model, schedule and what its metrics mean.

## Layout

| File | Role |
|---|---|
| [`tasks/`](tasks) | one module per task, all with the interface in [`tasks/__init__.py`](tasks/__init__.py) |
| [`harness.py`](harness.py) | one cached certified run; the rung grammar; which record is reported |
| [`run.py`](run.py) | launches runs. The only script that trains |
| [`aggregate.py`](aggregate.py) | cached runs of a task into a table and `.results/<task>.json` |
| [`plot_ladders.py`](plot_ladders.py) | the paper figure, from the four results files |
| [`reference/`](reference) | results files behind the published figure |
| [`isambard/`](isambard) | Slurm launchers for Isambard-AI |

Outputs go under `$AGT_ROOT` when it is set, otherwise next to these scripts: `.results/` (runs and
results files), `.models/`, `.data/`, `.figures/`. A run is `.results/runs/<task>/k<k>_eps<eps>_<dtype>/<rung>`:
`.json` is its record (certified metrics, run time, GPUs, peak memory, worst bound violation) and
`.pt` its certified parameter box.

## Setup

```bash
uv sync --frozen --extra experiments     # from the repository root; Linux x86_64 or aarch64
source .venv/bin/activate
cd scripts/refinement_paper
```

## Reproducing the figure

From the shipped results, with no training:

```bash
python plot_ladders.py --results reference     # -> .figures/refinement_ladders.{pdf,png}
```

From scratch on one machine:

```bash
for task in cifar octmnist halfmoons magic; do
    python run.py $task             # the unrefined run and the whole ladder; cached, so safe to re-run
    python aggregate.py $task       # table on stdout, .results/$task.json
done
python plot_ladders.py
```

The cheap rungs take seconds on one GPU. The largest do not: at the measured cost per leaf
(`SECONDS_PER_LEAF` in each task) half-moons 16^6 is about 7 GPU-hours and OCT-MNIST 3^15 about 24.
Choose rungs with `--rungs`, and shard a run's leaves over several GPUs with torchrun, which gives the
same result:

```bash
python run.py halfmoons --rungs unrefined 2^6 3^6 4^6
torchrun --standalone --nproc-per-node 4 run.py halfmoons --rungs 8^6
```

`aggregate.py` and `plot_ladders.py` never train. If a run is missing, `aggregate.py` exits with the
`run.py` commands that compute it.

### Threat-model sweeps

`--k` and `--eps` take several values and run every combination, so the same scripts sweep the attack:

```bash
python run.py cifar --k 10 25 50 100 200 --eps 0.005 0.01 0.02 0.05 0.1 --rungs unrefined 2^12 2^20
python aggregate.py cifar --k 10 25 50 100 200 --eps 0.005 0.01 0.02 0.05 0.1 --rungs 2^12 2^20 --out .results/cifar_threat.json
```

The results file then holds one cell per attack. `plot_ladders.py` draws only the cell at each task's
ladder attack.

### Precision

Bounds are computed without outward rounding
([limitation](../../docs/input_refinement_soundness.md#7-floating-point-the-limitation-and-where-outward-rounding-is-needed)),
so every run records the worst violation of its own bounds that AGT logged. A run above `1e-3` is not
reported: `aggregate.py` asks for its float64 rerun (`run.py ... --dtype float64`) and reports that in
its place.

### Reproducibility across machines

Half-moons and MAGIC start from a seeded initialisation and reproduce across machines. CIFAR-10 and
OCT-MNIST start from a model pre-trained on the machine, and their results moved in the third decimal
between an x86_64 workstation and the aarch64 cluster. The PCA basis and pre-trained model are cached
under `.data/` and `.models/`, so all runs on one machine share one starting point.

## Reference results

[`reference/`](reference) holds the results files the published figure was drawn from. CIFAR-10,
OCT-MNIST and half-moons were run on Isambard-AI (GH200, 1 or 4 GPUs per run) and MAGIC on a
workstation with two RTX 5090s. They were produced by the code before it was restructured into this
directory and converted to the current file format; the restructured code reproduces the half-moons
and MAGIC values, and the test suite checks the cheapest half-moons rungs against them.

## Running on Isambard-AI

One node is 4 GH200 GPUs; `--gpus=1` allocates one. Jobs are limited to 24 h. Login nodes are enough
for `uv sync`, run lists, `aggregate.py` and `plot_ladders.py`, and not for training.

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
S="$AGT_REPO/scripts/refinement_paper/isambard"
mkdir -p "$AGT_ROOT/logs" && cd "$AGT_ROOT/logs"     # job logs are written to the submit directory
sbatch "$S/smoke.sbatch"       # 1. GPUs, torch build, NCCL all-reduce, test suite. Wait for it to pass
sbatch "$S/prepare.sbatch"     # 2. datasets, PCA bases, pre-trained models. Wait for it to finish
for task in cifar octmnist halfmoons magic; do
    bash "$S/submit.sh" $task  # 3. every run of the ladder that is not cached yet
done
bash "$S/aggregate.sh"         # 4. once the jobs have finished: tables, results files, figure
```

1. [`smoke.sbatch`](isambard/smoke.sbatch) (4 GPUs). On aarch64 one upstream test,
   `test_bounded_sgd[ce-10-1-1.0-0.0-0.0]`, has failed from a float32 rounding difference between the
   Grace CPU's BLAS and x86; investigate any other failure.
2. [`prepare.sbatch`](isambard/prepare.sbatch) (1 GPU) builds the files that parallel jobs would
   otherwise race to write.
3. [`submit.sh`](isambard/submit.sh) `<task> [run.py options]` submits one 1-GPU job running every
   uncached run under 2^16 leaves in sequence, and one 4-GPU job per larger run. Time limits are twice
   the estimate from the task's `SECONDS_PER_LEAF` plus 30 minutes, capped at 24 h. It is safe to
   re-run: it submits only runs with no record, plus the float64 rerun of any run whose bounds were
   violated. It takes the same `--k`, `--eps` and `--rungs` as `run.py`, for sweeps.
4. [`aggregate.sh`](isambard/aggregate.sh) runs on a login node.

Check job outcomes with `sacct -X -S <date> --format=JobID%20,JobName%22,State,Elapsed,Timelimit,ExitCode`
rather than polling `squeue`. To bring results back, copy `$AGT_ROOT/.results/*.json` (enough for the
figure) or all of `$AGT_ROOT/.results/` (records and parameter boxes of every run).
