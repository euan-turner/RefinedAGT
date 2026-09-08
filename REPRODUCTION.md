# Reproducing the refinement ladders

How to reproduce the four refinement ladders of the input-ball refinement paper and the figure drawn
from them (`refinement_ladders.pdf`), from a fresh clone. What the experiments are, and every option of
the scripts, is in [scripts/refinement_paper/README.md](scripts/refinement_paper/README.md); the
soundness argument is in [docs/input_refinement_soundness.md](docs/input_refinement_soundness.md).

There are three routes, in increasing cost:

| Route | What it reproduces | Cost |
|---|---|---|
| [A. Figure from the shipped results](#a-figure-from-the-shipped-results) | the published figure | seconds, no GPU |
| [B. Slurm cluster (Isambard-AI)](#b-full-reproduction-on-isambard-ai) | every run, the tables, the figure | 19 jobs, about 45 GPU-hours |
| [C. One workstation](#c-full-reproduction-on-one-workstation) | the same | about 45 GPU-hours in sequence |

## What is run

Each ladder is one task at one attack: the unrefined certificate, then the certificate at each rung.
A rung `N^D` cuts the `D` most sensitive input features into `N` pieces each (`N^D` leaves).

| Task | Attack | Rungs | Runs | Estimated GPU-hours |
|---|---|---|---|---|
| `cifar` | k=200, eps=0.02 | 2^4 2^8 2^12 2^16 2^20 4^8 4^10 | 8 | 2.2 |
| `octmnist` | k=200, eps=0.1 | 2^5 2^10 2^15 3^10 3^12 4^10 3^15 | 8 | 26.7 |
| `halfmoons` | k=200, eps=0.01 | 2^6 3^6 4^6 5^6 6^6 8^6 10^6 12^6 16^6 | 10 | 8.8 |
| `magic` | k=200, eps=0.01 | 2^5 2^10 3^5x2^5 3^10 4^5x3^5 4^10 | 7 | 6.8 |

"Runs" includes the unrefined run. The estimates are leaves times each task's measured
`SECONDS_PER_LEAF` (one GH200; MAGIC was measured on an RTX 5090), and OCT-MNIST 3^15 alone is 24 of
the 45 hours. `python run.py <task> --list` prints the runs of a task with their estimates.

## Requirements

- Linux on x86_64 or aarch64. The lockfile resolves no other platform.
- An NVIDIA GPU for routes B and C. torch 2.13.0 is installed from the CUDA 13.0 index on x86_64 and
  the CUDA 12.6 index on aarch64. The per-task leaf chunk sizes were set for cards with 32 GB.
- [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and the locked environment.
- Network access for the first `uv sync` and for the dataset downloads.

## Clone and install

```bash
curl --location --silent --show-error --fail https://astral.sh/uv/install.sh | sh    # if uv is not installed
git clone --branch artefact https://github.com/euan-turner/RefinedAGT.git AbstractGradientTraining
cd AbstractGradientTraining
uv python install 3.12
uv sync --frozen --extra experiments
source .venv/bin/activate
```

On Isambard-AI, clone into project storage and source the environment file first; see
[route B](#b-full-reproduction-on-isambard-ai).

Check the installation (needs a GPU for a reasonable run time):

```bash
python -m pytest -q tests
```

The suite includes the cheapest half-moons rungs, run end to end and compared with the shipped
reference results.

## Datasets

Nothing is downloaded by hand. `python run.py <task> --prepare` downloads the task's data and builds
what later runs share; every run also does this on first use.

| Task | Source | Downloaded to | Also built by `--prepare` |
|---|---|---|---|
| `cifar` | CIFAR-10, torchvision | `.data/` | PCA basis (`.data/`), pre-trained model (`.models/`) |
| `octmnist` | OCT-MNIST, medmnist | `~/.medmnist/` | PCA basis (`.data/`), pre-trained model (`.models/`) |
| `magic` | MAGIC gamma telescope, OpenML `MagicTelescope` v1 | `.data/` | nothing |
| `halfmoons` | generated (`sklearn.datasets.make_moons`, seeded) | nowhere | nothing |

`.data/`, `.models/`, `.results/` and `.figures/` are under `$AGT_ROOT` when it is set, otherwise under
`scripts/refinement_paper/`. OCT-MNIST is the exception: medmnist always writes to the home directory.

Prepare all four tasks once before launching runs in parallel, so that parallel runs do not race to
write the PCA bases and pre-trained models:

```bash
cd scripts/refinement_paper
for task in cifar octmnist halfmoons magic; do python run.py $task --prepare; done
```

On Isambard-AI this is step 2 of route B and is not run by hand.

## A. Figure from the shipped results

[scripts/refinement_paper/reference/](scripts/refinement_paper/reference) holds the results files the
published figure was drawn from. No training and no GPU:

```bash
cd scripts/refinement_paper
python plot_ladders.py --results reference     # -> .figures/refinement_ladders.{pdf,png}
```

## B. Full reproduction on Isambard-AI

One node is 4 GH200 GPUs and jobs are limited to 24 h. Login nodes are used for installation,
submission, aggregation and plotting, and not for training. The launchers are in
[scripts/refinement_paper/isambard/](scripts/refinement_paper/isambard).

### 1. One-time setup, on a login node

```bash
curl --location --silent --show-error --fail https://astral.sh/uv/install.sh | sh
mkdir -p "$PROJECTDIR/$USER"
git clone --branch artefact https://github.com/euan-turner/RefinedAGT.git "$PROJECTDIR/$USER/AbstractGradientTraining"
source "$PROJECTDIR/$USER/AbstractGradientTraining/scripts/refinement_paper/isambard/env.sh"
cd "$AGT_REPO"
uv python install 3.12
uv sync --frozen --extra experiments
source scripts/refinement_paper/isambard/env.sh      # again: activates the venv now that it exists
```

`env.sh` sets `AGT_REPO` (the checkout, `$PROJECTDIR/$USER/AbstractGradientTraining`) and `AGT_ROOT`
(`$PROJECTDIR/$USER/agt`, where all outputs go). Source it at the start of every later session; jobs
inherit both variables. To use another location, export `AGT_REPO` or `AGT_ROOT` before sourcing it.

### 2. Check the node and prepare the data

```bash
S="$AGT_REPO/scripts/refinement_paper/isambard"
mkdir -p "$AGT_ROOT/logs" && cd "$AGT_ROOT/logs"     # job logs are written to the submit directory
sbatch "$S/smoke.sbatch"       # 4 GPUs, 45 min limit
sbatch "$S/prepare.sbatch"     # 1 GPU, 1 h limit
```

Wait for both before submitting anything else.

- `smoke.sbatch` checks the GPUs, the torch build, the NCCL all-reduce that sharded runs use, and runs
  the test suite. On aarch64 one upstream test, `test_bounded_sgd[ce-10-1-1.0-0.0-0.0]`, has failed
  from a float32 rounding difference between the Grace CPU's BLAS and x86. Investigate any other
  failure.
- `prepare.sbatch` runs `run.py <task> --prepare` for the four tasks: the downloads, PCA bases and
  pre-trained models of [Datasets](#datasets). The downloads happen inside the job, on a compute node.

### 3. Submit the ladders

```bash
for task in cifar octmnist halfmoons magic; do
    bash "$S/submit.sh" $task
done
```

For each task `submit.sh` submits one 1-GPU job that runs every run under 2^16 leaves in sequence, and
one 4-GPU job for each larger run, its leaves sharded over the node:

| Task | 1-GPU job (`agt-<task>`) | 4-GPU jobs (`agt-<task>-4gpu`) |
|---|---|---|
| `cifar` | unrefined 2^4 2^8 2^12 | 2^16, 4^8, 2^20, 4^10 |
| `octmnist` | unrefined 2^5 2^10 2^15 3^10 | 3^12, 4^10, 3^15 |
| `halfmoons` | unrefined 2^6 3^6 4^6 5^6 6^6 | 8^6, 10^6, 12^6, 16^6 |
| `magic` | unrefined 2^5 2^10 3^5x2^5 3^10 | 4^5x3^5, 4^10 |

That is 4 one-GPU jobs and 13 four-GPU jobs, 19 jobs with the smoke and prepare jobs. Each job's time
limit is twice its estimate plus 30 minutes. The longest, OCT-MNIST 3^15, is estimated at 6 h on four
GPUs and submitted with a 12.5 h limit. `submit.sh` prints each job id and writes the run lists it
submitted to `$AGT_ROOT/manifests/`.

Check outcomes with `sacct` rather than polling `squeue`:

```bash
sacct -X -S <date> --format=JobID%20,JobName%22,State,Elapsed,Timelimit,ExitCode
```

Finished runs are cached, so a job that fails or times out keeps the runs it completed. To resubmit
what is left, run `submit.sh <task>` again once that task's jobs have left the queue: it submits only
runs with no record, plus a float64 rerun of any float32 run whose bounds were violated by more than
`1e-3` (see [Precision](scripts/refinement_paper/README.md#precision)). Repeat until it submits
nothing.

### 4. Aggregate and plot, on a login node

```bash
bash "$S/aggregate.sh"
```

This prints each ladder's table, writes `$AGT_ROOT/.results/<task>.json`, and draws
`$AGT_ROOT/.figures/refinement_ladders.{pdf,png}`. If a run is missing it exits with the `run.py`
commands that compute it; go back to step 3.

To bring results back, copy `$AGT_ROOT/.results/*.json` (enough for the figure) or all of
`$AGT_ROOT/.results/` (the record and certified parameter box of every run).

## C. Full reproduction on one workstation

After [Clone and install](#clone-and-install) and the `--prepare` loop of [Datasets](#datasets):

```bash
cd scripts/refinement_paper
for task in cifar octmnist halfmoons magic; do
    python run.py $task             # the unrefined run and the whole ladder; cached, so safe to re-run
    python aggregate.py $task       # table on stdout, .results/$task.json
done
python plot_ladders.py              # -> .figures/refinement_ladders.{pdf,png}
```

Runs are cached, so an interrupted `run.py` resumes at the first unfinished run. With several GPUs,
shard the expensive rungs over them with torchrun, which gives the same result as one GPU:

```bash
python run.py octmnist --rungs unrefined 2^5 2^10 2^15 3^10
torchrun --standalone --nproc-per-node 2 run.py octmnist --rungs 3^12 4^10 3^15
```

If `aggregate.py` reports a run whose bounds were violated, it prints the `run.py ... --dtype float64`
command that reruns it.

## Checking the result

Compare the tables `aggregate.py` prints, or `.results/<task>.json`, with the file of the same name in
[scripts/refinement_paper/reference/](scripts/refinement_paper/reference). Each record's
`metrics` holds `cert_acc`, `nominal_acc` and `box_width`.

- `halfmoons` and `magic` start from a seeded initialisation and reproduce the reference values across
  machines.
- `cifar` and `octmnist` start from a model pre-trained on the machine. Their results moved in the
  third decimal between an x86_64 workstation and the aarch64 cluster, so expect agreement to about
  two decimals, not equality.
- Run times and GPU counts in the records depend on the hardware. The reference CIFAR-10, OCT-MNIST
  and half-moons runs are from Isambard-AI (GH200, 1 or 4 GPUs per run) and MAGIC from a workstation
  with two RTX 5090s.

`python plot_ladders.py --results reference --out <dir>` draws the published figure for a side-by-side
comparison with the reproduced one.
