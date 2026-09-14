# Isambard-AI runbook: sharded poison-certified training

Purpose: the ordered steps, job scripts and checks for running the poisoning-paper refinement
experiments -- CIFAR-10 PCA, OCT-MNIST PCA, UCI house-electric and half-moons -- on Isambard-AI, with
input-ball refinement leaves sharded data-parallel across the GPUs of one node.

Design docs: [../CIFAR_PCA_EXPERIMENT_DESIGN.md](../CIFAR_PCA_EXPERIMENT_DESIGN.md) (experiments E0-E3,
cost model, library sharding L1-L5), [../SHARED_GRID_PARTITION.md](../SHARED_GRID_PARTITION.md)
(soundness of refinement). Both are gitignored (`scripts/*.md`); this file is not.

Key external dependencies: Slurm, uv, torch 2.13.0 (`+cu126` on aarch64, from `uv.lock`), NCCL via
`torchrun --standalone`.

Legend: **[local]** workstation, **[login]** Isambard login node, **[job]** Slurm batch job.

---

## 0. Facts this runbook relies on

| | |
|---|---|
| Node | 4x GH200 (aarch64 Grace + 96 GB H100-class GPU); `--gpus=1` allocates one whole superchip |
| Driver | 565.57 (CUDA 12.7 native), hence torch `+cu126` on aarch64 |
| Slurm | `--nodes=1 --gpus=N`, default partition, no `--account`, no `--exclusive`, max `--time` 24 h |
| GPU cap | 4 GPUs concurrently, enforced by Slurm: jobs beyond the cap pend, nothing is rejected |
| Login nodes | 1 core, 4 GiB per session: fine for `uv sync`, downloads, manifests; not for training or aggregation |
| Storage | `$HOME` 100 GiB (over quota blocks SSH); `$PROJECTDIR` persistent, not backed up; `$SCRATCHDIR` 5 TiB |
| Etiquette | no `squeue`/`sinfo` polling loops (acceptable-use policy); use `sacct` after the fact |

Compute nodes need no internet: the environment is installed and every dataset is staged from a login
node (§2). CIFAR-10 is read with `download=True`, which does not touch the network when the files pass
their integrity check; OCT-MNIST is read with `download=False` from `~/.medmnist`; the UCI data ships
inside the `uci_datasets` package.

### Execution order (checklist)

1. [local] A1-A6: code changes, local checks, push (§1).
2. [login] One-time setup and data staging (§2).
3. [job] `smoke.sbatch` (§4.1).
4. [job] `prepare.sbatch`: PCA bases, UCI initial model, CIFAR E0 selection at d = 20, 22, 24 (§4.2).
5. [job] `benchmark.sbatch` (§4.3), then **decision checkpoint** (§4.4): fill in the `--time` values.
6. [login] `production.sh` (§4.5): all production jobs with dependencies.
7. [login] `reruns.sh` for float64 reruns and failures, until the manifests are empty (§4.6).
8. [job] `aggregate.sbatch` (§4.7), then [local] copy results and figures back (§4.8).

---

## 1. Code changes before Isambard [local]

Nothing below exists yet. Each is a separate commit; tests are part of each.

### A1. Commit and push the current work

- Commit the pyproject/uv migration on its own, then the input-refinement sharding work (library,
  tests, `cifar_pca*.py`, `octmnist_pca*.py`).
- `refine` has diverged from `origin/refine`: `origin/refine` holds `9ed1745 input ball feature
  refinement`, local holds `0f049f3` with the same message plus `4c61edb`. Decide whether to rebase onto
  the remote commit or force-push the local history, then push `refine` to
  `git@github.com:euan-turner/RefinedAGT.git`.

### A2. One storage root: `$AGT_ROOT`

- `script_utils.make_dirs()` (poisoning_paper): when `AGT_ROOT` is set, return and create
  `$AGT_ROOT/{.results,.models,.data,.figures}`; otherwise keep today's directories next to the scripts.
- `cifar_pca.dirs()`: delegate to `script_utils.make_dirs()` and drop `CIFAR_PCA_ROOT`, so CIFAR,
  OCT-MNIST, UCI and half-moons share one root.
- Check: `AGT_ROOT=$(mktemp -d) python -c "import script_utils; print(script_utils.make_dirs())"`.

### A3. Atomic cache writes

A job killed at its wall-time can leave a parameter file without its record. The next launch then
treats the run as cached and crashes reading the record, and `cifar_pca_manifest.py --missing` never
lists it again.

- Add `script_utils.atomic_write(path, write)`. It calls `write(tmp_path)` on a sibling temporary path,
  then `os.replace(tmp_path, path)`.
- The temporary path keeps the original suffix (`x.npz` becomes `x.tmp<pid>.npz`), because `np.savez`
  appends `.npz` to any path that does not end in it.
- Write the record (`.json` / `.violation`) **before** the parameters. The parameter file's existence
  is the cache-hit test, so it must be the last file to appear.
- Sites:
  - `cifar_pca.fit_pca`, `get_pretrained_model` and `train_certified`
  - `cifar_pca_selection.py` (selection JSON)
  - `octmnist_pca.fit_pca`, `get_pretrained_model` and `run_certified`
  - `train_uci.get_model`
  - the UCI run cache (A4)
- Check: kill a cheap CIFAR run between the two writes (or delete its `.json`), relaunch, and confirm
  it retrains instead of crashing.

### A4. UCI: a sharding-aware run entry point

`uci_refinement_sweep.py` trains inline, single-process, with no violation or cost record. Split it
the way CIFAR is split:

- **`uci_refinement.py` (new module).** Holds the constants now in the sweep: `SEED=15`,
  `BATCHSIZE=10000`, `HIDDEN_LAY=1`, `HIDDEN_SIZE=64`, `MAX_ITERS=150`, `EPSILON=0.01`,
  `K_POISON=200`, `STRATEGY`, `MAX_LEAVES`, `LEAF_CHUNK=192`, `INTERVAL_MATMUL="rump"`, and
  `SPLIT_DIMS=11`. It also provides:
  - `init_distributed()`, `rank()` and `world_size()`: the same pattern as `cifar_pca`, and a third
    copy of it. That is a refactor flag for later, not part of this change.
  - `DEVICE = f"cuda:{LOCAL_RANK}"`. Device is excluded from `AGTConfig.hash()`, so this does not
    change cache keys.
  - `make_config(n_splits | None)`, with `shard_leaves=world_size() > 1`.
  - `run(n_splits | None, require_cached=False)`. Rank 0 decides the cache hit and broadcasts it.
    Violations are counted with `octmnist_pca.count_violations` (the existing §9.3 refactor flag).
    Rank 0 writes, atomically, the record `{"violation", "seconds", "world_size", "peak_gib"}`.
  - `certified_metrics(bounded_model)`: the MSE metrics and box width.
- **`uci_run.py` (new).** Usage: `[torchrun --standalone --nproc-per-node 4] uci_run.py [--n-splits N]`.
  Omitting `--n-splits` gives the unrefined baseline. Rank 0 prints one JSON line: the record plus
  the metrics.
- **`uci_refinement_sweep.py`.** Becomes aggregation only: `--rungs 2 3 [4]`, reading with
  `require_cached=True`, then the existing table and figure.
- **Unchanged: `train_uci.get_model`.** It still loads the (small) model onto `cuda:0` on every rank
  before `run` moves it to `DEVICE`. It writes the initial checkpoint unguarded, so `prepare.sbatch`
  creates that checkpoint before any sharded run.
- **Test.** Check `nvidia-smi` first: `LEAF_CHUNK=192` peaks around 27 GB. In a throwaway `AGT_ROOT`,
  run `python uci_run.py --n-splits 2`. Then delete its cache and run
  `torchrun --nproc-per-node 2 uci_run.py --n-splits 2`. The saved parameter boxes must be equal
  exactly, as in test T10.

### A5. Job scripts in `scripts/isambard/`

Add the files listed in §4 verbatim:

- `env.sh`, `nccl_check.py`
- `smoke.sbatch`, `prepare.sbatch`, `benchmark.sbatch`
- `cifar_cheap.sbatch`, `cifar_sharded.sbatch`
- `octmnist_threat.sbatch`, `octmnist_single.sbatch`
- `uci_single.sbatch`, `uci_sharded.sbatch`, `halfmoons.sbatch`
- `aggregate.sbatch`, `production.sh`, `reruns.sh`

Run `bash -n` on each.

### A6. Local sharded check (design doc §7.3 step 1)

If not already done: in a throwaway `AGT_ROOT`, a d=20 `--n-splits 2 --split-dims 16` CIFAR run on
1 GPU and under `torchrun --nproc-per-node 2` produce identical parameter boxes.

---

## 2. One-time setup [login]

```bash
# 1. uv (installs to ~/.local/bin)
curl --location --silent --show-error --fail https://astral.sh/uv/install.sh | sh

# 2. repository on project storage (GitHub needs an SSH key on Isambard, or clone over HTTPS with a token)
mkdir -p "$PROJECTDIR/$USER"
git clone -b refine git@github.com:euan-turner/RefinedAGT.git "$PROJECTDIR/$USER/AbstractGradientTraining"

# 3. environment variables (also add this line to ~/.bashrc if you want it in every login shell)
source "$PROJECTDIR/$USER/AbstractGradientTraining/scripts/isambard/env.sh"

# 4. python + locked environment (aarch64 resolves torch 2.13.0+cu126)
cd "$AGT_REPO"
uv python install 3.12
uv sync --frozen --extra experiments
source scripts/isambard/env.sh           # re-source: activates the venv now that it exists
```

If the login node's 1-core/4-GiB limit kills or stalls step 4, run it on a compute node instead:
`srun --nodes=1 --gpus=1 --time=00:30:00 uv sync --frozen --extra experiments`. The Isambard
distributed-training tutorial installs packages this way.

```bash
# 5. verify the GPU build (expect: 2.13.0+cu126 12.6 True GH200)
srun --nodes=1 --gpus=1 --time=00:05:00 python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 6. stage datasets
mkdir -p "$AGT_ROOT/.data" "$AGT_ROOT/logs" "$AGT_ROOT/manifests"
python -c "import os, torchvision; [torchvision.datasets.CIFAR10(root=os.environ['AGT_ROOT'] + '/.data', train=t, download=True) for t in (True, False)]"
python -c "from medmnist import OCTMNIST; OCTMNIST(split='train', download=True)"   # -> ~/.medmnist/octmnist.npz (all splits)
python -c "import uci_datasets; uci_datasets.Dataset('houseelectric')"              # bundled: confirms it loads
```

**Every later session**: `source "$PROJECTDIR/$USER/AbstractGradientTraining/scripts/isambard/env.sh"`
before `sbatch`. Jobs inherit the variables (`sbatch` exports the environment by default) and
re-source the file to activate the venv. **Updating code**: `git pull && uv sync --frozen --extra
experiments`, never while jobs are running from the checkout.

---

## 3. Caches: what must exist, and who creates it

All paths are under `$AGT_ROOT`. Everything is recomputed on Isambard; nothing is copied from the
workstation.

| Pipeline | Prerequisite (created once, before parallel jobs) | Created by | Run outputs |
|---|---|---|---|
| CIFAR-10 PCA | raw `.data/cifar-10-batches-py/` | §2 step 6 | |
| | PCA basis `.data/cifar_pca_*.npz` | `prepare` | |
| | selection `.results/cifar_pca_selected_*_d{20,22,24}.json` + E0 grid runs | `prepare` | |
| | pre-trained models `.models/cifar_pca_*.ckpt` (36 per d) | `prepare` (E0) | |
| | | | `.results/cifar_{d}_{pretrain}_{hash}` + `.json` |
| OCT-MNIST PCA | raw `~/.medmnist/octmnist.npz` | §2 step 6 | |
| | PCA basis `.data/octmnist_pca_32_*.npz` (shared by all sweeps) | `prepare` | |
| | pre-trained models `.models/octmnist_pca_*.ckpt` | first use, rank-0 guarded; the d values of the threat job (15) and the single-GPU job (4-20, not 15) do not overlap | |
| | | | `.results/octmnist_{tag}_{d}_{pretrain}_{hash}` + `.violation` + `.json`; `octmnist_pca_threat_sweep.json` |
| UCI | data bundled in the venv | `uv sync` | |
| | initial model `.models/uci_15_1_64.ckpt` (same start for every run) | `prepare` | |
| | | | `.results/uci_refinement_15_1_64_rump_{hash}` + `.json` (after A4) |
| Half-moons | none (generated by sklearn) | | none: the sweep recomputes; its log and figure are the result |
| All | | | figures in `.figures/`, job logs in `logs/` |

Why the prerequisites are built first: the PCA bases and the UCI initial model are written by whichever
process gets there first, and two jobs starting together would both write them.

---

## 4. Jobs

Common conventions for every `.sbatch` file:

- **Submission.** Submit from `$AGT_ROOT/logs`: `cd "$AGT_ROOT/logs" && sbatch "$AGT_REPO/scripts/isambard/<file>"`.
  `#SBATCH` lines cannot expand variables, so `--output=%x_%j.out` is relative to the submit directory.
- **Header.** Every file starts with the block below. In the listings, `# (common header)` marks where
  it goes; the job's own `#SBATCH` lines, shown above that marker, go directly after the header's
  `#SBATCH` lines and before `set -euo pipefail`. Slurm ignores directives after the first command.
  A repeated directive, like `cifar_sharded.sbatch`'s `--output`, replaces the header's.

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/isambard/env.sh"
cd "$AGT_REPO/scripts/poisoning_paper"
```

- **Multi-GPU launch.** Single-node, so `torchrun --standalone --nproc-per-node=N`, with no `brics/nccl`
  modules and no `srun --mpi`. The scripts read `LOCAL_RANK` and `WORLD_SIZE`, which torchrun sets.

### 4.0 Support files

#### `env.sh`

```bash
# Environment for the Isambard-AI experiment jobs: repository, cache root, uv locations, venv.
# Source on the login node before sbatch (jobs inherit it) and at the top of every job.
export AGT_REPO="$PROJECTDIR/$USER/AbstractGradientTraining"
export AGT_ROOT="$PROJECTDIR/$USER/agt"
export UV_CACHE_DIR="$SCRATCHDIR/uv-cache"   # keeps multi-GB wheels out of the 100 GiB $HOME
export UV_LINK_MODE=copy                      # cache and venv are on different filesystems
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
if [ -f "$AGT_REPO/.venv/bin/activate" ]; then
    source "$AGT_REPO/.venv/bin/activate"
fi
```

#### `nccl_check.py`

```python
"""
Smoke test for NCCL on one Isambard-AI node: the MIN/MAX all-reduces the sharded refinement uses.

Usage: torchrun --standalone --nproc-per-node 4 nccl_check.py
Key external dependencies: torch.distributed (NCCL).
"""

import os

import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
rank, world_size = dist.get_rank(), dist.get_world_size()
lower = torch.full((1 << 20,), float(rank), device="cuda")
upper = lower.clone()
dist.all_reduce(lower, op=dist.ReduceOp.MIN)
dist.all_reduce(upper, op=dist.ReduceOp.MAX)
assert lower.eq(0).all() and upper.eq(world_size - 1).all(), "all-reduce mismatch"
if rank == 0:
    print(f"NCCL MIN/MAX all-reduce OK on {world_size} ranks")
dist.destroy_process_group()
```

### 4.1 `smoke.sbatch` (4 GPUs, 30 min)

```bash
#SBATCH --job-name=agt-smoke
#SBATCH --gpus=4
#SBATCH --time=00:30:00
# (common header)
nvidia-smi --list-gpus
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
torchrun --standalone --nproc-per-node=4 "$AGT_REPO/scripts/isambard/nccl_check.py"
cd "$AGT_REPO" && python -m pytest -q tests
```

Pass criteria:
- 4 GH200 devices listed.
- torch reports `2.13.0+cu126`.
- The NCCL check prints OK.
- pytest reports the same pass count as locally (1244 passed, 24 skipped).

### 4.2 `prepare.sbatch` (4 GPUs, 6 h)

```bash
#SBATCH --job-name=agt-prepare
#SBATCH --gpus=4
#SBATCH --time=06:00:00
# (common header)
python -c "import cifar_pca; cifar_pca.fit_pca()"
python -c "import octmnist_pca; octmnist_pca.fit_pca()"
python -c "import train_uci, uci_refinement as u; train_uci.get_model(u.HIDDEN_LAY, u.HIDDEN_SIZE, u.SEED)"

# E0: the 324-point selection grid per feature width, the pre-training grid split over the 4 GPUs.
# Shards touch disjoint pre-trained models; the unsharded call reads the grid back and writes the selection.
for d in 20 22 24; do
    pids=()
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$i python cifar_pca_selection.py --d "$d" --shard "$i" 4 \
            > "$SLURM_SUBMIT_DIR/selection_d${d}_shard${i}_${SLURM_JOB_ID}.log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done   # a failed shard fails the job (set -e)
    python cifar_pca_selection.py --d "$d"
done
```

Check: `$AGT_ROOT/.results/cifar_pca_selected_*_d{20,22,24}.json` exist, and each job log names an
admissible selected point. E3 at d = 22 and 24 is meaningless without its own selection (design §3).

### 4.3 `benchmark.sbatch` (4 GPUs, 3 h)

Measures what the time requests depend on. It runs in a throwaway root, because `shard_leaves` is not
part of the cache key and a second run would otherwise just read the first one's result.

```bash
#SBATCH --job-name=agt-benchmark
#SBATCH --gpus=4
#SBATCH --time=03:00:00
# (common header)
BENCH="$AGT_ROOT/benchmark"
mkdir -p "$BENCH/.results" "$BENCH/.models"
ln -sfn "$AGT_ROOT/.data" "$BENCH/.data"
cp "$AGT_ROOT"/.results/cifar_pca_selected_*_d20.json "$BENCH/.results/"
cp "$AGT_ROOT"/.models/cifar_pca_*n_dims=20_*.ckpt "$AGT_ROOT"/.models/uci_*.ckpt "$BENCH/.models/"
export AGT_ROOT="$BENCH"
OUT="$SLURM_SUBMIT_DIR/benchmark_${SLURM_JOB_ID}.jsonl"

CIFAR="--pca-dims 20 --k 100 --eps 0.05 --n-splits 2"
for n in 1 2 4; do                                   # 2^16 leaves: scaling across 1/2/4 GPUs
    rm -f "$BENCH"/.results/cifar_20_*
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n - 1))) torchrun --standalone --nproc-per-node="$n" \
        cifar_pca_run.py $CIFAR --split-dims 16 | tee -a "$OUT"
done
rm -f "$BENCH"/.results/cifar_20_*                   # 2^20 leaves on 4 GPUs: the E1 production size
torchrun --standalone --nproc-per-node=4 cifar_pca_run.py $CIFAR --split-dims 20 | tee -a "$OUT"

for n in 1 4; do                                     # UCI 2^11 leaves on 1 and 4 GPUs
    rm -f "$BENCH"/.results/uci_refinement_*
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n - 1))) torchrun --standalone --nproc-per-node="$n" \
        uci_run.py --n-splits 2 | tee -a "$OUT"
done
```

### 4.4 Decision checkpoint (after the benchmark)

Read `seconds`, `world_size` and `peak_gib` from `benchmark_<jobid>.jsonl`, and fill in:

| Quantity | Formula | Design-doc estimate |
|---|---|---|
| CIFAR 4-GPU scaling | t(2^16, 1 GPU) / t(2^16, 4 GPU) | ~4 |
| E1 bisect-all run, `T20` | measured t(2^20, 4 GPU) | ~15 min |
| E3 d=22 bisect-all | `T20 * 4 * 24/22` | ~1.2 h |
| E3 d=24 bisect-all | `T20 * 16 * 26/22` | ~5 h |
| UCI (3,11) on 4 GPUs | t(2^11, 4 GPU) * 3^11/2^11 (x86.5) | "hours" on one 5090 |
| UCI (4,11) on 4 GPUs | t(2^11, 4 GPU) * 2048 | ~24x (3,11) |

Per-leaf cost scales with the parameter count `H(d+2)+1`, which is where the `24/22` and `26/22`
factors come from. The UCI refinement is Python-loop-bound at batch 10k (see the `LEAF_CHUNK` comment).
The Grace CPU, not the GPU, may set its speed, and the 1-vs-4 GPU UCI pair shows whether sharding helps
it at all.

Decisions:

- **`--time` per class** = 2x the projection, rounded up, and at most 24:00:00. These go into
  `production.sh` (§4.5): `T_D20`, `T_D22`, `T_D24`, `T_UCI3`, `T_UCI4`.
- **CIFAR d=24.** If the projection exceeds ~20 h, stop. Either implement checkpoint/resume at
  iteration boundaries (deferred in design §7.2) or drop the d=24 bisect-all run, and record which.
- **UCI (4,11).** Include only if its projection fits under 24 h. Otherwise the UCI ladder is 2^11 and 3^11.
- **Scaling well below 4x.** Pack several 2^16-2^20 CIFAR runs per 4-GPU job instead of one array task
  each.
- **Before production:** `rm -rf "$AGT_ROOT/benchmark"`.

### 4.5 Production

#### `cifar_cheap.sbatch` (1 GPU): every CIFAR run below 2^16 leaves, in one allocation

```bash
#SBATCH --job-name=agt-cifar-cheap
#SBATCH --gpus=1
#SBATCH --time=04:00:00
# (common header)
MANIFEST=$1
failed=0
while read -r args; do
    echo "run: $args"
    python cifar_pca_run.py $args < /dev/null || { echo "FAILED: $args"; failed=1; }   # /dev/null: python must not eat the manifest
done < "$MANIFEST"
exit "$failed"
```

About 69 runs: the baselines, bisect-12 screens, `eps/2` diagnostics and E2 rungs below 2^16. Each
launch reloads CIFAR and re-projects it, which costs seconds per run.

#### `cifar_sharded.sbatch` (4 GPUs, array): one CIFAR run of at least 2^16 leaves per task

```bash
#SBATCH --job-name=agt-cifar-sharded
#SBATCH --gpus=4
#SBATCH --output=%x_%A_%a.out
# --array and --time are given at submission, per cost class
# (common header)
MANIFEST=$1
args=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
echo "run: $args"
torchrun --standalone --nproc-per-node=4 cifar_pca_run.py $args
```

#### `octmnist_threat.sbatch` (4 GPUs, 3 h)

```bash
#SBATCH --job-name=agt-octmnist-threat
#SBATCH --gpus=4
#SBATCH --time=03:00:00
# (common header)
torchrun --standalone --nproc-per-node=4 octmnist_pca_threat_sweep.py
```

The sweep is 30 runs: 15 cells, each a baseline and a 2^15-leaf run. The local cost records total
1.08 h on 2x 5090, but they cover 68 cached runs, so treat that as an upper bound.

#### `octmnist_single.sbatch` (1 GPU, 8 h): the sweeps that are not sharding-aware, then the summary figure

```bash
#SBATCH --job-name=agt-octmnist-single
#SBATCH --gpus=1
#SBATCH --time=08:00:00
# (common header)
failed=0
for script in octmnist_pca_refinement_sweep.py octmnist_pca_epsilon_sweep.py octmnist_pca_leaf_sweep.py octmnist_pca_plots.py; do
    echo "== $script"
    python "$script" || { echo "FAILED: $script"; failed=1; }
done
exit "$failed"
```

- `octmnist_pca_plots.py` trains its pre-training-radius panel, so it belongs in this job and not on a
  login node.
- The largest local rung (2^16 leaves) took 650 s on a 5090.

#### `uci_single.sbatch` (1 GPU, 4 h)

```bash
#SBATCH --job-name=agt-uci-single
#SBATCH --gpus=1
#SBATCH --time=04:00:00
# (common header)
python uci_run.py
python uci_run.py --n-splits 2
```

#### `uci_sharded.sbatch` (4 GPUs; `--time` at submission)

```bash
#SBATCH --job-name=agt-uci-sharded
#SBATCH --gpus=4
# (common header)
torchrun --standalone --nproc-per-node=4 uci_run.py --n-splits "$1"
```

#### `halfmoons.sbatch` (1 GPU, 2 h)

```bash
#SBATCH --job-name=agt-halfmoons
#SBATCH --gpus=1
#SBATCH --time=02:00:00
# (common header)
python halfmoons_refinement_sweep.py
```

This sweep has no cache: the table in the job log and `.figures/halfmoons_refinement_sweep.pdf` are
the result.

#### `production.sh` [login]

It writes frozen manifests and submits everything with dependencies. The manifests are frozen at
submit time because `--missing` shrinks as runs finish, which would shift array indices mid-array.

- **First wave:** the four 1-GPU jobs run together, which fills the cap.
- **Then:** the 4-GPU jobs run one after another. They are chained with `afterany`, so one failure
  does not block the rest.

```bash
#!/bin/bash
# Submit the production runs. Run on a login node after sourcing env.sh; set the times from §4.4.
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh first}"
T_D20=01:00:00; T_D22=04:00:00; T_D24=16:00:00; T_UCI3=08:00:00; T_UCI4=""   # "" = skip (4,11)

S="$AGT_REPO/scripts/isambard"
M="$AGT_ROOT/manifests/$(date +%Y%m%d-%H%M)"
mkdir -p "$M"
cd "$AGT_REPO/scripts/poisoning_paper"
python cifar_pca_manifest.py --gpus 1 --missing > "$M/cifar_1gpu.txt"
python cifar_pca_manifest.py --gpus 4 --missing > "$M/cifar_4gpu.txt"
grep -Ev -- '--split-dims 2[24]( |$)' "$M/cifar_4gpu.txt" > "$M/cifar_4gpu_d20.txt" || true
grep -E  -- '--split-dims 22( |$)'    "$M/cifar_4gpu.txt" > "$M/cifar_4gpu_d22.txt" || true
grep -E  -- '--split-dims 24( |$)'    "$M/cifar_4gpu.txt" > "$M/cifar_4gpu_d24.txt" || true
wc -l "$M"/*.txt

cd "$AGT_ROOT/logs"
submit() { sbatch --parsable "$@"; }
first=$(submit "$S/cifar_cheap.sbatch" "$M/cifar_1gpu.txt")
first+=":$(submit "$S/octmnist_single.sbatch")"
first+=":$(submit "$S/uci_single.sbatch")"
first+=":$(submit "$S/halfmoons.sbatch")"

prev=$first
chain() {   # chain <time> <sbatch file> [args...]: run after the previous 4-GPU job, whatever its outcome
    prev=$(submit --dependency="afterany:$prev" --time="$1" "${@:2}")
}
chain_array() {   # chain_array <time> <manifest>
    local n
    n=$(wc -l < "$2")
    if [ "$n" -gt 0 ]; then chain "$1" --array="0-$((n - 1))%1" "$S/cifar_sharded.sbatch" "$2"; fi
}
chain_array "$T_D20" "$M/cifar_4gpu_d20.txt"
chain 03:00:00 "$S/octmnist_threat.sbatch"
chain "$T_UCI3" "$S/uci_sharded.sbatch" 3
if [ -n "$T_UCI4" ]; then chain "$T_UCI4" "$S/uci_sharded.sbatch" 4; fi
chain_array "$T_D22" "$M/cifar_4gpu_d22.txt"
chain_array "$T_D24" "$M/cifar_4gpu_d24.txt"
echo "submitted; last job $prev. Manifests in $M"
```

`chain` passes its options to `sbatch` ahead of the script path, because `sbatch` treats every
argument after the script path as a script argument.

A 4-GPU job cannot start while any of your 1-GPU jobs holds a GPU. The chain makes that ordering
explicit instead of leaving it to the scheduler.

### 4.6 Float64 reruns and failures: `reruns.sh` [login]

Run it when the production chain has finished; check with `sacct`, not a polling loop.
`cifar_pca_manifest.py --missing` then lists:
- runs that failed or timed out (they have no cache entry);
- `--float64` reruns for float32 runs whose violation exceeded 1e-3 (the design §3 precision policy).

`reruns.sh` is `production.sh` restricted to the CIFAR manifests: the same manifest generation and
splitting (the `( |$)` patterns already match lines ending in `--float64`), `cifar_cheap.sbatch`
for the 1-GPU list, and the chained `cifar_sharded.sbatch` arrays. Allow about 2x time for float64
arrays. Repeat until both `--missing` manifests are empty.

The OCT-MNIST and UCI jobs are simply resubmitted if they failed: finished cells are cached and
skipped. OCT-MNIST violations are printed as warnings in the sweep log. UCI violations are in the
records written by A4. An unsound cell there is reported as unsound rather than rerun, unless you
decide otherwise.

### 4.7 `aggregate.sbatch` (1 GPU, 2 h)

```bash
#SBATCH --job-name=agt-aggregate
#SBATCH --gpus=1
#SBATCH --time=02:00:00
# (common header)
failed=0
for cmd in "cifar_pca_threat_sweep.py" "cifar_pca_leaf_sweep.py" "cifar_pca_dims_sweep.py" "uci_refinement_sweep.py --rungs 2 3"; do
    echo "== $cmd"
    python $cmd || { echo "FAILED: $cmd"; failed=1; }
done
exit "$failed"
```

- Add `4` to `--rungs` if (4,11) ran.
- The CIFAR aggregation scripts read cached runs only (`collect` exits listing whatever is missing),
  so a failure here names the runs to resubmit via §4.6.
- This runs as a job rather than on a login node because it loads and projects all of CIFAR-10,
  which needs more than 4 GiB.

### 4.8 Bring results back [local]

```bash
DEST=scripts/poisoning_paper/.isambard
mkdir -p "$DEST"
rsync -av "<user>@<isambard-login>:<AGT_ROOT>/.figures/" "$DEST/figures/"
rsync -av "<user>@<isambard-login>:<AGT_ROOT>/logs/" "$DEST/logs/"
rsync -av --include='*/' --include='*.json' --include='*.violation' --exclude='*' \
    "<user>@<isambard-login>:<AGT_ROOT>/.results/" "$DEST/results/"
```

`<AGT_ROOT>` is the expanded path; `echo $AGT_ROOT` on Isambard shows it. Use the login host you
normally SSH to.

---

## 5. Monitoring and recovery

- **Status**, run by hand, not in a loop:
  - `squeue --me`
  - `sacct -X -S today --format=JobID%20,JobName%22,State,Elapsed,Timelimit,ExitCode`
  - Pending jobs show a reason: `Dependency` is the chain; a GRES/QOS limit reason is the 4-GPU cap.
- **Timeouts.** Resubmit through `--missing` (§4.6). Finished runs are cached. After A3, a run killed
  mid-write retrains instead of poisoning its cache entry.
- **NCCL hangs or errors.** Add `export NCCL_DEBUG=INFO` to the job and rerun the smoke test's
  `nccl_check.py` on the same node type.
- **OOM.** Not expected on 96 GB: the largest local peaks were about 27 GB (UCI, `LEAF_CHUNK=192`) and
  11 GiB (CIFAR). `LEAF_CHUNK` is part of the cache key, so changing it invalidates that pipeline's
  finished runs.
- **Code changes mid-campaign.** Any change to a config default, `LEAF_CHUNK`, the split tag or the
  selection changes cache keys, so later runs will not match earlier ones. Do not pull such changes
  until a pipeline's runs are complete.

---

## 6. Budget (estimates until the benchmark replaces them)

GH200 is assumed ~2x a 5090 (design §4).

| Job | GPUs x wall (est.) | GPU-hours (est.) |
|---|---|---|
| smoke + prepare (E0 x3) + benchmark | 4 x ~4 h | ~16 |
| CIFAR cheap | 1 x ~2 h | ~2 |
| CIFAR 2^16-2^20 (28 runs: 25 E1 bisect-all, plus E2 2^16, 4^8, 4^10) | 4 x ~7 h | ~28 |
| CIFAR d=22 bisect-all | 4 x ~1.2 h | ~5 |
| CIFAR d=24 bisect-all | 4 x ~5 h | ~20 |
| OCT-MNIST threat + single | 4 x ~1 h + 1 x ~3 h | ~7 |
| UCI (2^11, 3^11) | benchmark decides | ? |
| Half-moons | 1 x < 1 h | ~1 |
| Float64 reruns, aggregation | benchmark and violations decide | ~5-15 |
| **Total** | **~1-2 days of wall time at the 4-GPU cap, plus queueing** | **~85-95 plus UCI** |

---

## 7. Open items

1. `refine` has diverged from `origin/refine` (A1): rebase or force-push?
2. UCI (4,11): in or out, decided from the benchmark (§4.4).
3. CIFAR d=24: whether it fits in 24 h, decided from the benchmark. If not, checkpoint/resume or drop it.
4. UCI is Python-loop-bound: if the Grace CPU is the bottleneck, sharding buys little (§4.4).
5. Refactor flags, not part of this work:
   - `init_distributed`, `rank` and `world_size` now have three copies (CIFAR, OCT, UCI);
   - `count_violations` is imported from `octmnist_pca` by the other pipelines.

   Both belong in `script_utils`.
6. Scope: this covers the refinement experiments. The original paper scripts (`train_uci.py` plots and
   attacks, `halfmoons.py`, the OCT-MNIST pixel-space sweeps) are not scheduled.
