"""
Shared runner for the input-ball refinement experiments: one certified-training run of a task at one
attack and one partition of the eps-ball, cached on disk under a readable key.

A run is identified by a task (``tasks/``), an attack ``(k, epsilon)``, a rung (the partition, written
``unrefined``, ``N^D`` or ``N^DxM^E...``) and a dtype. Its record (``<run>.json``) holds the certified
metrics and the cost of the run; its certified parameter box (``<run>.pt``) is saved beside it. run.py
launches runs through this module, aggregate.py and plot_ladders.py read the records back, and no
script other than run.py ever trains.

Outputs live under ``$AGT_ROOT`` when it is set (e.g. project storage on a cluster), else next to
this file: ``.results/`` (runs and aggregated results), ``.models/`` (pre-trained models), ``.data/``
(datasets and PCA bases) and ``.figures/``.

Under torchrun the leaves of a refined run are sharded across the GPUs with the same result; rank 0
decides whether the cache is hit, evaluates the model and writes the record.

Design doc: docs/input_refinement_soundness.md (what the certificate means and its floating-point
limitation, which ``count_violations`` monitors).

Key external dependencies: torch, torch.distributed (NCCL, for sharded refinement),
abstract_gradient_training.
"""

import contextlib
import itertools
import json
import logging
import math
import os
import time

import torch
import torch.distributed as dist

import abstract_gradient_training as agt
from abstract_gradient_training.bounded_models import IntervalBoundedModel

ROOT = os.environ.get("AGT_ROOT", os.path.dirname(os.path.abspath(__file__)))

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"

DTYPES = {"float32": torch.float32, "float64": torch.float64}
VIOLATION_TOLERANCE = 1e-3  # a run whose bounds were violated by more than this is not reported
SHARDED_FROM_LEAVES = 2**16  # runs with at least this many leaves are meant for a multi-GPU node


class MissingRun(FileNotFoundError):
    """A run that a reader needs has not been computed. The message is the run.py arguments that compute it."""


def directory(name):
    """``ROOT/name``, created if absent. ``name`` is one of .results, .models, .data, .figures."""
    path = os.path.join(ROOT, name)
    os.makedirs(path, exist_ok=True)
    return path


def atomic_write(path, write):
    """Write ``path`` so that no reader ever observes a partial file: ``write(tmp_path)`` writes a
    sibling temporary file, which is then renamed onto ``path``. ``tmp_path`` keeps ``path``'s suffix."""
    root, ext = os.path.splitext(path)
    tmp_path = f"{root}.tmp{os.getpid()}{ext}"
    write(tmp_path)
    os.replace(tmp_path, path)


# ======================================================================================
# Distributed runs
# ======================================================================================


def init_distributed():
    """Join the torchrun process group when launched with more than one process."""
    if int(os.environ.get("WORLD_SIZE", 1)) > 1 and not dist.is_initialized():
        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group("nccl", device_id=torch.device(DEVICE))


def finalize_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def barrier():
    """Wait until every rank reaches this point (a no-op in a single process)."""
    if dist.is_initialized():
        dist.barrier()


# ======================================================================================
# Rungs: partitions of the eps-ball
# ======================================================================================

UNREFINED = "unrefined"


def parse_rung(text):
    """
    ``'unrefined'`` -> None; ``'N^D'`` or ``'N^DxM^E...'`` -> the ``InputRefinementConfig.splits``
    list: the D most sensitive features cut into N pieces each, the next E into M, and so on.
    """
    if text == UNREFINED:
        return None
    try:
        tiers = [tuple(int(v) for v in tier.split("^")) for tier in text.split("x")]
        if any(len(tier) != 2 or tier[0] < 2 or tier[1] < 1 for tier in tiers):
            raise ValueError
    except ValueError:
        raise ValueError(f"rung {text!r} is not 'unrefined', N^D or N^DxM^E... with N >= 2 and D >= 1") from None
    return [n for n, d in tiers for _ in range(d)]


def rung_tiers(splits):
    """``splits`` as ``[(N, D), ...]`` tiers: runs of equal cut counts, in order."""
    return [(n, len(list(group))) for n, group in itertools.groupby(splits or [])]


def rung_label(splits):
    """The inverse of ``parse_rung``."""
    return UNREFINED if splits is None else "x".join(f"{n}^{d}" for n, d in rung_tiers(splits))


def n_leaves(splits):
    return 1 if splits is None else math.prod(splits)


# ======================================================================================
# One certified run
# ======================================================================================


class _ViolationCounter(logging.Handler):
    """Records the interval-validation violations AGT logs."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.worst = 0.0

    def emit(self, record):
        if "Violated bound" in record.getMessage():
            self.worst = max(self.worst, float(record.args[1]))


@contextlib.contextmanager
def count_violations():
    """
    Capture the interval-validation violations AGT logs during the enclosed run.

    A violation means floating-point rounding has put a nominal value outside its own bounds, so the
    certificate is not sound (docs/input_refinement_soundness.md, "Floating point"). AGT only logs
    these, so a run that does not watch for them can report numbers from bounds that silently
    failed. Yields the counter; read ``.worst`` after the block (0.0 when no bound was violated).
    """
    counter = _ViolationCounter()
    logger = logging.getLogger("abstract_gradient_training")
    logger.addHandler(counter)
    try:
        yield counter
    finally:
        logger.removeHandler(counter)


def dtype_name(dtype):
    return str(dtype).split(".")[-1]


def run_path(task, k, epsilon, splits, dtype):
    """Path of a run, without extension: ``.json`` is its record and ``.pt`` its parameter box."""
    cell = os.path.join(directory(".results"), "runs", task.NAME, f"k{k}_eps{epsilon:g}_{dtype_name(dtype)}")
    os.makedirs(cell, exist_ok=True)
    return os.path.join(cell, rung_label(splits))


def run_args(task, k, epsilon, splits, dtype):
    """The run.py arguments that compute a run."""
    args = f"{task.NAME} --k {k} --eps {epsilon:g} --rungs {rung_label(splits)}"
    return args if dtype == task.DTYPE else f"{args} --dtype {dtype_name(dtype)}"


def make_config(task, k, epsilon, splits):
    """The task's training configuration at an attack and a partition, on this process's device.
    Leaves are sharded across ranks whenever the run is launched with more than one process."""
    config = task.make_config(k, epsilon)
    config.device = DEVICE
    if splits is not None:
        config.input_refinement = agt.InputRefinementConfig(
            splits=splits,
            strategy="sensitivity",
            max_leaves=n_leaves(splits),
            leaf_chunk=task.LEAF_CHUNK,
            shard_leaves=world_size() > 1,
        )
    return config


def load_record(task, k, epsilon, splits, dtype=None):
    """The record of a finished run, or None if it has not been computed."""
    path = f"{run_path(task, k, epsilon, splits, task.DTYPE if dtype is None else dtype)}.json"
    if not os.path.isfile(path):
        return None
    with open(path) as file:
        return json.load(file)


def load_model(task, k, epsilon, splits, dtype=None):
    """The certified model of a finished run: nominal parameters and their certified box."""
    dtype = task.DTYPE if dtype is None else dtype
    if load_record(task, k, epsilon, splits, dtype) is None:
        raise MissingRun(run_args(task, k, epsilon, splits, dtype))
    bounded_model = IntervalBoundedModel(task.initial_model(dtype))
    bounded_model.load_params(f"{run_path(task, k, epsilon, splits, dtype)}.pt")
    return bounded_model


def run(task, k, epsilon, splits, dtype=None):
    """
    Certified training of ``task`` against up to ``k`` poisoned samples per batch within an l_inf
    ball of radius ``epsilon``, with the ball partitioned by ``splits`` (None for the unrefined
    bound). Returns at once if the run is cached.

    Returns:
        dict: the run record -- the run's identity (task, k, epsilon, rung, splits, leaves, dtype),
            its cost (seconds, world_size, peak_gib), the worst bound violation, and ``metrics``:
            the task's certified test metrics, always including ``cert_acc``, ``nominal_acc`` and
            ``box_width``. Only rank 0's record carries the metrics.
    """
    dtype = task.DTYPE if dtype is None else dtype
    path = run_path(task, k, epsilon, splits, dtype)

    cached = torch.tensor(int(os.path.isfile(f"{path}.json")))
    if dist.is_initialized():
        cached = cached.to(DEVICE)
        dist.broadcast(cached, src=0)
    if cached.item():
        return load_record(task, k, epsilon, splits, dtype) if rank() == 0 else {}

    config = make_config(task, k, epsilon, splits)
    bounded_model = IntervalBoundedModel(task.initial_model(dtype))
    dl_train, dl_clean = task.training_data(dtype)
    if torch.cuda.is_available():
        torch.cuda.init()  # the model is still on the CPU here; AGT moves it to config.device
        torch.cuda.reset_peak_memory_stats(DEVICE)
    start = time.time()
    with count_violations() as counter:
        agt.poison_certified_training(bounded_model, config, dl_train, dl_clean=dl_clean)
    record = {
        "task": task.NAME,
        "k": k,
        "epsilon": epsilon,
        "rung": rung_label(splits),
        "splits": splits,
        "leaves": n_leaves(splits),
        "dtype": dtype_name(dtype),
        "violation": counter.worst,
        "seconds": time.time() - start,
        "world_size": world_size(),
        "peak_gib": torch.cuda.max_memory_allocated(DEVICE) / 2**30 if torch.cuda.is_available() else None,
    }
    if rank() == 0:
        record["metrics"] = task.evaluate(bounded_model, dtype)

        def write_record(tmp):
            with open(tmp, "w") as file:
                json.dump(record, file, indent=1)

        # the record marks the run as finished, so it is written last
        atomic_write(f"{path}.pt", bounded_model.save_params)
        atomic_write(f"{path}.json", write_record)
    return record


def reported_record(task, k, epsilon, splits):
    """
    The record to report for a run: the run at the task's dtype when its bounds held, otherwise its
    float64 rerun. Never trains: raises ``MissingRun`` naming the run (or rerun) that is needed.
    """
    record = load_record(task, k, epsilon, splits)
    if record is None:
        raise MissingRun(run_args(task, k, epsilon, splits, task.DTYPE))
    if record["violation"] <= VIOLATION_TOLERANCE:
        return record
    if task.DTYPE != torch.float64:
        record = load_record(task, k, epsilon, splits, torch.float64)
        if record is None:
            raise MissingRun(run_args(task, k, epsilon, splits, torch.float64))
    if record["violation"] > VIOLATION_TOLERANCE:
        raise ValueError(
            f"float64 run violated its bounds by {record['violation']:.1e}: "
            f"{run_args(task, k, epsilon, splits, torch.float64)}"
        )
    return record


def box_width(bounded_model):
    """The sum of the per-parameter interval widths of the certified box. Smaller means tighter."""
    return sum((u - l).sum().item() for l, u in zip(bounded_model.param_l, bounded_model.param_u))
