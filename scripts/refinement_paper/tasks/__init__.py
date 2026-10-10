"""
The tasks of the input-ball refinement experiments. Each task is a module that fixes a dataset, a
model, a training schedule and a default attack, and exposes the same interface to harness.py:

    NAME              the task's name on the command line and in file names
    TITLE             the task's name in figures (matplotlib mathtext)
    DTYPE             the precision runs use unless a float64 rerun is requested
    K_POISON, EPSILON the attack of the task's refinement ladder
    LADDER            the rungs of that ladder (see harness.parse_rung)
    LEAF_CHUNK        leaves stacked into one bounding call (memory only; does not change results)
    SECONDS_PER_LEAF  measured GPU-seconds per leaf of a whole run, for Slurm time limits
    prepare()         download the data and build anything cached (PCA basis, pre-trained model)
    initial_model(dtype)        the torch model every run starts from
    make_config(k, epsilon)     the AGTConfig of the task at an attack, unrefined
    training_data(dtype)        (dl_train, dl_clean): the poisonable and trusted dataloaders (dl_clean may be None)
    evaluate(bounded_model, dtype)   certified test metrics, including cert_acc, nominal_acc and box_width

Key external dependencies: none beyond those of the task modules.
"""

import importlib

NAMES = ["cifar", "octmnist", "halfmoons", "magic"]  # in the order of the paper figure's panels


def get(name):
    """The task module called ``name``."""
    if name not in NAMES:
        raise ValueError(f"unknown task {name!r}; choose from {NAMES}")
    return importlib.import_module(f"tasks.{name}")
