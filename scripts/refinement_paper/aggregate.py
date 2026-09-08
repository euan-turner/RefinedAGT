"""
Collects the cached runs of one task into a table and a results file, for one or more attacks. Never
trains: if any run is missing, it exits listing the run.py arguments that compute them.

Hypothesis behind the ladders. Partitioning the feature-poisoning ball more finely keeps tightening the
certificate, with diminishing returns per leaf. Each ladder holds one attack and one nominal model
fixed (the nominal model does not depend on the attack or on refinement; this script asserts it) and
compares the unrefined certificate with the certificate at each rung.

Usage: python aggregate.py <task> [--k K ...] [--eps E ...] [--rungs RUNG ...] [--out FILE]
``--k``, ``--eps`` and ``--rungs`` default to the task's ladder, as in run.py; several ``--k`` or
``--eps`` values aggregate a threat-model sweep, one cell per attack.

Output: a table per attack on stdout (leaves, box width and its reduction against the unrefined run,
certified and nominal accuracy, run time, number of GPUs, worst bound violation), and
``.results/<task>.json`` (or ``--out``):

    {"task", "title", "cells": [{"k", "epsilon", "baseline": <record>, "rungs": [<record>, ...]}]}

where each record is a run record of harness.run, the reported one (harness.reported_record).

Key external dependencies: the sibling modules harness and tasks.
"""

import argparse
import itertools
import json
import os
import sys

import harness
import tasks

parser = argparse.ArgumentParser()
parser.add_argument("task", choices=tasks.NAMES)
parser.add_argument("--k", type=int, nargs="+", help="attack sizes; default the task's K_POISON")
parser.add_argument("--eps", type=float, nargs="+", help="attack radii; default the task's EPSILON")
parser.add_argument("--rungs", nargs="+", help="partitions; default the task's LADDER")
parser.add_argument("--out", help="results file; default .results/<task>.json")
args = parser.parse_args()

task = tasks.get(args.task)
rungs = [harness.parse_rung(rung) for rung in (args.rungs or task.LADDER) if rung != harness.UNREFINED]
attacks = list(itertools.product(args.k or [task.K_POISON], args.eps or [task.EPSILON]))

records, missing = {}, []
for k, epsilon in attacks:
    for splits in [None] + rungs:
        try:
            records[k, epsilon, harness.rung_label(splits)] = harness.reported_record(task, k, epsilon, splits)
        except harness.MissingRun as error:
            missing.append(f"  python run.py {error}")
if missing:
    sys.exit(f"{len(missing)} runs not available; compute them with:\n" + "\n".join(missing))

cells = []
for k, epsilon in attacks:
    base = records[k, epsilon, harness.UNREFINED]
    refined = sorted((records[k, epsilon, harness.rung_label(splits)] for splits in rungs), key=lambda r: r["leaves"])
    for record in refined:
        if record["dtype"] == base["dtype"]:
            # the nominal trajectory is refinement-independent: same nominal model at every rung
            assert abs(record["metrics"]["nominal_acc"] - base["metrics"]["nominal_acc"]) < 1e-9, (
                f"refinement moved the nominal model at {record['rung']}"
            )
    cells.append({"k": k, "epsilon": epsilon, "baseline": base, "rungs": refined})

    print(f"\n{task.NAME} | k={k} eps={epsilon:g} | nominal acc {base['metrics']['nominal_acc']:.4f}")
    print(f"  {'rung':>10s} {'leaves':>9s}  {'box width':>10s} {'gain':>7s}  {'cert acc':>8s} {'gain':>7s}  "
          f"{'time':>8s} {'gpus':>4s} {'dtype':>7s} {'viol':>7s}")
    for record in [base] + refined:
        m, b = record["metrics"], base["metrics"]
        print(f"  {record['rung']:>10s} {record['leaves']:>9d}  {m['box_width']:10.4e} "
              f"{1 - m['box_width'] / b['box_width']:7.2%}  {m['cert_acc']:8.4f} {m['cert_acc'] - b['cert_acc']:+7.4f}  "
              f"{record['seconds']:7.0f}s {record['world_size']:>4d} {record['dtype']:>7s} {record['violation']:7.1e}")

out = args.out or os.path.join(harness.directory(".results"), f"{task.NAME}.json")


def write(tmp):
    with open(tmp, "w") as file:
        json.dump({"task": task.NAME, "title": task.TITLE, "cells": cells}, file, indent=1)


harness.atomic_write(out, write)
print(f"\nresults: {out}")
