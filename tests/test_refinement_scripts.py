"""
Tests for the refinement experiment scripts (scripts/refinement_paper): the rung grammar, the cached
run and its record, the choice of which record is reported, the reference results and figures, and
the experiment list of reproduce.sh.

Key external dependencies: pytest, torch; scikit-learn, matplotlib and scienceplots for the tests that
load a real task or draw the figure (skipped when absent).
"""

import json
import logging
import os
import sys
import types

import pytest
import torch

import abstract_gradient_training as agt

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "refinement_paper")
sys.path.insert(0, SCRIPTS)

import harness  # noqa: E402
import tasks  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_root(tmp_path, monkeypatch):
    """Every test writes its runs, models and data under its own temporary root."""
    monkeypatch.setattr(harness, "ROOT", str(tmp_path))


def _toy_task(dtype=torch.float64):
    """A task small enough to train in well under a second, with the interface of tasks/__init__.py."""
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(80, 4, generator=generator, dtype=torch.float64)
    y = (x[:, 0] > 0).long()

    def initial_model(dtype):
        torch.manual_seed(0)
        return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).to(dtype)

    def make_config(k, epsilon):
        return agt.AGTConfig(
            n_epochs=1, learning_rate=0.5, loss="cross_entropy", k_poison=k, epsilon=epsilon, log_level="ERROR"
        )

    def training_data(dtype):
        dataset = torch.utils.data.TensorDataset(x[:60].to(dtype), y[:60])
        return torch.utils.data.DataLoader(dataset, batch_size=30), None

    def evaluate(bounded_model, dtype):
        worst, nominal, _ = agt.test_metrics.test_accuracy(bounded_model, x[60:].to(dtype), y[60:])
        return {"cert_acc": worst, "nominal_acc": nominal, "box_width": harness.box_width(bounded_model)}

    return types.SimpleNamespace(
        NAME="toy", TITLE="toy", DTYPE=dtype, K_POISON=3, EPSILON=0.05, LADDER=["2^2", "3^2x2^2"], LEAF_CHUNK=4,
        SECONDS_PER_LEAF=1e-3, prepare=lambda: None, initial_model=initial_model, make_config=make_config,
        training_data=training_data, evaluate=evaluate,
    )


# ======================================================================================
# Rungs
# ======================================================================================


@pytest.mark.parametrize(
    "text, splits, leaves",
    [("unrefined", None, 1), ("2^5", [2] * 5, 32), ("3^5x2^5", [3] * 5 + [2] * 5, 7776), ("16^6", [16] * 6, 16**6),
     ("4^2x3^1x2^3", [4, 4, 3, 2, 2, 2], 384)],
)
def test_rung_grammar_round_trips(text, splits, leaves):
    assert harness.parse_rung(text) == splits
    assert harness.rung_label(splits) == text
    assert harness.n_leaves(splits) == leaves
    if splits is not None:
        assert agt.InputRefinementConfig(splits=splits, max_leaves=leaves).n_leaves == leaves


@pytest.mark.parametrize("text", ["", "2", "2^", "1^5", "2^0", "2^5x", "a^b", "2^5 3^5", "2**5"])
def test_rung_grammar_rejects_malformed_rungs(text):
    with pytest.raises(ValueError, match="rung"):
        harness.parse_rung(text)


def test_task_ladders_are_valid_and_fit_their_inputs():
    pytest.importorskip("sklearn")
    for name, n_features in (("halfmoons", 6), ("magic", 10)):
        task = tasks.get(name)
        for rung in task.LADDER:
            assert len(harness.parse_rung(rung)) <= n_features


# ======================================================================================
# The cached run
# ======================================================================================


def test_run_writes_a_complete_record_and_the_certified_box():
    task = _toy_task()
    record = harness.run(task, 3, 0.05, [2, 2])
    assert record["task"] == "toy" and record["rung"] == "2^2" and record["splits"] == [2, 2]
    assert record["leaves"] == 4 and record["dtype"] == "float64" and record["world_size"] == 1
    assert record["violation"] == 0.0 and record["seconds"] > 0
    assert {"cert_acc", "nominal_acc", "box_width"} <= set(record["metrics"])
    assert harness.load_record(task, 3, 0.05, [2, 2]) == json.loads(json.dumps(record))

    bounded_model = harness.load_model(task, 3, 0.05, [2, 2])
    assert harness.box_width(bounded_model) == pytest.approx(record["metrics"]["box_width"], rel=1e-12)
    for l, n, u in zip(bounded_model.param_l, bounded_model.param_n, bounded_model.param_u):
        assert (l <= n).all() and (n <= u).all()


def test_run_is_cached_and_never_retrains(monkeypatch):
    task = _toy_task()
    record = harness.run(task, 3, 0.05, None)

    def _boom(*args, **kwargs):  # pragma: no cover - only runs on regression
        raise AssertionError("a cached run was trained again")

    monkeypatch.setattr(agt, "poison_certified_training", _boom)
    assert harness.run(task, 3, 0.05, None) == json.loads(json.dumps(record))


def test_runs_are_keyed_by_attack_rung_and_dtype():
    task = _toy_task()
    paths = {
        harness.run_path(task, k, epsilon, splits, dtype)
        for k in (3, 5) for epsilon in (0.05, 0.1) for splits in (None, [2, 2], [2, 3])
        for dtype in (torch.float32, torch.float64)
    }
    assert len(paths) == 24


def test_refinement_tightens_the_toy_certificate_and_keeps_the_nominal_model():
    task = _toy_task()
    base = harness.run(task, 3, 0.05, None)["metrics"]
    refined = harness.run(task, 3, 0.05, [3, 3, 2, 2])["metrics"]
    assert refined["box_width"] < base["box_width"]
    assert refined["nominal_acc"] == base["nominal_acc"]
    assert refined["cert_acc"] >= base["cert_acc"]


def test_load_model_of_a_missing_run_names_the_command():
    with pytest.raises(harness.MissingRun, match="toy --k 3 --eps 0.05 --rungs 2\\^2"):
        harness.load_model(_toy_task(), 3, 0.05, [2, 2])


def test_count_violations_records_the_worst_violation():
    logger = logging.getLogger("abstract_gradient_training.interval_arithmetic")
    with harness.count_violations() as counter:
        logger.info("Violated bound: %s by %s", "grads", 1e-5)
        logger.error("Violated bound: %s by %s", "grads", 3e-2)
        logger.info("something else")
    assert counter.worst == 3e-2
    with harness.count_violations() as counter:
        pass
    assert counter.worst == 0.0


def test_agt_logs_violations_in_the_format_the_counter_reads():
    """Pins the upstream log format that count_violations depends on."""
    lower = [torch.tensor([0.0, 1.0])]
    upper = [torch.tensor([1.0, 0.5])]
    with harness.count_violations() as counter:
        agt.interval_arithmetic.validate_interval(lower, upper, msg="test")
    assert counter.worst == pytest.approx(0.5)


# ======================================================================================
# Which record is reported
# ======================================================================================


def _write_record(task, splits, dtype, violation):
    path = harness.run_path(task, task.K_POISON, task.EPSILON, splits, dtype)
    with open(f"{path}.json", "w") as file:
        json.dump({"violation": violation, "dtype": harness.dtype_name(dtype)}, file)


def test_reported_record_is_the_default_precision_run_when_its_bounds_held():
    task = _toy_task(torch.float32)
    _write_record(task, None, torch.float32, 1e-6)
    assert harness.reported_record(task, 3, 0.05, None)["dtype"] == "float32"


def test_reported_record_falls_back_to_the_float64_rerun_of_an_unsound_run():
    task = _toy_task(torch.float32)
    _write_record(task, None, torch.float32, 0.1)
    with pytest.raises(harness.MissingRun, match="--dtype float64"):
        harness.reported_record(task, 3, 0.05, None)
    _write_record(task, None, torch.float64, 0.0)
    assert harness.reported_record(task, 3, 0.05, None)["dtype"] == "float64"


def test_reported_record_refuses_an_unsound_float64_run():
    task = _toy_task(torch.float64)
    _write_record(task, None, torch.float64, 0.1)
    with pytest.raises(ValueError, match="violated"):
        harness.reported_record(task, 3, 0.05, None)


def test_reported_record_of_a_missing_run_names_the_command():
    with pytest.raises(harness.MissingRun, match="toy --k 3 --eps 0.05 --rungs unrefined"):
        harness.reported_record(_toy_task(), 3, 0.05, None)


# ======================================================================================
# Real tasks, reference results and the figure
# ======================================================================================


def _reference(name):
    with open(os.path.join(SCRIPTS, "reference", f"{name}.json")) as file:
        return json.load(file)


@pytest.mark.parametrize("name", tasks.NAMES)
def test_reference_results_hold_each_tasks_whole_ladder(name):
    (cell,) = _reference(name)["cells"]
    assert cell["baseline"]["rung"] == "unrefined"
    rungs = [record["rung"] for record in cell["rungs"]]
    assert all(harness.rung_label(harness.parse_rung(rung)) == rung for rung in rungs)
    for record in [cell["baseline"]] + cell["rungs"]:
        assert record["violation"] <= harness.VIOLATION_TOLERANCE
        assert record["leaves"] == harness.n_leaves(record["splits"])
        assert record["metrics"]["cert_acc"] <= record["metrics"]["nominal_acc"]
        # refinement never loosens the certificate
        assert record["metrics"]["box_width"] <= cell["baseline"]["metrics"]["box_width"]
        assert record["metrics"]["cert_acc"] >= cell["baseline"]["metrics"]["cert_acc"]


def test_halfmoons_reproduces_its_reference_results():
    """The cheapest real runs, end to end through the harness, against the shipped reference."""
    pytest.importorskip("sklearn")
    task = tasks.get("halfmoons")
    (cell,) = _reference("halfmoons")["cells"]
    assert (cell["k"], cell["epsilon"]) == (task.K_POISON, task.EPSILON)
    assert [record["rung"] for record in cell["rungs"]] == task.LADDER
    for expected in (cell["baseline"], cell["rungs"][0]):
        record = harness.run(task, task.K_POISON, task.EPSILON, expected["splits"])
        assert record["violation"] == 0.0
        for metric in ("cert_acc", "nominal_acc", "box_width"):
            assert record["metrics"][metric] == pytest.approx(expected["metrics"][metric], rel=1e-9)


@pytest.mark.xfail(
    strict=True, reason="reference/octmnist.json holds the eps=0.1 ladder; the OCT-MNIST ladder attack is now eps=0.02"
)
def test_the_figure_draws_from_the_reference_results():
    pytest.importorskip("matplotlib")
    pytest.importorskip("scienceplots")
    pytest.importorskip("sklearn")
    pytest.importorskip("medmnist")
    import matplotlib

    matplotlib.use("Agg")
    import plot_ladders

    reference = os.path.join(SCRIPTS, "reference")
    panel = plot_ladders.load_panel(reference, "magic")
    assert panel["title"] == "MAGIC ($d=10$), $k=200$, $\\epsilon=0.01$"
    assert [rung["label"] for rung in panel["rungs"]] == [
        "2^{5}", "2^{10}", "3^{5}\\cdot2^{5}", "3^{10}", "4^{5}\\cdot3^{5}", "4^{10}"
    ]
    assert [rung["bisection"] for rung in panel["rungs"]] == [True, True, False, False, False, False]

    figure = plot_ladders.make_figure(reference)
    assert len(figure.axes) == 4
    assert [ax.get_title() for ax in figure.axes] == [
        "CIFAR-10 (PCA, $d=20$), $k=200$, $\\epsilon=0.02$",
        "OCT-MNIST (PCA, $d=15$), $k=200$, $\\epsilon=0.02$",
        "half-moons ($d=6$), $k=200$, $\\epsilon=0.01$",
        "MAGIC ($d=10$), $k=200$, $\\epsilon=0.01$",
    ]


def _gap_results(task, ks, epsilons):
    """A <task>_threat.json results dict whose gap at (k, eps) is k * eps / 100, nominal 0.9."""
    def record(k, epsilon):
        return {"k": k, "epsilon": epsilon, "rung": "unrefined",
                "metrics": {"nominal_acc": 0.9, "cert_acc": 0.9 - k * epsilon / 100}}
    return {"task": task.NAME, "title": task.TITLE,
            "cells": [{"k": k, "epsilon": e, "baseline": record(k, e), "rungs": []} for k in ks for e in epsilons]}


def test_the_threat_grid_figure_draws_every_task_and_outlines_its_ladder_attack(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("scienceplots")
    pytest.importorskip("sklearn")
    pytest.importorskip("medmnist")
    import matplotlib

    matplotlib.use("Agg")
    import plot_threat_grid

    for name in tasks.NAMES:
        task = tasks.get(name)
        ks, epsilons = sorted({5, task.K_POISON}), sorted({0.005, task.EPSILON})
        with open(tmp_path / f"{name}_threat.json", "w") as file:
            json.dump(_gap_results(task, ks, epsilons), file)

    task = tasks.get("octmnist")
    panel = plot_threat_grid.load_panel(tmp_path, "octmnist")
    assert panel["ks"] == [5, task.K_POISON] and panel["epsilons"] == [0.005, task.EPSILON]
    assert panel["ladder"] == (1, 1)
    assert panel["nominal"] == pytest.approx(0.9)
    assert panel["gap"][1, 1] == pytest.approx(task.K_POISON * task.EPSILON / 100)

    figure = plot_threat_grid.make_figure(tmp_path)
    assert len(figure.axes) == len(tasks.NAMES) + 1  # one panel per task and the colour bar


def test_reproduce_lists_every_experiment_once():
    """Every line of reproduce.sh's EXPERIMENTS is accepted by run.py, and holds the runs it should."""
    pytest.importorskip("sklearn")
    pytest.importorskip("medmnist")
    pytest.importorskip("torchvision")
    import subprocess

    listing = subprocess.run(
        ["bash", os.path.join(SCRIPTS, "reproduce.sh"), "list"],
        capture_output=True, text=True, check=True, env={**os.environ, "AGT_ROOT": harness.ROOT},
    ).stdout.splitlines()
    *runs, total = listing
    runs = [line.split("  #")[0] for line in runs]
    assert total.startswith("# total") and len(runs) == len(set(runs))

    grid = [f"{name} --k {k} --eps {e:g} --rungs unrefined" for name in tasks.NAMES
            for k in (5, 10, 25, 50, 100, 200) for e in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)]
    ladders = [f"{name} --k {tasks.get(name).K_POISON} --eps {tasks.get(name).EPSILON:g} --rungs {rung}"
               for name in tasks.NAMES for rung in tasks.get(name).LADDER]
    small_k = [f"halfmoons --k 25 --eps 0.05 --rungs {rung}" for rung in tasks.get("halfmoons").LADDER]
    small_k += [f"magic --k 25 --eps 0.1 --rungs {rung}" for rung in tasks.get("magic").LADDER]
    small_k += ["cifar --k 25 --eps 0.1 --rungs 2^20", "octmnist --k 25 --eps 0.1 --rungs 2^15"]
    assert sorted(runs) == sorted(grid + ladders + small_k)
