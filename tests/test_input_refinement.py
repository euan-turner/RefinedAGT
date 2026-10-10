"""
Tests for input-ball refinement of poison-certified training (abstract_gradient_training.input_refinement).

Soundness tests check that the refined bounds contain what they claim to: the leaves cover the ball
exactly, sampled per-sample gradients lie inside the refined gradient bounds, and a model trained on
actually poisoned data lies inside the certified parameter box. Functionality tests check that the
refinement is a drop-in, never-wider replacement, that chunking and sharding do not change the result,
and that invalid configurations are rejected.

Design doc: docs/input_refinement_soundness.md.

Key external dependencies: pytest, torch (torch.multiprocessing and the gloo backend for the sharding tests).
"""

from __future__ import annotations

import copy
import itertools
import math

import pytest
import torch

import abstract_gradient_training as agt
from abstract_gradient_training import input_refinement, test_metrics, training_utils
from abstract_gradient_training.bounded_losses import BoundedBCEWithLogitsLoss, BoundedCrossEntropyLoss, BoundedMSELoss
from abstract_gradient_training.bounded_models import IntervalBoundedModel
from abstract_gradient_training.gradient_accumulation import PoisoningGradientAccumulator

ATOL = 1e-9


def _model(in_dim: int, hidden: int, out_dim: int, seed: int = 0, depth: int = 1) -> torch.nn.Sequential:
    torch.manual_seed(seed)
    layers: list[torch.nn.Module] = [torch.nn.Linear(in_dim, hidden), torch.nn.ReLU()]
    for _ in range(depth - 1):
        layers += [torch.nn.Linear(hidden, hidden), torch.nn.ReLU()]
    layers.append(torch.nn.Linear(hidden, out_dim))
    return torch.nn.Sequential(*layers).double()


def _bounded(model: torch.nn.Sequential, param_radius: float = 0.0) -> IntervalBoundedModel:
    bounded_model = IntervalBoundedModel(model)
    if param_radius > 0:
        bounded_model._param_l = [[p - param_radius for p in module] for module in bounded_model._param_n]
        bounded_model._param_u = [[p + param_radius for p in module] for module in bounded_model._param_n]
    return bounded_model


def _loss_and_labels(loss_name: str, batchsize: int, generator: torch.Generator):
    """A bounded loss with reduction 'none', its output width, and labels of the shape it expects."""
    if loss_name == "cross_entropy":
        return BoundedCrossEntropyLoss(reduction="none"), 3, torch.randint(0, 3, (batchsize,), generator=generator)
    if loss_name == "binary_cross_entropy":
        labels = torch.randint(0, 2, (batchsize, 1), generator=generator)
        return BoundedBCEWithLogitsLoss(reduction="none"), 1, labels
    labels = torch.rand(batchsize, 1, generator=generator, dtype=torch.float64)
    return BoundedMSELoss(reduction="none"), 1, labels


def _refined(bounded_model, x, y, loss, eps, dims, splits, leaf_chunk=4, **kwargs):
    return input_refinement.refined_bound_backward_combined(
        bounded_model, x, y, loss, epsilon=eps, dims=dims, splits=splits, leaf_chunk=leaf_chunk, **kwargs
    )


# ======================================================================================
# Partition: the leaves tile the eps-ball exactly
# ======================================================================================

GRIDS = [
    ([0, 1, 2], [2, 2, 2]),
    ([4, 1], [3, 3]),
    ([5], [4]),
    ([0, 1, 2], [3, 2, 2]),
    ([4, 1], [2, 5]),
    ([0, 1, 2], [3, 1, 2]),
    ([3, 0, 5, 2], [16, 10, 7, 6]),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(64, 6), (9, 2, 3)])
@pytest.mark.parametrize("eps", [0.01, 0.02, 0.1, 0.3])
@pytest.mark.parametrize("dims, splits", GRIDS)
def test_leaves_tile_the_ball_exactly(dtype, shape, eps, dims, splits):
    """Soundness: the union of the leaves is the ball the unrefined bound uses, with no gap between
    neighbouring leaves and none at the faces of the ball, bit for bit."""
    torch.manual_seed(0)
    batch = torch.randn(shape, dtype=dtype)
    batchsize = shape[0]
    n_leaves = math.prod(splits)
    x_l, x_u = input_refinement.leaf_boxes(batch, eps, dims, splits, torch.arange(n_leaves))
    assert x_l.shape == (n_leaves * batchsize, *shape[1:])
    x_l = x_l.view(n_leaves, batchsize, -1)
    x_u = x_u.view(n_leaves, batchsize, -1)
    ball_l = (batch - eps).view(batchsize, -1)
    ball_u = (batch + eps).view(batchsize, -1)

    assert (x_l <= x_u).all()
    assert torch.equal(x_l.amin(dim=0), ball_l)
    assert torch.equal(x_u.amax(dim=0), ball_u)

    uncut = [d for d in range(ball_l.size(1)) if d not in dims]
    assert torch.equal(x_l[:, :, uncut], ball_l[:, uncut].expand(n_leaves, -1, -1))
    assert torch.equal(x_u[:, :, uncut], ball_u[:, uncut].expand(n_leaves, -1, -1))

    # leaf id -> offsets: the digits of the id in the mixed radix `splits`, most significant first
    digits = torch.tensor(list(itertools.product(*(range(s) for s in splits))))
    for position, (d, s) in enumerate(zip(dims, splits)):
        for offset in range(s):
            cells_l = x_l[digits[:, position] == offset][:, :, d]
            cells_u = x_u[digits[:, position] == offset][:, :, d]
            # every leaf with this offset has the same interval along d
            assert torch.equal(cells_l, cells_l[:1].expand_as(cells_l))
            assert torch.equal(cells_u, cells_u[:1].expand_as(cells_u))
            if offset == 0:
                assert torch.equal(cells_l[0], ball_l[:, d])
            else:
                assert torch.equal(cells_l[0], previous_upper)  # noqa: F821 - set on the previous offset
            previous_upper = cells_u[0]
        assert torch.equal(previous_upper, ball_u[:, d])


def test_leaf_cells_are_equal_width():
    batch = torch.zeros(1, 3, dtype=torch.float64)
    x_l, x_u = input_refinement.leaf_boxes(batch, 0.3, [0, 2], [3, 4], torch.arange(12))
    widths = (x_u - x_l).view(12, 3)
    assert torch.allclose(widths[:, 0], torch.full((12,), 0.2, dtype=torch.float64))
    assert torch.allclose(widths[:, 1], torch.full((12,), 0.6, dtype=torch.float64))
    assert torch.allclose(widths[:, 2], torch.full((12,), 0.15, dtype=torch.float64))


@pytest.mark.parametrize("dims, splits", GRIDS)
def test_leaf_subsets_match_the_full_partition(dims, splits):
    """Any subset of leaf ids, in any order, gives the same corners as the full enumeration."""
    torch.manual_seed(0)
    batch = torch.randn(5, 6, dtype=torch.float32)
    n_leaves = math.prod(splits)
    full_l, full_u = input_refinement.leaf_boxes(batch, 0.1, dims, splits, torch.arange(n_leaves))
    leaf_ids = torch.randperm(n_leaves)[: max(1, n_leaves // 2)]
    sub_l, sub_u = input_refinement.leaf_boxes(batch, 0.1, dims, splits, leaf_ids)
    assert torch.equal(sub_l.view(len(leaf_ids), 5, 6), full_l.view(n_leaves, 5, 6)[leaf_ids])
    assert torch.equal(sub_u.view(len(leaf_ids), 5, 6), full_u.view(n_leaves, 5, 6)[leaf_ids])


@pytest.mark.parametrize(
    "dims, splits, match",
    [([0, 1], [2], "split counts"), ([0, 0], [2, 2], "distinct"), ([0, 6], [2, 2], "inside"), ([0], [0], "at least 1")],
)
def test_leaf_boxes_reject_an_invalid_grid(dims, splits, match):
    with pytest.raises(ValueError, match=match):
        input_refinement.leaf_boxes(torch.zeros(2, 6), 0.1, dims, splits, torch.arange(1))


@pytest.mark.parametrize("n_leaves", [1, 4, 27, 1024])
@pytest.mark.parametrize("world_size", [1, 2, 3, 5, 8])
def test_leaf_blocks_partition_the_leaves(n_leaves, world_size):
    blocks = [input_refinement.leaf_block(n_leaves, rank, world_size) for rank in range(world_size)]
    assert [i for start, stop in blocks for i in range(start, stop)] == list(range(n_leaves))
    sizes = [stop - start for start, stop in blocks]
    assert max(sizes) - min(sizes) <= 1


# ======================================================================================
# Gradient bounds: drop-in, never wider, monotone, chunk-invariant
# ======================================================================================


@pytest.mark.parametrize("dims, splits", [([], []), ([0, 1], [1, 1]), ([2], [1])])
def test_nothing_cut_is_the_unrefined_bound(dims, splits):
    torch.manual_seed(0)
    bounded_model = _bounded(_model(4, 8, 3), param_radius=0.01)
    loss = BoundedCrossEntropyLoss(reduction="none")
    x = torch.randn(6, 4, dtype=torch.float64)
    y = torch.randint(0, 3, (6,))
    eps = 0.1

    base_l, base_u = bounded_model.bound_backward_combined(x - eps, x + eps, y, loss)
    ref_l, ref_u = _refined(bounded_model, x, y, loss, eps, dims, splits, leaf_chunk=1)
    for a, b in zip(ref_l + ref_u, base_l + base_u):
        assert torch.equal(a, b)


def test_disabled_refinement_never_enters_the_refined_path(monkeypatch):
    def _boom(*args, **kwargs):  # pragma: no cover - only runs on regression
        raise AssertionError("refined_bound_backward_combined called with input_refinement disabled")

    monkeypatch.setattr(input_refinement, "refined_bound_backward_combined", _boom)
    bounded_model = _bounded(_model(5, 8, 2))
    config = agt.AGTConfig(n_epochs=1, learning_rate=0.01, loss="cross_entropy", k_poison=3, epsilon=0.05)
    x = torch.randn(8, 5, dtype=torch.float64)
    y = torch.randint(0, 2, (8,))
    training_utils.compute_batch_gradients(bounded_model, x, y, config, nominal=False, poisoned=True)


def test_enabled_refinement_requires_split_dims():
    bounded_model = _bounded(_model(5, 8, 2))
    config = agt.AGTConfig(
        n_epochs=1, learning_rate=0.01, loss="cross_entropy", k_poison=3, epsilon=0.05,
        input_refinement=agt.InputRefinementConfig(splits=[2, 2]),
    )
    x = torch.randn(8, 5, dtype=torch.float64)
    y = torch.randint(0, 2, (8,))
    with pytest.raises(ValueError, match="split_dims"):
        training_utils.compute_batch_gradients(bounded_model, x, y, config, nominal=False, poisoned=True)


@pytest.mark.parametrize("loss_name", ["cross_entropy", "binary_cross_entropy", "mse"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_refinement_is_never_wider(loss_name, seed):
    generator = torch.Generator().manual_seed(seed)
    loss, out_dim, y = _loss_and_labels(loss_name, 7, generator)
    bounded_model = _bounded(_model(5, 12, out_dim, seed=seed, depth=2), param_radius=0.02)
    x = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    eps = 0.08

    base_l, base_u = bounded_model.bound_backward_combined(x - eps, x + eps, y, loss)
    ref_l, ref_u = _refined(bounded_model, x, y, loss, eps, [0, 1, 2], [2, 3, 2])
    for a, b in zip(ref_l, base_l):
        assert (a >= b - ATOL).all(), (a - b).min()
    for a, b in zip(ref_u, base_u):
        assert (a <= b + ATOL).all(), (a - b).max()
    # and strictly tighter somewhere, or the refinement is doing nothing
    assert any((a > b + 1e-6).any() for a, b in zip(ref_l, base_l))


@pytest.mark.parametrize("coarse, fine", [([2, 2, 2, 2], [4, 4, 2, 2]), ([2, 3], [4, 6]), ([3, 1], [3, 2])])
def test_a_nested_finer_grid_is_never_wider(coarse, fine):
    """Cutting every cell of a grid again refines it, so the hull can only shrink."""
    torch.manual_seed(0)
    bounded_model = _bounded(_model(4, 10, 3, depth=2), param_radius=0.02)
    loss = BoundedCrossEntropyLoss(reduction="none")
    x = torch.randn(5, 4, dtype=torch.float64)
    y = torch.randint(0, 3, (5,))
    dims = list(range(len(coarse)))
    coarse_l, coarse_u = _refined(bounded_model, x, y, loss, 0.1, dims, coarse, leaf_chunk=7)
    fine_l, fine_u = _refined(bounded_model, x, y, loss, 0.1, dims, fine, leaf_chunk=7)
    for cl, cu, fl, fu in zip(coarse_l, coarse_u, fine_l, fine_u):
        assert (fl >= cl - ATOL).all() and (fu <= cu + ATOL).all()


def test_chunking_does_not_change_the_bound():
    torch.manual_seed(0)
    bounded_model = _bounded(_model(4, 10, 3, depth=2), param_radius=0.02)
    loss = BoundedCrossEntropyLoss(reduction="none")
    x = torch.randn(5, 4, dtype=torch.float64)
    y = torch.randint(0, 3, (5,))

    ref = _refined(bounded_model, x, y, loss, 0.1, [0, 1, 2], [2, 3, 2], leaf_chunk=1)
    for leaf_chunk in (2, 5, 12, 64):
        other = _refined(bounded_model, x, y, loss, 0.1, [0, 1, 2], [2, 3, 2], leaf_chunk=leaf_chunk)
        for a, b in zip(ref[0] + ref[1], other[0] + other[1]):
            assert torch.equal(a, b)


def test_leaf_chunk_resolution():
    cfg = agt.InputRefinementConfig(splits=[2, 2])
    assert input_refinement.resolve_leaf_chunk(cfg, fragsize=1000, n_rows=100) == 10
    assert input_refinement.resolve_leaf_chunk(cfg, fragsize=10, n_rows=100) == 1
    cfg = agt.InputRefinementConfig(splits=[2, 2], leaf_chunk=3)
    assert input_refinement.resolve_leaf_chunk(cfg, fragsize=1000, n_rows=100) == 3


# ======================================================================================
# Soundness of the gradient bounds: sampled gradients lie inside them
# ======================================================================================


def _sample_in_box(lower, upper, generator, corners):
    """A point of the box [lower, upper]: one of its vertices when `corners`, else uniform inside."""
    u = torch.rand(lower.shape, generator=generator, dtype=lower.dtype)
    if corners:
        u = (u > 0.5).to(lower.dtype)
    return lower + u * (upper - lower)


@pytest.mark.parametrize("loss_name", ["cross_entropy", "binary_cross_entropy", "mse"])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("dims, splits", [([0, 1], [2, 2]), ([2, 0], [3, 2]), ([0, 1, 2], [2, 2, 4])])
def test_sampled_gradients_lie_inside_the_refined_bounds(loss_name, depth, dims, splits):
    """Soundness: for parameters anywhere in the parameter box and inputs anywhere in the eps-ball
    (including the vertices of both, where interval bounds are attained), each sample's parameter
    gradient lies inside the refined bounds."""
    generator = torch.Generator().manual_seed(1234)
    in_dim, hidden, batchsize = 3, 6, 4
    radius, eps = 0.02, 0.1
    loss, out_dim, y = _loss_and_labels(loss_name, batchsize, generator)
    base_model = _model(in_dim, hidden, out_dim, seed=0, depth=depth)
    bounded_model = _bounded(base_model, param_radius=radius)
    x = torch.randn(batchsize, in_dim, generator=generator, dtype=torch.float64)

    ref_l, ref_u = _refined(bounded_model, x, y, loss, eps, dims, splits)

    for trial in range(60):
        corners = trial % 2 == 0
        sampled_model = copy.deepcopy(base_model)
        with torch.no_grad():
            for p in sampled_model.parameters():
                p.copy_(_sample_in_box(p - radius, p + radius, generator, corners))
        x_sampled = _sample_in_box(x - eps, x + eps, generator, corners)

        # the nominal per-sample gradient at the sampled point, as AGT itself computes it
        sampled = IntervalBoundedModel(sampled_model)
        grads = sampled.backward(loss.backward(sampled.forward(x_sampled, retain_intermediate=True), y))
        for g, lo, hi in zip(grads, ref_l, ref_u):
            assert (g >= lo - ATOL).all(), (g - lo).min()
            assert (g <= hi + ATOL).all(), (g - hi).max()


def test_agt_nominal_gradients_match_autograd():
    """Ties the oracle of the containment test to an independent one."""
    torch.manual_seed(0)
    model = _model(3, 6, 3, depth=2)
    x = torch.randn(4, 3, dtype=torch.float64)
    y = torch.randint(0, 3, (4,))
    loss = BoundedCrossEntropyLoss(reduction="none")
    bounded_model = IntervalBoundedModel(model)
    grads = bounded_model.backward(loss.backward(bounded_model.forward(x, retain_intermediate=True), y))
    losses = torch.nn.functional.cross_entropy(model(x), y, reduction="none")
    for i in range(4):
        expected = torch.autograd.grad(losses[i], list(model.parameters()), retain_graph=True)
        for g, e in zip(grads, expected):
            assert torch.allclose(g[i], e, atol=1e-12)


# ======================================================================================
# The per-sample hull loses nothing under clamp clipping (assumption A2)
# ======================================================================================


def _update(k, param_n, grads_n, wp_l, wp_u, iwp_l, iwp_u, batchsize):
    accumulator = PoisoningGradientAccumulator(k, param_n)
    accumulator.add_poisoned_fragment_gradients(grads_n, wp_l, wp_u, iwp_l, iwp_u)
    update_l, _, update_u = accumulator.concretize_gradient_update(batchsize)
    return update_l, update_u


def _clip(grads_l, grads_n, grads_u, gamma):
    cloned = [[g.clone() for g in grads] for grads in (grads_l, grads_n, grads_u)]
    return training_utils.propagate_clipping(*cloned, gamma, "clamp")


@pytest.mark.parametrize("k, gamma", [(1, float("inf")), (2, float("inf")), (2, 0.05), (3, 0.05)])
def test_per_sample_hull_equals_joint_leaf_enumeration(k, gamma):
    """Hulling each sample's gradient over its own leaves gives the same batch update bound as
    enumerating every assignment of a leaf to each sample (L**B of them)."""
    torch.manual_seed(0)
    in_dim, hidden, out_dim, batchsize = 3, 4, 2, 3
    eps = 0.15
    bounded_model = _bounded(_model(in_dim, hidden, out_dim, seed=0), param_radius=0.02)
    loss = BoundedCrossEntropyLoss(reduction="none")
    x = torch.randn(batchsize, in_dim, dtype=torch.float64)
    y = torch.randint(0, out_dim, (batchsize,))
    param_n = bounded_model.param_n
    n_params = len(param_n)

    logits_n = bounded_model.forward(x, retain_intermediate=True)
    grads_n = bounded_model.backward(loss.backward(logits_n, y))
    wp_l, wp_u = bounded_model.bound_backward_combined(x, x, y, loss)

    n_leaves = 4
    x_l, x_u = input_refinement.leaf_boxes(x, eps, [0, 1], [2, 2], torch.arange(n_leaves))
    per_leaf = [
        bounded_model.bound_backward_combined(a, b, y, loss)
        for a, b in zip(x_l.split(batchsize), x_u.split(batchsize))
    ]

    hull_l = [torch.stack([leaf[0][p] for leaf in per_leaf]).amin(dim=0) for p in range(n_params)]
    hull_u = [torch.stack([leaf[1][p] for leaf in per_leaf]).amax(dim=0) for p in range(n_params)]
    wp_cl, gn_cl, wp_cu = _clip(wp_l, grads_n, wp_u, gamma)
    ps_l, _, ps_u = _clip(hull_l, grads_n, hull_u, gamma)
    ps_update_l, ps_update_u = _update(k, param_n, gn_cl, wp_cl, wp_cu, ps_l, ps_u, batchsize)

    joint_l = [torch.full_like(g, float("inf")) for g in ps_update_l]
    joint_u = [torch.full_like(g, float("-inf")) for g in ps_update_u]
    for assignment in itertools.product(range(n_leaves), repeat=batchsize):
        iwp_l = [torch.stack([per_leaf[assignment[i]][0][p][i] for i in range(batchsize)]) for p in range(n_params)]
        iwp_u = [torch.stack([per_leaf[assignment[i]][1][p][i] for i in range(batchsize)]) for p in range(n_params)]
        j_l, _, j_u = _clip(iwp_l, grads_n, iwp_u, gamma)
        u_l, u_u = _update(k, param_n, gn_cl, wp_cl, wp_cu, j_l, j_u, batchsize)
        joint_l = [torch.minimum(a, b) for a, b in zip(joint_l, u_l)]
        joint_u = [torch.maximum(a, b) for a, b in zip(joint_u, u_u)]

    for a, b in zip(ps_update_l + ps_update_u, joint_l + joint_u):
        assert torch.equal(a, b), (a - b).abs().max()


# ======================================================================================
# End to end: certified training
# ======================================================================================

N_TRAIN, N_FEATURES, BATCHSIZE = 240, 6, 60


def _dataset(loss_name: str, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(N_TRAIN + 100, N_FEATURES, generator=generator, dtype=torch.float64)
    y = (x[:, 0] + x[:, 1] ** 2 > 0.5).long()
    if loss_name == "binary_cross_entropy":
        y = y[:, None]
    return (x[:N_TRAIN], y[:N_TRAIN]), (x[N_TRAIN:], y[N_TRAIN:])


def _loader(x, y):
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=BATCHSIZE, shuffle=False)


def _config(loss_name: str, k_poison: int, epsilon: float, refinement=None, **overrides):
    settings = dict(n_epochs=2, learning_rate=0.2, loss=loss_name, k_poison=k_poison, epsilon=epsilon, clip_gamma=1.0)
    return agt.AGTConfig(**{**settings, **overrides}, input_refinement=refinement, log_level="ERROR")


def _fresh_model(loss_name: str):
    return _bounded(_model(N_FEATURES, 12, 2 if loss_name == "cross_entropy" else 1, seed=1, depth=2))


def _box_width(bounded_model: IntervalBoundedModel) -> float:
    return sum((u - l).sum().item() for l, u in zip(bounded_model.param_l, bounded_model.param_u))


def test_refined_training_is_valid_and_no_wider():
    (x, y), _ = _dataset("cross_entropy")
    baseline = agt.poison_certified_training(
        _fresh_model("cross_entropy"), _config("cross_entropy", 5, 0.05), _loader(x, y)
    )
    refinement = agt.InputRefinementConfig(splits=[2] * 4)
    refined = agt.poison_certified_training(
        _fresh_model("cross_entropy"), _config("cross_entropy", 5, 0.05, refinement), _loader(x, y)
    )
    for l, n, u in zip(refined.param_l, refined.param_n, refined.param_u):
        assert (l <= n).all() and (n <= u).all()
    for a, b in zip(refined.param_n, baseline.param_n):
        assert torch.equal(a, b)  # the nominal trajectory does not depend on the refinement
    assert _box_width(refined) < _box_width(baseline)


def _poison(x, k, epsilon, attack, seed):
    """Move k samples of every training batch by at most epsilon per feature (the certified threat
    model), without looking at the model."""
    generator = torch.Generator().manual_seed(seed)
    x = x.clone()
    for start in range(0, x.size(0), BATCHSIZE):
        rows = start + torch.randperm(BATCHSIZE, generator=generator)[:k]
        shape = (k, x.size(1))
        if attack == "corner":  # a random vertex of each ball
            delta = torch.where(torch.rand(shape, generator=generator) > 0.5, 1.0, -1.0)
        elif attack == "face":  # every poisoned sample pushed the same way: a coordinated shift
            delta = torch.where(torch.rand(x.size(1), generator=generator) > 0.5, 1.0, -1.0).expand(shape)
        else:  # "interior": uniform inside each ball
            delta = 2 * torch.rand(shape, generator=generator) - 1
        x[rows] += epsilon * delta.to(x.dtype)
    return x


class _AdaptivePoisoner:
    """
    A training "dataloader" that poisons each batch against the model as it currently stands, within
    the certified threat model: at most ``k`` samples per batch, each moved at most ``epsilon`` per
    feature.

    The attack steers the parameters along ``direction`` (one tensor per parameter). For every sample
    it searches its eps-ball, by projected signed gradient ascent, for the point whose clipped
    per-sample gradient moves the update furthest along ``direction``, then poisons the ``k`` samples
    that gain most. This is the adversary the per-step AGT bound is taken over, restricted to one
    direction, so it lands far closer to the edge of the certified box than model-blind poisoning.

    Set ``model`` to the bounded model being trained before iterating; batches are built lazily, so
    each one sees the parameters left by the previous step.
    """

    def __init__(self, x, y, k, epsilon, direction, loss_name, clip_gamma, clip_method, n_steps=10):
        self.x, self.y, self.k, self.epsilon = x, y, k, epsilon
        self.direction, self.loss_name = direction, loss_name
        self.clip_gamma, self.clip_method, self.n_steps = clip_gamma, clip_method, n_steps
        self.model: IntervalBoundedModel | None = None
        self.max_poisoned, self.max_shift = 0, 0.0

    def _loss(self, params, x, y):
        h = x
        for i in range(0, len(params), 2):
            h = h @ params[i].T + params[i + 1]
            if i + 2 < len(params):
                h = torch.relu(h)
        if self.loss_name == "cross_entropy":
            return torch.nn.functional.cross_entropy(h[None], y[None])
        return torch.nn.functional.binary_cross_entropy_with_logits(h, y.to(h.dtype))

    def _score(self, x, y, params):
        """How far one sample's clipped gradient moves the update along -direction."""
        grads = torch.func.grad(self._loss)(params, x, y)
        if self.clip_method == "clamp":
            grads = [g.clamp(-self.clip_gamma, self.clip_gamma) for g in grads]
        else:
            grads = [g * (self.clip_gamma / (g.norm() + 1e-6)).clamp(max=1.0) for g in grads]
        return -sum((g * v).sum() for g, v in zip(grads, self.direction))

    def _poison_batch(self, x, y):
        params = tuple(p.detach() for p in self.model.param_n)
        score = torch.func.vmap(self._score, in_dims=(0, 0, None))
        ascent = torch.func.vmap(torch.func.grad(self._score), in_dims=(0, 0, None))
        clean_score = score(x, y, params)
        best, best_score = x.clone(), clean_score.clone()
        candidate = x.clone()
        for _ in range(self.n_steps):
            step = self.epsilon / 4 * ascent(candidate, y, params).sign()
            candidate = torch.minimum(torch.maximum(candidate + step, x - self.epsilon), x + self.epsilon)
            # also try the vertex of the ball the search is heading for
            vertex = x + self.epsilon * (candidate - x).sign()
            for trial in (candidate, vertex):
                trial_score = score(trial, y, params)
                better = trial_score > best_score
                best[better], best_score[better] = trial[better], trial_score[better]
        rows = torch.topk(best_score - clean_score, self.k).indices
        poisoned = x.clone()
        poisoned[rows] = best[rows]
        self.max_poisoned = max(self.max_poisoned, ((poisoned - x).abs() > 0).any(dim=1).sum().item())
        self.max_shift = max(self.max_shift, (poisoned - x).abs().max().item())
        return poisoned

    def __iter__(self):
        for start in range(0, self.x.size(0), BATCHSIZE):
            x, y = self.x[start : start + BATCHSIZE], self.y[start : start + BATCHSIZE]
            yield self._poison_batch(x, y), y


def _attack_directions(certified: IntervalBoundedModel, seed: int = 0) -> list[list[torch.Tensor]]:
    """Directions to steer the parameters along: up and down the widest coordinate of each parameter
    tensor of the certified box, and two random sign patterns over all of them."""
    directions = []
    for i, (l, u) in enumerate(zip(certified.param_l, certified.param_u)):
        for sign in (1.0, -1.0):
            direction = [torch.zeros_like(p) for p in certified.param_n]
            direction[i].view(-1)[(u - l).argmax()] = sign
            directions.append(direction)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(2):
        directions.append(
            [torch.where(torch.rand(p.shape, generator=generator) > 0.5, 1.0, -1.0).to(p) for p in certified.param_n]
        )
    return directions


def _train_under_adaptive_attack(loss_name, x, y, k, epsilon, direction, clip_method, n_epochs):
    """Ordinary training (AGT's nominal pass) on batches poisoned adaptively along ``direction``."""
    config = _config(loss_name, 0, 0.0, clip_method=clip_method, n_epochs=n_epochs)
    poisoner = _AdaptivePoisoner(x, y, k, epsilon, direction, loss_name, config.clip_gamma, clip_method)
    poisoner.model = _fresh_model(loss_name)
    agt.poison_certified_training(poisoner.model, config, poisoner)
    assert poisoner.max_poisoned <= k and poisoner.max_shift <= epsilon + 1e-12
    return poisoner.model


def _worst_violation(poisoned: IntervalBoundedModel, certified: IntervalBoundedModel) -> float:
    """How far outside the certified box the poisoned parameters are (<= 0 when inside)."""
    return max(
        max((l - p).max().item(), (p - u).max().item())
        for p, l, u in zip(poisoned.param_n, certified.param_l, certified.param_u)
    )


# The certified box compounds from step to step much faster than any real attack does, so short runs
# are where an attack gets close enough to the edge of the box to expose an unsound bound; the
# 8-step case checks that nothing breaks over a longer run.
CONTAINMENT_CASES = [
    # loss, clip method, splits, strategy, batches per epoch, epochs
    ("cross_entropy", "clamp", [2, 2, 2], "sensitivity", 1, 1),
    ("cross_entropy", "clamp", [3, 2], "sensitivity", 2, 1),
    ("cross_entropy", "clamp", [2] * 6, "first", 2, 1),
    ("cross_entropy", "norm", [2, 2, 2], "sensitivity", 2, 1),
    ("cross_entropy", "clamp", [3, 2, 2], "sensitivity", 4, 2),
    ("binary_cross_entropy", "clamp", [2, 2, 2], "sensitivity", 1, 1),
    ("binary_cross_entropy", "clamp", [3, 2], "first", 2, 1),
    ("binary_cross_entropy", "clamp", [2] * 6, "sensitivity", 2, 1),
    ("binary_cross_entropy", "norm", [3, 2], "sensitivity", 2, 1),
    ("binary_cross_entropy", "norm", [2, 2, 2], "sensitivity", 4, 2),
]


@pytest.mark.parametrize("loss_name, clip_method, splits, strategy, n_batches, n_epochs", CONTAINMENT_CASES)
def test_models_trained_on_poisoned_data_lie_inside_the_refined_box(
    loss_name, clip_method, splits, strategy, n_batches, n_epochs
):
    """
    Soundness, end to end: train on data that an adversary within the threat model has actually
    poisoned, and check the resulting parameters lie inside the refined certified box and score at
    least the certified accuracy. Both model-blind poisoning and the adaptive attack are tried.

    The poisoned models are trained by AGT's own nominal pass (k_poison = 0, epsilon = 0), which is
    ordinary SGD on whatever data it is given.
    """
    k_poison, epsilon = 5, 0.05
    (x, y), (x_test, y_test) = _dataset(loss_name)
    x, y = x[: n_batches * BATCHSIZE], y[: n_batches * BATCHSIZE]
    refinement = agt.InputRefinementConfig(splits=splits, strategy=strategy)
    certified = agt.poison_certified_training(
        _fresh_model(loss_name),
        _config(loss_name, k_poison, epsilon, refinement, clip_method=clip_method, n_epochs=n_epochs),
        _loader(x, y),
    )
    certified_acc, _, _ = test_metrics.test_accuracy(certified, x_test, y_test)

    poisoned_models = []
    for attack in ("corner", "face", "interior"):
        x_poisoned = _poison(x, k_poison, epsilon, attack, seed=0)
        assert ((x_poisoned - x).abs() > 0).any(dim=1).view(-1, BATCHSIZE).sum(dim=1).max() <= k_poison
        assert (x_poisoned - x).abs().max() <= epsilon + 1e-12
        poisoned_models.append(
            agt.poison_certified_training(
                _fresh_model(loss_name),
                _config(loss_name, 0, 0.0, clip_method=clip_method, n_epochs=n_epochs),
                _loader(x_poisoned, y),
            )
        )
    for direction in _attack_directions(certified):
        poisoned_models.append(
            _train_under_adaptive_attack(loss_name, x, y, k_poison, epsilon, direction, clip_method, n_epochs)
        )

    for poisoned in poisoned_models:
        assert _worst_violation(poisoned, certified) <= ATOL
        _, poisoned_acc, _ = test_metrics.test_accuracy(poisoned, x_test, y_test)
        assert poisoned_acc >= certified_acc - ATOL


@pytest.mark.parametrize("loss_name", ["cross_entropy", "binary_cross_entropy"])
@pytest.mark.parametrize("certified_k, certified_eps", [(1, 0.05), (5, 0.01)])
@pytest.mark.parametrize("n_batches", [1, 2])
def test_the_adaptive_attack_escapes_an_under_certified_box(loss_name, certified_k, certified_eps, n_batches):
    """The containment test above has teeth: a box certified against a weaker adversary than the one
    attacking (fewer poisoned samples, or a smaller radius) is escaped by the adaptive attack."""
    k_poison, epsilon = 5, 0.05
    (x, y), _ = _dataset(loss_name)
    x, y = x[: n_batches * BATCHSIZE], y[: n_batches * BATCHSIZE]
    refinement = agt.InputRefinementConfig(splits=[2] * 6)
    certified = agt.poison_certified_training(
        _fresh_model(loss_name), _config(loss_name, certified_k, certified_eps, refinement, n_epochs=1), _loader(x, y)
    )
    violations = [
        _worst_violation(
            _train_under_adaptive_attack(loss_name, x, y, k_poison, epsilon, direction, "clamp", 1), certified
        )
        for direction in _attack_directions(certified)
    ]
    assert max(violations) > 1e-6


@pytest.mark.parametrize("loss_name", ["cross_entropy", "binary_cross_entropy"])
def test_the_adaptive_attack_escapes_a_box_built_from_too_few_leaves(monkeypatch, loss_name):
    """The containment test catches a broken refinement: a hull that silently leaves out part of the
    ball (here, every leaf but the first) gives a box the adaptive attack escapes."""
    leaf_boxes = input_refinement.leaf_boxes
    monkeypatch.setattr(
        input_refinement,
        "leaf_boxes",
        lambda batch, epsilon, dims, splits, leaf_ids: leaf_boxes(
            batch, epsilon, dims, splits, torch.zeros_like(leaf_ids)
        ),
    )
    k_poison, epsilon = 5, 0.05
    (x, y), _ = _dataset(loss_name)
    x, y = x[:BATCHSIZE], y[:BATCHSIZE]
    refinement = agt.InputRefinementConfig(splits=[2] * 6)
    broken = agt.poison_certified_training(
        _fresh_model(loss_name), _config(loss_name, k_poison, epsilon, refinement, n_epochs=1), _loader(x, y)
    )
    violations = [
        _worst_violation(_train_under_adaptive_attack(loss_name, x, y, k_poison, epsilon, direction, "clamp", 1), broken)
        for direction in _attack_directions(broken)
    ]
    assert max(violations) > 1e-6


def test_poisoned_models_lie_inside_the_refined_box_with_clean_data():
    """As above with a trusted clean batch alongside every poisoned one (the dl_clean path)."""
    k_poison, epsilon = 5, 0.05
    (x, y), _ = _dataset("cross_entropy")
    (x_clean, y_clean), _ = _dataset("cross_entropy", seed=7)
    config = _config("cross_entropy", k_poison, epsilon, agt.InputRefinementConfig(splits=[2, 2, 2]))
    certified = agt.poison_certified_training(
        _fresh_model("cross_entropy"), config, _loader(x, y), dl_clean=_loader(x_clean, y_clean)
    )
    for seed in range(4):
        poisoned = agt.poison_certified_training(
            _fresh_model("cross_entropy"),
            _config("cross_entropy", 0, 0.0),
            _loader(_poison(x, k_poison, epsilon, "corner", seed), y),
            dl_clean=_loader(x_clean, y_clean),
        )
        for p, l, u in zip(poisoned.param_n, certified.param_l, certified.param_u):
            assert (p >= l - ATOL).all() and (p <= u + ATOL).all()


# ======================================================================================
# Configuration and validation
# ======================================================================================


def _refined_config(**refinement) -> agt.AGTConfig:
    return agt.AGTConfig(
        n_epochs=1, learning_rate=0.1, loss="cross_entropy", k_poison=5, epsilon=0.1,
        input_refinement=agt.InputRefinementConfig(**refinement),
    )


def test_config_rejects_epsilon_zero():
    with pytest.raises(ValueError, match="epsilon"):
        agt.AGTConfig(
            n_epochs=1, learning_rate=0.01, loss="cross_entropy", k_poison=5, epsilon=0.0,
            input_refinement=agt.InputRefinementConfig(splits=[2]),
        )


def test_config_rejects_k_poison_zero():
    with pytest.raises(ValueError, match="k_poison"):
        agt.AGTConfig(
            n_epochs=1, learning_rate=0.01, loss="cross_entropy", k_poison=0, epsilon=0.1,
            input_refinement=agt.InputRefinementConfig(splits=[2]),
        )


@pytest.mark.parametrize("splits", [[], [1], [2, 1], [2, 0]])
def test_config_rejects_splits_that_do_not_cut(splits):
    with pytest.raises(ValueError, match="splits"):
        agt.InputRefinementConfig(splits=splits)


def test_config_leaf_count():
    assert agt.InputRefinementConfig(splits=[3, 3, 2, 2, 2]).n_leaves == 3**2 * 2**3


def test_entry_rejects_over_max_leaves():
    config = _refined_config(splits=[3, 3, 2, 2, 2], max_leaves=3**2 * 2**3 - 1)
    with pytest.raises(ValueError, match="max_leaves"):
        input_refinement.validate_input_refinement(_bounded(_model(6, 8, 2)), config)


def test_sensitivity_ranks_by_first_layer_weight():
    bounded_model = _bounded(_model(6, 8, 2))
    weight = bounded_model.param_n[0]
    expected = torch.argsort(weight.abs().sum(dim=0), descending=True).tolist()
    cfg = agt.InputRefinementConfig(splits=[3, 3, 2, 2])
    assert input_refinement.select_split_dims(bounded_model, 0.1, cfg) == expected[:4]
    assert input_refinement.select_split_dims(
        bounded_model, 0.1, agt.InputRefinementConfig(splits=[2, 2, 2], strategy="first")
    ) == [0, 1, 2]


def test_sensitivity_rejects_transform():
    torch.manual_seed(0)
    transform = IntervalBoundedModel(torch.nn.Sequential(torch.nn.Linear(6, 6)).double(), trainable=False)
    bounded_model = IntervalBoundedModel(_model(6, 8, 2), transform=transform)
    with pytest.raises(ValueError, match="transform"):
        input_refinement.select_split_dims(bounded_model, 0.1, agt.InputRefinementConfig(splits=[2]))


def test_sensitivity_rejects_non_linear_first_layer():
    bounded_model = IntervalBoundedModel(torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(6, 2)).double())
    with pytest.raises(ValueError, match="Linear"):
        input_refinement.select_split_dims(bounded_model, 0.1, agt.InputRefinementConfig(splits=[2]))


def test_sensitivity_rejects_more_splits_than_inputs():
    with pytest.raises(ValueError, match="inputs"):
        input_refinement.select_split_dims(_bounded(_model(3, 8, 2)), 0.1, agt.InputRefinementConfig(splits=[2] * 4))


def test_entry_rejects_batch_coupled_layer():
    bounded_model = _bounded(_model(6, 8, 2))
    bounded_model.modules.append(torch.nn.BatchNorm1d(8))  # IBP forbids it at construction; guard it anyway
    with pytest.raises(ValueError, match="batch-coupled"):
        input_refinement.validate_input_refinement(bounded_model, _refined_config(splits=[2, 2], strategy="first"))


def test_shard_leaves_requires_process_group():
    assert not torch.distributed.is_initialized()
    with pytest.raises(ValueError, match="process group"):
        input_refinement.validate_input_refinement(
            _bounded(_model(6, 8, 2)), _refined_config(splits=[2, 2], shard_leaves=True)
        )


def test_hash_ignores_sharding_and_tracks_the_partition():
    assert _refined_config(splits=[2, 2]).hash() == _refined_config(splits=[2, 2], shard_leaves=True).hash()
    hashes = {_refined_config(splits=splits).hash() for splits in ([2, 2], [2, 3], [3, 2], [2, 2, 2])}
    assert len(hashes) == 4


def test_disabled_refinement_is_absent_from_the_serialised_config():
    config = agt.AGTConfig(n_epochs=1, learning_rate=0.1, loss="cross_entropy")
    assert "input_refinement" not in config.asdict()


# ======================================================================================
# Sharding: splitting the leaves across ranks (gloo, CPU) changes nothing.
# Workers are module-level so torch.multiprocessing.spawn can import them.
# ======================================================================================


def _init_process_group(rank: int, world_size: int, init_file: str) -> None:
    torch.distributed.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)


def _sharding_problem(splits: list[int], dims: list[int]):
    torch.manual_seed(0)
    bounded_model = _bounded(_model(4, 10, 3, depth=2), param_radius=0.02)
    x = torch.randn(5, 4, dtype=torch.float64)
    y = torch.randint(0, 3, (5,))
    return bounded_model, x, y, BoundedCrossEntropyLoss(reduction="none"), 0.1, dims, splits


def _sharded_bound_worker(rank: int, world_size: int, init_file: str, out_dir: str, splits, dims) -> None:
    _init_process_group(rank, world_size, init_file)
    try:
        grads = _refined(*_sharding_problem(splits, dims), leaf_chunk=2, shard=True)
        torch.save(grads, f"{out_dir}/rank{rank}.pt")
    finally:
        torch.distributed.destroy_process_group()


# world size 5 against 4 leaves leaves one rank empty; 18 leaves over 4 ranks gives uneven blocks
@pytest.mark.parametrize(
    "splits, dims, world_size", [([2, 2], [0, 1], 2), ([2, 2], [0, 1], 3), ([2, 2], [0, 1], 5), ([3, 2, 3], [0, 2, 3], 4)]
)
def test_sharded_bound_matches_single_process(tmp_path, splits, dims, world_size):
    ref_l, ref_u = _refined(*_sharding_problem(splits, dims), leaf_chunk=1)
    torch.multiprocessing.spawn(
        _sharded_bound_worker, args=(world_size, str(tmp_path / "pg"), str(tmp_path), splits, dims), nprocs=world_size
    )
    for rank in range(world_size):
        grads_l, grads_u = torch.load(tmp_path / f"rank{rank}.pt")
        for a, b in zip(ref_l + ref_u, grads_l + grads_u):
            assert torch.equal(a, b)


def _sharded_train(shard_leaves: bool, data_seed: int = 0) -> IntervalBoundedModel:
    (x, y), _ = _dataset("cross_entropy", seed=data_seed)
    refinement = agt.InputRefinementConfig(splits=[3, 2, 2, 2], shard_leaves=shard_leaves)
    return agt.poison_certified_training(
        _fresh_model("cross_entropy"), _config("cross_entropy", 5, 0.05, refinement), _loader(x, y)
    )


def _sharded_train_worker(rank: int, world_size: int, init_file: str, out_dir: str, rank_dependent_data: bool) -> None:
    _init_process_group(rank, world_size, init_file)
    try:
        trained = _sharded_train(shard_leaves=True, data_seed=rank if rank_dependent_data else 0)
        torch.save((trained.param_l, trained.param_n, trained.param_u), f"{out_dir}/rank{rank}.pt")
    finally:
        torch.distributed.destroy_process_group()


# rank_dependent_data gives ranks > 0 different data: rank 0 is authoritative, so the result must not move
@pytest.mark.parametrize("rank_dependent_data", [False, True])
def test_sharded_training_matches_single_process(tmp_path, rank_dependent_data):
    reference = _sharded_train(shard_leaves=False)
    world_size = 2
    torch.multiprocessing.spawn(
        _sharded_train_worker,
        args=(world_size, str(tmp_path / "pg"), str(tmp_path), rank_dependent_data),
        nprocs=world_size,
    )
    for rank in range(world_size):
        params = torch.load(tmp_path / f"rank{rank}.pt")
        for expected, actual in zip((reference.param_l, reference.param_n, reference.param_u), params):
            for a, b in zip(expected, actual):
                assert torch.equal(a, b)
