from __future__ import annotations

import copy
import itertools

import pytest
import torch

import abstract_gradient_training as agt
from abstract_gradient_training import input_refinement, training_utils
from abstract_gradient_training.bounded_losses import BoundedCrossEntropyLoss
from abstract_gradient_training.bounded_models import IntervalBoundedModel
from abstract_gradient_training.gradient_accumulation import PoisoningGradientAccumulator

ATOL = 1e-6


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


def _clone(grads: list[torch.Tensor]) -> list[torch.Tensor]:
    return [g.clone() for g in grads]


# ======================================================================================
# T1 -- identity: a disabled refinement is bit-identical to the shipped single-box call.
# ======================================================================================


@pytest.mark.parametrize("dims, n_splits", [([0, 1], 1), ([], 4), ([0, 2], 1)])
def test_t1_identity_direct(dims, n_splits):
    torch.manual_seed(0)
    bounded_model = _bounded(_model(4, 8, 3), param_radius=0.01)
    loss = BoundedCrossEntropyLoss(reduction="none")
    x = torch.randn(6, 4, dtype=torch.float64)
    y = torch.randint(0, 3, (6,))
    eps = 0.1

    base_l, base_u = bounded_model.bound_backward_combined(x - eps, x + eps, y, loss)
    ref_l, ref_u = input_refinement.refined_bound_backward_combined(
        bounded_model, x, y, loss, epsilon=eps, dims=dims, n_splits=n_splits, leaf_chunk=1
    )
    for a, b in zip(ref_l, base_l):
        assert torch.equal(a, b)
    for a, b in zip(ref_u, base_u):
        assert torch.equal(a, b)


def test_t1_disabled_path_never_refines(monkeypatch):
    """With input_refinement=None the poisoned path must not enter the refinement code at all."""

    def _boom(*args, **kwargs):  # pragma: no cover - only runs on regression
        raise AssertionError("refined_bound_backward_combined called with input_refinement disabled")

    monkeypatch.setattr(input_refinement, "refined_bound_backward_combined", _boom)

    bounded_model = _bounded(_model(5, 8, 2), param_radius=0.0)
    config = agt.AGTConfig(
        n_epochs=1, learning_rate=0.01, loss="cross_entropy", k_poison=3, epsilon=0.05, clip_gamma=1.0
    )
    x = torch.randn(8, 5, dtype=torch.float64)
    y = torch.randint(0, 2, (8,))
    training_utils.compute_batch_gradients(bounded_model, x, y, config, nominal=False, poisoned=True)


# ======================================================================================
# T2 -- tightening direction: refinement can never widen the bound.
# ======================================================================================


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_t2_tightening_direction(seed):
    torch.manual_seed(seed)
    bounded_model = _bounded(_model(5, 12, 3, seed=seed, depth=2), param_radius=0.02)
    loss = BoundedCrossEntropyLoss(reduction="none")
    x = torch.randn(7, 5, dtype=torch.float64)
    y = torch.randint(0, 3, (7,))
    eps = 0.08

    base_l, base_u = bounded_model.bound_backward_combined(x - eps, x + eps, y, loss)
    ref_l, ref_u = input_refinement.refined_bound_backward_combined(
        bounded_model, x, y, loss, epsilon=eps, dims=[0, 1, 2], n_splits=2, leaf_chunk=4
    )
    for a, b in zip(ref_l, base_l):
        assert (a >= b - ATOL).all(), (a - b).min()
    for a, b in zip(ref_u, base_u):
        assert (a <= b + ATOL).all(), (a - b).max()


# ======================================================================================
# T3 -- ground-truth containment: exact autograd gradients lie within the refined bounds.
# ======================================================================================


def test_t3_ground_truth_containment():
    torch.manual_seed(0)
    in_dim, hidden, out_dim, batchsize = 3, 6, 2, 4
    radius, eps = 0.02, 0.1
    base_model = _model(in_dim, hidden, out_dim, seed=0)
    bounded_model = _bounded(base_model, param_radius=radius)
    loss = BoundedCrossEntropyLoss(reduction="none")

    x = torch.randn(batchsize, in_dim, dtype=torch.float64)
    y = torch.randint(0, out_dim, (batchsize,))

    ref_l, ref_u = input_refinement.refined_bound_backward_combined(
        bounded_model, x, y, loss, epsilon=eps, dims=[0, 1], n_splits=2, leaf_chunk=4
    )

    param_n = bounded_model.param_n
    generator = torch.Generator().manual_seed(1234)
    for _ in range(40):
        ref_model = copy.deepcopy(base_model)
        with torch.no_grad():
            for p, p_n in zip(ref_model.parameters(), param_n):
                noise = (2 * torch.rand(p.shape, generator=generator, dtype=torch.float64) - 1) * radius
                p.copy_(p_n + noise)
        x_prime = x + (2 * torch.rand(x.shape, generator=generator, dtype=torch.float64) - 1) * eps

        out = ref_model(x_prime)
        losses = torch.nn.functional.cross_entropy(out, y, reduction="none")
        for i in range(batchsize):
            grads_i = torch.autograd.grad(losses[i], list(ref_model.parameters()), retain_graph=True)
            for g, lo, hi in zip(grads_i, ref_l, ref_u):
                assert (g >= lo[i] - ATOL).all(), (g - lo[i]).min()
                assert (g <= hi[i] + ATOL).all(), (g - hi[i]).max()


# ======================================================================================
# T4 -- chunk invariance: any grouping of the same leaves gives bit-identical bounds.
# ======================================================================================


def test_t4_chunk_invariance():
    torch.manual_seed(0)
    bounded_model = _bounded(_model(4, 10, 3, depth=2), param_radius=0.02)
    loss = BoundedCrossEntropyLoss(reduction="none")
    x = torch.randn(5, 4, dtype=torch.float64)
    y = torch.randint(0, 3, (5,))
    eps = 0.1
    kwargs = dict(epsilon=eps, dims=[0, 1, 2], n_splits=2)  # L = 8

    ref = input_refinement.refined_bound_backward_combined(bounded_model, x, y, loss, leaf_chunk=1, **kwargs)
    for leaf_chunk in (2, 3, 8, 64):
        other = input_refinement.refined_bound_backward_combined(
            bounded_model, x, y, loss, leaf_chunk=leaf_chunk, **kwargs
        )
        for a, b in zip(ref[0], other[0]):
            assert torch.equal(a, b)
        for a, b in zip(ref[1], other[1]):
            assert torch.equal(a, b)


# ======================================================================================
# T5 -- joint-leaf equivalence: per-sample hulling equals the L**B joint enumeration
#       under clamp clipping (the whole shared-grid argument).
# ======================================================================================


def _apply_clip(grads_l, grads_n, grads_u, gamma):
    return training_utils.propagate_clipping(
        _clone(grads_l), _clone(grads_n), _clone(grads_u), gamma, "clamp"
    )


def _update(k, param_n, grads_n, wp_l, wp_u, iwp_l, iwp_u, batchsize):
    accumulator = PoisoningGradientAccumulator(k, param_n)
    accumulator.add_poisoned_fragment_gradients(_clone(grads_n), _clone(wp_l), _clone(wp_u), _clone(iwp_l), _clone(iwp_u))
    update_l, _, update_u = accumulator.concretize_gradient_update(batchsize)
    return update_l, update_u


@pytest.mark.parametrize("k, gamma", [(1, float("inf")), (2, float("inf")), (2, 0.05), (3, 0.05)])
def test_t5_joint_leaf_equivalence(k, gamma):
    torch.manual_seed(0)
    in_dim, hidden, out_dim, batchsize = 3, 4, 2, 3
    eps = 0.15
    bounded_model = _bounded(_model(in_dim, hidden, out_dim, seed=0), param_radius=0.02)
    loss = BoundedCrossEntropyLoss(reduction="none")
    x = torch.randn(batchsize, in_dim, dtype=torch.float64)
    y = torch.randint(0, out_dim, (batchsize,))
    param_n = bounded_model.param_n

    # nominal gradients and the exact-input (weight-perturbed) bounds
    logits_n = bounded_model.forward(x, retain_intermediate=True)
    grads_n = bounded_model.backward(loss.backward(logits_n, y))
    wp_l, wp_u = bounded_model.bound_backward_combined(x, x, y, loss)

    # per-leaf per-sample bounds, computed once and shared by both enumerations
    leaves = list(input_refinement.iter_leaf_boxes(x, eps, [0, 1], 2))  # L = 4
    per_leaf = [bounded_model.bound_backward_combined(x_l, x_u, y, loss) for x_l, x_u in leaves]
    n_leaves = len(leaves)

    # per-sample hull over leaves -> clip -> accumulator
    hull_l = [torch.stack([leaf[0][p] for leaf in per_leaf]).amin(dim=0) for p in range(len(param_n))]
    hull_u = [torch.stack([leaf[1][p] for leaf in per_leaf]).amax(dim=0) for p in range(len(param_n))]
    wp_cl, gn_cl, wp_cu = _apply_clip(wp_l, grads_n, wp_u, gamma)
    ps_l, _, ps_u = _apply_clip(hull_l, grads_n, hull_u, gamma)
    ps_update_l, ps_update_u = _update(k, param_n, gn_cl, wp_cl, wp_cu, ps_l, ps_u, batchsize)

    # joint enumeration: every assignment of a leaf to each sample, hulled over assignments
    joint_l = [torch.full_like(g, float("inf")) for g in ps_update_l]
    joint_u = [torch.full_like(g, float("-inf")) for g in ps_update_u]
    for assignment in itertools.product(range(n_leaves), repeat=batchsize):
        iwp_l = [
            torch.stack([per_leaf[assignment[i]][0][p][i] for i in range(batchsize)]) for p in range(len(param_n))
        ]
        iwp_u = [
            torch.stack([per_leaf[assignment[i]][1][p][i] for i in range(batchsize)]) for p in range(len(param_n))
        ]
        j_l, _, j_u = _apply_clip(iwp_l, grads_n, iwp_u, gamma)
        u_l, u_u = _update(k, param_n, gn_cl, wp_cl, wp_cu, j_l, j_u, batchsize)
        joint_l = [torch.minimum(a, b) for a, b in zip(joint_l, u_l)]
        joint_u = [torch.maximum(a, b) for a, b in zip(joint_u, u_u)]

    for a, b in zip(ps_update_l, joint_l):
        assert torch.equal(a, b), (a - b).abs().max()
    for a, b in zip(ps_update_u, joint_u):
        assert torch.equal(a, b), (a - b).abs().max()


# ======================================================================================
# T6 -- config validation: every degenerate or unsound configuration raises.
# ======================================================================================


def test_t6_config_rejects_epsilon_zero():
    with pytest.raises(ValueError, match="epsilon"):
        agt.AGTConfig(
            n_epochs=1, learning_rate=0.01, loss="cross_entropy", k_poison=5, epsilon=0.0,
            input_refinement=agt.InputRefinementConfig(n_splits=2),
        )


def test_t6_config_rejects_k_poison_zero():
    with pytest.raises(ValueError, match="k_poison"):
        agt.AGTConfig(
            n_epochs=1, learning_rate=0.01, loss="cross_entropy", k_poison=0, epsilon=0.1,
            input_refinement=agt.InputRefinementConfig(n_splits=2),
        )


def test_t6_config_rejects_n_splits_one():
    with pytest.raises(ValueError, match="n_splits"):
        agt.InputRefinementConfig(n_splits=1)


def test_t6_entry_rejects_over_max_leaves():
    bounded_model = _bounded(_model(6, 8, 2), param_radius=0.0)
    config = agt.AGTConfig(
        n_epochs=1, learning_rate=0.01, loss="cross_entropy", k_poison=5, epsilon=0.1,
        input_refinement=agt.InputRefinementConfig(n_splits=4, n_dims=5, max_leaves=256),
    )
    with pytest.raises(ValueError, match="max_leaves"):
        input_refinement.validate_input_refinement(bounded_model, config)


def test_t6_sensitivity_rejects_transform():
    torch.manual_seed(0)
    transform = IntervalBoundedModel(torch.nn.Sequential(torch.nn.Linear(6, 6)).double(), trainable=False)
    bounded_model = IntervalBoundedModel(_model(6, 8, 2), transform=transform)
    with pytest.raises(ValueError, match="transform"):
        input_refinement.select_split_dims(bounded_model, 0.1, agt.InputRefinementConfig(n_splits=2, strategy="sensitivity"))


def test_t6_sensitivity_rejects_non_linear_first_layer():
    bounded_model = IntervalBoundedModel(torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(6, 2)).double())
    with pytest.raises(ValueError, match="Linear"):
        input_refinement.select_split_dims(bounded_model, 0.1, agt.InputRefinementConfig(n_splits=2, strategy="sensitivity"))


def test_t6_entry_rejects_batch_coupled_layer():
    bounded_model = _bounded(_model(6, 8, 2), param_radius=0.0)
    bounded_model.modules.append(torch.nn.BatchNorm1d(8))  # IBP forbids it at construction; guard it anyway
    config = agt.AGTConfig(
        n_epochs=1, learning_rate=0.01, loss="cross_entropy", k_poison=5, epsilon=0.1,
        input_refinement=agt.InputRefinementConfig(n_splits=2, n_dims=2, strategy="widest"),
    )
    with pytest.raises(ValueError, match="batch-coupled"):
        input_refinement.validate_input_refinement(bounded_model, config)


# ======================================================================================
# T7 -- end to end: a short poisoning run, baseline vs 16 leaves.
# ======================================================================================


def _halfmoons_loader(seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(240, 6, dtype=torch.float64)
    y = (x[:, 0] + x[:, 1] > 0).long()
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=60)


def _box_width(bounded_model: IntervalBoundedModel) -> float:
    return sum((u - l).sum().item() for l, u in zip(bounded_model.param_l, bounded_model.param_u))


def test_t7_end_to_end_no_wider():
    common = dict(n_epochs=2, learning_rate=0.02, loss="cross_entropy", k_poison=5, epsilon=0.05, clip_gamma=1.0)

    baseline = agt.poison_certified_training(
        _bounded(_model(6, 12, 2, seed=1, depth=2)), agt.AGTConfig(**common), _halfmoons_loader()
    )
    refined = agt.poison_certified_training(
        _bounded(_model(6, 12, 2, seed=1, depth=2)),
        agt.AGTConfig(**common, input_refinement=agt.InputRefinementConfig(n_splits=2, n_dims=4)),
        _halfmoons_loader(),
    )

    training_utils.validate_bounded_model(refined)
    for l, u in zip(refined.param_l, refined.param_u):
        assert (u >= l).all()
    assert _box_width(refined) <= _box_width(baseline) + ATOL
