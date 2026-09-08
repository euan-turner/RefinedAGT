"""
Input-ball refinement for poison-certified training.

Partitions the feature-poisoning l_inf ball into a grid of sub-boxes, propagates each leaf through
the ordinary bounding pass, and hulls the per-sample parameter-gradient bounds over the leaves. The
result is a sound, and never wider, replacement for the single-box bound. When ``n_splits <= 1``, refinement
reduces to the default bounding method.

The refinement is exponential in the number of split dimensions (``n_splits ** len(dims)`` bounding
passes per fragment) and flat in the batch size. It calls only ``bound_backward_combined``, so it
composes with every ``BoundedModel`` bounding method; but is impractical when the per-leaf work
is heavy, such as the MIP-bounded model.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator

import torch

from abstract_gradient_training.bounded_losses import BoundedLoss
from abstract_gradient_training.bounded_models import BoundedModel
from abstract_gradient_training.configuration import AGTConfig, InputRefinementConfig

LOGGER = logging.getLogger(__name__)

# Layers whose per-sample output depends on the rest of the batch. Their presence breaks the
# requirement no coupling across the batch
_BATCH_COUPLED_LAYERS = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
)


def select_split_dims(bounded_model: BoundedModel, epsilon: float, cfg: InputRefinementConfig) -> list[int]:
    """
    Choose which flattened input coordinates to partition.

    The ``"sensitivity"`` strategy ranks coordinate ``d`` by the first-layer pre-activation width it
    injects, ``2 * epsilon * sum_j |W_1[j, d]|``, read from the nominal weights; ``"widest"`` and
    ``"first"`` split the leading coordinates (identical under a scalar ``epsilon``).

    Raises for ``"sensitivity"`` when the first-layer weight matrix is not the one the eps-ball sees
    -- a fixed ``transform`` or a non-``Linear`` first layer -- rather than falling back silently.

    Returns:
        list[int]: coordinate indices into the flattened input, at most ``cfg.n_dims`` of them.
    """
    first = bounded_model.modules[0] if bounded_model.modules else None

    if cfg.strategy == "sensitivity":
        if bounded_model.transform is not None:
            raise ValueError(
                "input_refinement strategy='sensitivity' cannot see through a fixed transform: the "
                "eps-ball is on the raw input but the first trainable layer sees the transformed "
                "features. Use strategy='widest' or 'first'."
            )
        if not isinstance(first, torch.nn.Linear):
            raise ValueError(
                f"input_refinement strategy='sensitivity' needs a Linear first layer to score input "
                f"coordinates; got {type(first).__name__}. Use strategy='widest' or 'first'. Note "
                f"that input splitting is not expected to pay above ~10 input dimensions, where ReLU "
                f"branching dominates."
            )
        weight = first.weight.detach()
        scores = 2 * epsilon * weight.abs().sum(dim=0)
        return torch.argsort(scores, descending=True)[: cfg.n_dims].tolist()

    # "widest" and "first": split the leading coordinates. iter_leaf_boxes clamps to the true
    # flattened input dimension, so requesting more than the model has is harmless.
    return list(range(cfg.n_dims))


def iter_leaf_boxes(
    batch: torch.Tensor, epsilon: float, dims: list[int], n_splits: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Yield the ``(lower, upper)`` corners of each leaf of the partitioned l_inf ball, one leaf at a
    time.

    Each coordinate in ``dims`` (an index into the flattened input) is cut into ``n_splits`` equal
    intervals; every other coordinate keeps its full ``[x - epsilon, x + epsilon]`` range. There are
    ``n_splits ** len(dims)`` leaves, a single grid shared by every sample in ``batch``. 
    Leaves are built lazily from ``batch`` rather than materialised together, 
    so peak memory is one leaf's worth of corners regardless of the count.

    When ``n_splits <= 1`` or no coordinate is cut, yields a single leaf equal to
    ``(batch - epsilon, batch + epsilon)``.

    Yields:
        tuple[torch.Tensor, torch.Tensor]: ``(x_l, x_u)`` shaped like ``batch``, the corners of one
            leaf for the whole batch.
    """
    lower = batch - epsilon
    upper = batch + epsilon

    if batch.dim() < 2:
        yield lower, upper
        return

    lower_flat = lower.flatten(1)
    upper_flat = upper.flatten(1)
    cut_dims = [d for d in dims if d < lower_flat.size(1)]

    if n_splits <= 1 or not cut_dims:
        yield lower, upper
        return

    width = 2.0 * epsilon / n_splits
    for offsets in itertools.product(range(n_splits), repeat=len(cut_dims)):
        leaf_l = lower_flat.clone()
        leaf_u = upper_flat.clone()
        for d, offset in zip(cut_dims, offsets):
            leaf_l[:, d] = lower_flat[:, d] + offset * width
            leaf_u[:, d] = lower_flat[:, d] + (offset + 1) * width
        yield leaf_l.view_as(batch), leaf_u.view_as(batch)


def _propagate_chunk(
    bounded_model: BoundedModel,
    chunk: list[tuple[torch.Tensor, torch.Tensor]],
    labels: torch.Tensor,
    loss: BoundedLoss,
    batchsize: int,
    label_k_poison: int,
    label_epsilon: float,
    poison_target_idx: int,
    hull_l: list[torch.Tensor] | None,
    hull_u: list[torch.Tensor] | None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Propagate one group of leaves as a single batched call and fold it into the running hull."""
    n_chunk = len(chunk)
    x_l = torch.cat([leaf[0] for leaf in chunk], dim=0)
    x_u = torch.cat([leaf[1] for leaf in chunk], dim=0)
    y = labels.repeat(n_chunk, *([1] * (labels.dim() - 1)))

    grads_l, grads_u = bounded_model.bound_backward_combined(
        x_l,
        x_u,
        y,
        loss,
        label_k_poison=label_k_poison,
        label_epsilon=label_epsilon,
        poison_target_idx=poison_target_idx,
    )
    # unstack the leaf-major batch dimension ([n_chunk * B, ...] -> [n_chunk, B, ...]) and hull over
    # the leaf axis, leaving per-sample bounds.
    grads_l = [g.view(n_chunk, batchsize, *g.shape[1:]).amin(dim=0) for g in grads_l]
    grads_u = [g.view(n_chunk, batchsize, *g.shape[1:]).amax(dim=0) for g in grads_u]

    if hull_l is None or hull_u is None:
        return grads_l, grads_u
    hull_l = [torch.minimum(a, b) for a, b in zip(hull_l, grads_l)]
    hull_u = [torch.maximum(a, b) for a, b in zip(hull_u, grads_u)]
    return hull_l, hull_u


def refined_bound_backward_combined(
    bounded_model: BoundedModel,
    batch: torch.Tensor,
    labels: torch.Tensor,
    loss: BoundedLoss,
    *,
    epsilon: float,
    dims: list[int],
    n_splits: int,
    leaf_chunk: int,
    label_k_poison: int = 0,
    label_epsilon: float = 0.0,
    poison_target_idx: int = -1,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Per-sample parameter-gradient bounds under the feature-poisoning adversary, refined over a grid
    of input sub-boxes.

    Drop-in tightening of
    ``bounded_model.bound_backward_combined(batch - epsilon, batch + epsilon, labels, loss, ...)``:
    the return shape is identical (per-sample bounds in the library's flat parameter order) and the
    bounds are never wider. Reduces to exactly that call when ``n_splits <= 1`` or ``dims`` selects
    no coordinate, so a disabled refinement is bit-identical to the shipped path.

    Leaves are propagated in groups of ``leaf_chunk``, stacked into the batch dimension so each
    group is one batched bounding call, then hulled at the per-sample gradient. Chunking does not
    change the result: ``amin``/``amax`` are associative and exact.
    """
    batchsize = batch.size(0)
    chunk_size = max(1, leaf_chunk)
    hull_l: list[torch.Tensor] | None = None
    hull_u: list[torch.Tensor] | None = None
    chunk: list[tuple[torch.Tensor, torch.Tensor]] = []

    for leaf in iter_leaf_boxes(batch, epsilon, dims, n_splits):
        chunk.append(leaf)
        if len(chunk) < chunk_size:
            continue
        hull_l, hull_u = _propagate_chunk(
            bounded_model, chunk, labels, loss, batchsize,
            label_k_poison, label_epsilon, poison_target_idx, hull_l, hull_u,
        )
        chunk = []

    if chunk:
        hull_l, hull_u = _propagate_chunk(
            bounded_model, chunk, labels, loss, batchsize,
            label_k_poison, label_epsilon, poison_target_idx, hull_l, hull_u,
        )

    assert hull_l is not None and hull_u is not None
    return hull_l, hull_u


def resolve_leaf_chunk(cfg: InputRefinementConfig, fragsize: int, n_rows: int) -> int:
    """
    Number of leaves to stack into one bounding call.
    """
    if cfg.leaf_chunk is not None:
        return max(1, cfg.leaf_chunk)
    return max(1, fragsize // max(1, n_rows))


def validate_input_refinement(bounded_model: BoundedModel, config: AGTConfig) -> list[int]:
    """
    Check the assumptions the shared-grid argument needs against the concrete model and refuse
    configurations that would be unsound or unexpectedly expensive.

    Returns:
        list[int]: the split dimensions selected at entry, for logging.
    """
    cfg = config.input_refinement
    assert cfg is not None, "validate_input_refinement called without an input_refinement config"

    # A1 -- per-sample separability: no batch-coupled layer, loss reduction "none".
    transform_modules = list(getattr(bounded_model.transform, "modules", []) or [])
    for module in transform_modules + list(bounded_model.modules):
        if isinstance(module, _BATCH_COUPLED_LAYERS):
            raise ValueError(
                f"input_refinement is unsound with the batch-coupled layer {type(module).__name__}: "
                f"the per-sample gradient then depends on the rest of the batch, breaking assumption "
                f"A1 (scripts/SHARED_GRID_PARTITION.md section 7)."
            )
    if getattr(config.get_bounded_loss_fn(), "reduction", "none") != "none":
        raise ValueError("input_refinement requires a loss with reduction='none' (assumption A1).")

    # A2 -- clamp clipping commutes with the hull; norm clipping stays sound but strictly looser.
    if config.clip_method == "norm":
        LOGGER.warning(
            "input_refinement with clip_method='norm' is sound but strictly looser than clipping "
            "inside each leaf (assumption A2, scripts/SHARED_GRID_PARTITION.md section 7)."
        )

    dims = select_split_dims(bounded_model, config.epsilon, cfg)
    n_leaves = config.input_refinement.n_splits ** len(dims)
    if n_leaves > cfg.max_leaves:
        raise ValueError(
            f"input_refinement would propagate {n_leaves} leaves per fragment "
            f"(n_splits={cfg.n_splits} raised to {len(dims)} split dimensions), above "
            f"max_leaves={cfg.max_leaves}. Lower n_splits or n_dims, or raise max_leaves."
        )
    return dims
