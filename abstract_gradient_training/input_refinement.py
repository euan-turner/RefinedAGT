"""
Input-ball refinement for poison-certified training.

Partitions the feature-poisoning l_inf ball into a grid of sub-boxes, propagates each leaf through
the ordinary bounding pass, and hulls the per-sample parameter-gradient bounds over the leaves. The
result is a sound, and never wider, replacement for the single-box bound. When ``n_splits <= 1``, refinement
reduces to the default bounding method.

The refinement is exponential in the number of split dimensions (``n_splits ** len(dims)`` bounding
passes per fragment, or the product of the per-dimension cuts when the primary and secondary tiers of
``InputRefinementConfig`` cut different counts) and flat in the batch size. It calls only ``bound_backward_combined``, so it
composes with every ``BoundedModel`` bounding method; but is impractical when the per-leaf work
is heavy, such as the MIP-bounded model. The leaves can be sharded across the ranks of a
``torch.distributed`` process group (``InputRefinementConfig.shard_leaves``).
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Iterator, Sequence

import torch
import torch.distributed as dist

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
        list[int]: coordinate indices into the flattened input, at most ``cfg.n_dims +
            cfg.secondary_n_dims`` of them. The first ``cfg.n_dims`` are the primary tier (cut into
            ``cfg.n_splits``); ``split_counts`` pairs each with its cut count.
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
        return torch.argsort(scores, descending=True)[: len(cfg.dim_splits)].tolist()

    # "widest" and "first": split the leading coordinates. iter_leaf_boxes clamps to the true
    # flattened input dimension, so requesting more than the model has is harmless.
    return list(range(len(cfg.dim_splits)))


def split_counts(cfg: InputRefinementConfig, dims: list[int]) -> list[int]:
    """The cut count for each coordinate ``select_split_dims`` returned, in the same order."""
    return cfg.dim_splits[: len(dims)]


def iter_leaf_boxes(
    batch: torch.Tensor, epsilon: float, dims: list[int], n_splits: int | Sequence[int]
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Yield the ``(lower, upper)`` corners of each leaf of the partitioned l_inf ball, one leaf at a
    time.

    Each coordinate in ``dims`` (an index into the flattened input) is cut into ``n_splits`` equal
    intervals -- one count for every coordinate, or one per coordinate of ``dims`` -- and every other
    coordinate keeps its full ``[x - epsilon, x + epsilon]`` range. There are as many leaves as the
    product of the cuts, a single grid shared by every sample in ``batch``.
    Leaves are built lazily from ``batch`` rather than materialised together,
    so peak memory is one leaf's worth of corners regardless of the count.

    When no coordinate is cut (every count ``<= 1``, or ``dims`` selects nothing), yields a single leaf
    equal to ``(batch - epsilon, batch + epsilon)``.

    Yields:
        tuple[torch.Tensor, torch.Tensor]: ``(x_l, x_u)`` shaped like ``batch``, the corners of one
            leaf for the whole batch.
    """
    lower = batch - epsilon
    upper = batch + epsilon
    cut_dims, cut_splits = cut_grid(batch, dims, n_splits)

    if not cut_dims:
        yield lower, upper
        return

    lower_flat = lower.flatten(1)
    upper_flat = upper.flatten(1)
    widths = [2.0 * epsilon / s for s in cut_splits]
    for offsets in itertools.product(*(range(s) for s in cut_splits)):
        leaf_l = lower_flat.clone()
        leaf_u = upper_flat.clone()
        for d, offset, width in zip(cut_dims, offsets, widths):
            leaf_l[:, d] = lower_flat[:, d] + offset * width
            leaf_u[:, d] = lower_flat[:, d] + (offset + 1) * width
        yield leaf_l.view_as(batch), leaf_u.view_as(batch)


def cut_grid(
    batch: torch.Tensor, dims: list[int], n_splits: int | Sequence[int]
) -> tuple[list[int], list[int]]:
    """
    The coordinates of ``dims`` that are actually cut, and the number of cuts along each.

    ``n_splits`` is one count for every coordinate of ``dims`` or a sequence with one count per
    coordinate. A coordinate is cut when its count is above 1 and it lies inside the flattened input;
    nothing is cut when ``batch`` has no feature axis.
    """
    splits = [n_splits] * len(dims) if isinstance(n_splits, int) else list(n_splits)
    if len(splits) != len(dims):
        raise ValueError(f"got {len(splits)} split counts for {len(dims)} split dimensions")
    if batch.dim() < 2:
        return [], []
    pairs = [(d, s) for d, s in zip(dims, splits) if s > 1 and d < batch[0].numel()]
    return [d for d, _ in pairs], [s for _, s in pairs]


def cut_dims_of(batch: torch.Tensor, dims: list[int], n_splits: int | Sequence[int]) -> list[int]:
    """The coordinates of ``dims`` that are actually cut; see ``cut_grid``."""
    return cut_grid(batch, dims, n_splits)[0]


def count_leaves(batch: torch.Tensor, dims: list[int], n_splits: int | Sequence[int]) -> int:
    """Number of leaves in the partition ``iter_leaf_boxes`` enumerates."""
    return math.prod(cut_grid(batch, dims, n_splits)[1])


def leaf_boxes(
    batch: torch.Tensor,
    epsilon: float,
    dims: list[int],
    n_splits: int | Sequence[int],
    leaf_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Corners of the leaves ``leaf_ids`` of the partitioned l_inf ball, for the whole batch, stacked
    leaf-major.

    Leaf ``i`` is the ``i``-th leaf ``iter_leaf_boxes`` yields, with identical corners: its offsets
    along the coordinates of ``cut_grid(batch, dims, n_splits)`` are the digits of ``i`` in the
    mixed radix given by their cut counts, most significant first. Unlike ``iter_leaf_boxes`` any
    subset of leaves can be built directly, which is what lets ranks take disjoint blocks of the
    partition.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(x_l, x_u)`` of shape ``[len(leaf_ids) * B, ...]``, the
            corners of leaf ``leaf_ids[0]`` for every sample first.
    """
    batchsize = batch.size(0)
    lower = (batch - epsilon).reshape(batchsize, -1).repeat(len(leaf_ids), 1)
    upper = (batch + epsilon).reshape(batchsize, -1).repeat(len(leaf_ids), 1)
    cut_dims, cut_splits = cut_grid(batch, dims, n_splits)

    if cut_dims:
        radix = torch.tensor(cut_splits, device=batch.device)
        places = torch.tensor([math.prod(cut_splits[p + 1 :]) for p in range(len(cut_splits))], device=batch.device)
        offsets = (leaf_ids.to(batch.device)[:, None] // places) % radix
        offsets = offsets.repeat_interleave(batchsize, dim=0).double()
        # scale in float64 then cast, so the corners round exactly as iter_leaf_boxes' scalar arithmetic does
        widths = torch.tensor([2.0 * epsilon / s for s in cut_splits], dtype=torch.float64, device=batch.device)
        base = lower[:, cut_dims]
        lower[:, cut_dims] = base + (offsets * widths).to(batch.dtype)
        upper[:, cut_dims] = base + ((offsets + 1) * widths).to(batch.dtype)

    shape = (len(leaf_ids) * batchsize, *batch.shape[1:])
    return lower.view(shape), upper.view(shape)


def leaf_block(n_leaves: int, rank: int, world_size: int) -> tuple[int, int]:
    """The contiguous ``[start, stop)`` block of leaf ids that ``rank`` of ``world_size`` propagates.
    Blocks differ in size by at most one and are empty when there are more ranks than leaves."""
    return rank * n_leaves // world_size, (rank + 1) * n_leaves // world_size


def _propagate_chunk(
    bounded_model: BoundedModel,
    x_l: torch.Tensor,
    x_u: torch.Tensor,
    n_chunk: int,
    labels: torch.Tensor,
    loss: BoundedLoss,
    batchsize: int,
    label_k_poison: int,
    label_epsilon: float,
    poison_target_idx: int,
    hull_l: list[torch.Tensor] | None,
    hull_u: list[torch.Tensor] | None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Propagate one group of ``n_chunk`` leaves, stacked leaf-major in ``x_l``/``x_u``, as a single
    batched call and fold it into the running hull."""
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
    n_splits: int | Sequence[int],
    leaf_chunk: int,
    label_k_poison: int = 0,
    label_epsilon: float = 0.0,
    poison_target_idx: int = -1,
    shard: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Per-sample parameter-gradient bounds under the feature-poisoning adversary, refined over a grid
    of input sub-boxes.

    Drop-in tightening of ``bounded_model.bound_backward_combined(batch - epsilon, batch + epsilon, labels, loss, ...)``:
    the return shape is identical (per-sample bounds in flat parameter order) and the
    bounds are never wider. ``n_splits`` is one cut count for every coordinate of ``dims`` or one per
    coordinate (see ``cut_grid``). Reduces to exactly that call when no coordinate is cut, so a
    disabled refinement is bit-identical to the shipped path.

    Leaves are propagated in groups of ``leaf_chunk``, stacked into the batch dimension so each
    group is one batched bounding call, then hulled at the per-sample gradient. Chunking does not
    change the result: ``amin``/``amax`` are associative and exact.

    With ``shard``, each rank of the default ``torch.distributed`` process group propagates only its
    ``leaf_block`` of the partition and the per-rank hulls are combined with a MIN/MAX all-reduce, so
    every rank returns the full hull. The caller must give every rank the same model, batch and
    ``dims`` (see ``broadcast_parameters``, ``broadcast_batch`` and ``broadcast_split_dims``).
    """
    batchsize = batch.size(0)
    chunk_size = max(1, leaf_chunk)
    n_leaves = count_leaves(batch, dims, n_splits)
    start, stop = 0, n_leaves
    if shard:
        start, stop = leaf_block(n_leaves, dist.get_rank(), dist.get_world_size())

    hull_l: list[torch.Tensor] | None = None
    hull_u: list[torch.Tensor] | None = None
    for chunk_start in range(start, stop, chunk_size):
        leaf_ids = torch.arange(chunk_start, min(chunk_start + chunk_size, stop))
        x_l, x_u = leaf_boxes(batch, epsilon, dims, n_splits, leaf_ids)
        hull_l, hull_u = _propagate_chunk(
            bounded_model, x_l, x_u, len(leaf_ids), labels, loss, batchsize,
            label_k_poison, label_epsilon, poison_target_idx, hull_l, hull_u,
        )

    if shard:
        return _all_reduce_hull(bounded_model, batchsize, hull_l, hull_u)
    assert hull_l is not None and hull_u is not None
    return hull_l, hull_u


def _all_reduce_hull(
    bounded_model: BoundedModel,
    batchsize: int,
    hull_l: list[torch.Tensor] | None,
    hull_u: list[torch.Tensor] | None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Combine per-rank hulls into the hull over every rank's leaves, on every rank. A rank that
    propagated no leaves contributes the identity of MIN (+inf) and MAX (-inf)."""
    shapes = [(batchsize, *p.shape) for p in bounded_model.param_n]
    if hull_l is None or hull_u is None:
        options = dict(dtype=bounded_model.dtype, device=bounded_model.device)
        hull_l = [torch.full(s, float("inf"), **options) for s in shapes]
        hull_u = [torch.full(s, float("-inf"), **options) for s in shapes]

    # one flat buffer per bound, so each fragment costs two collectives rather than two per parameter
    flat_l = torch.cat([g.reshape(-1) for g in hull_l])
    flat_u = torch.cat([g.reshape(-1) for g in hull_u])
    dist.all_reduce(flat_l, op=dist.ReduceOp.MIN)
    dist.all_reduce(flat_u, op=dist.ReduceOp.MAX)
    sizes = [g.numel() for g in hull_l]
    hull_l = [g.view(s) for g, s in zip(flat_l.split(sizes), shapes)]
    hull_u = [g.view(s) for g, s in zip(flat_u.split(sizes), shapes)]
    return hull_l, hull_u


def broadcast_parameters(bounded_model: BoundedModel) -> None:
    """
    Overwrite every rank's nominal parameters and parameter bounds with rank 0's, in place.

    The sharded hull mixes leaves bounded on different ranks, so it is only sound if every rank
    bounds them under the same parameter box. Broadcasting enforces that directly rather than relying
    on identical data order and bitwise-deterministic kernels across ranks.
    """
    for param in bounded_model.param_n + bounded_model.param_l + bounded_model.param_u:
        dist.broadcast(param, src=0)


def broadcast_batch(
    batch: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank 0's batch and labels, on ``device``, on every rank."""
    batch, labels = batch.to(device), labels.to(device)
    dist.broadcast(batch, src=0)
    dist.broadcast(labels, src=0)
    return batch, labels


def broadcast_split_dims(split_dims: list[int], device: torch.device) -> list[int]:
    """Rank 0's split dimensions, on every rank."""
    dims = torch.tensor(split_dims, dtype=torch.int64, device=device)
    dist.broadcast(dims, src=0)
    return dims.tolist()


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

    if cfg.shard_leaves and not (dist.is_available() and dist.is_initialized()):
        raise ValueError(
            "input_refinement.shard_leaves requires an initialised torch.distributed process group "
            "(e.g. launch with torchrun and call torch.distributed.init_process_group)."
        )

    dims = select_split_dims(bounded_model, config.epsilon, cfg)
    n_leaves = math.prod(split_counts(cfg, dims))
    if n_leaves > cfg.max_leaves:
        raise ValueError(
            f"input_refinement would propagate {n_leaves} leaves per fragment "
            f"(cuts {split_counts(cfg, dims)} over split dimensions {dims}), above "
            f"max_leaves={cfg.max_leaves}. Lower n_splits or n_dims, or raise max_leaves."
        )
    return dims
