"""
Input-ball refinement for poison-certified training.

Partitions the feature-poisoning l_inf ball into a grid of sub-boxes ("leaves"), propagates each leaf
through the ordinary bounding pass, and hulls the per-sample parameter-gradient bounds over the leaves.
The result is a sound, and never wider, replacement for the single-box bound.

The cost is one bounding pass per leaf per fragment: the product of ``InputRefinementConfig.splits``,
exponential in the number of coordinates cut and flat in the batch size. The leaves can be sharded
across the ranks of a ``torch.distributed`` process group (``InputRefinementConfig.shard_leaves``).

Design doc: docs/input_refinement_soundness.md (the soundness argument, its assumptions A1 and A2,
and the floating-point limitations).

Key external dependencies: torch, torch.distributed (sharded refinement only).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import torch
import torch.distributed as dist

from abstract_gradient_training.bounded_losses import BoundedLoss
from abstract_gradient_training.bounded_models import BoundedModel
from abstract_gradient_training.configuration import AGTConfig, InputRefinementConfig

LOGGER = logging.getLogger(__name__)

# Layers whose per-sample output depends on the rest of the batch, which breaks assumption A1
_BATCH_COUPLED_LAYERS = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
)


# ======================================================================================
# The partition: which coordinates are cut, and the corners of each leaf
# ======================================================================================


def select_split_dims(bounded_model: BoundedModel, epsilon: float, cfg: InputRefinementConfig) -> list[int]:
    """
    Choose which flattened input coordinates to partition, one per entry of ``cfg.splits``.

    The ``"sensitivity"`` strategy ranks coordinate ``d`` by the first-layer pre-activation width it
    injects, ``2 * epsilon * sum_j |W_1[j, d]|``, read from the nominal weights, most sensitive first.
    ``"first"`` takes the leading coordinates. Any choice is sound; the strategy only affects how
    tight the refined bound is.

    Raises for ``"sensitivity"`` when the first-layer weight matrix is not the one the eps-ball sees
    (a fixed ``transform`` or a non-``Linear`` first layer), or has fewer inputs than ``cfg.splits``
    has entries.

    Returns:
        list[int]: ``len(cfg.splits)`` distinct coordinate indices into the flattened input;
            coordinate ``i`` is cut into ``cfg.splits[i]`` pieces.
    """
    if cfg.strategy == "first":
        return list(range(len(cfg.splits)))

    first = bounded_model.modules[0] if bounded_model.modules else None
    if bounded_model.transform is not None:
        raise ValueError(
            "input_refinement strategy='sensitivity' cannot see through a fixed transform: the "
            "eps-ball is on the raw input but the first trainable layer sees the transformed "
            "features. Use strategy='first'."
        )
    if not isinstance(first, torch.nn.Linear):
        raise ValueError(
            f"input_refinement strategy='sensitivity' needs a Linear first layer to score input "
            f"coordinates; got {type(first).__name__}. Use strategy='first'."
        )
    weight = first.weight.detach()
    if weight.size(1) < len(cfg.splits):
        raise ValueError(
            f"input_refinement.splits cuts {len(cfg.splits)} coordinates but the model has only "
            f"{weight.size(1)} inputs."
        )
    scores = 2 * epsilon * weight.abs().sum(dim=0)
    return torch.argsort(scores, descending=True)[: len(cfg.splits)].tolist()


def _check_grid(batch: torch.Tensor, dims: Sequence[int], splits: Sequence[int]) -> None:
    """Raise unless ``dims`` are distinct coordinates of the flattened input with one cut count each."""
    n_features = batch[0].numel()
    if len(splits) != len(dims):
        raise ValueError(f"got {len(splits)} split counts for {len(dims)} split dimensions")
    if len(set(dims)) != len(dims):
        raise ValueError(f"split dimensions must be distinct; got {list(dims)}")
    if any(not 0 <= d < n_features for d in dims):
        raise ValueError(f"split dimensions {list(dims)} are not all inside the {n_features} input features")
    if any(s < 1 for s in splits):
        raise ValueError(f"split counts must be at least 1; got {list(splits)}")


def leaf_boxes(
    batch: torch.Tensor,
    epsilon: float,
    dims: Sequence[int],
    splits: Sequence[int],
    leaf_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Corners of the leaves ``leaf_ids`` of the partitioned l_inf ball, for the whole batch, stacked
    leaf-major.

    Coordinate ``dims[i]`` of the flattened input is cut into ``splits[i]`` equal intervals and every
    other coordinate keeps its full ``[x - epsilon, x + epsilon]`` range, giving ``prod(splits)``
    leaves on a grid shared by every sample. Leaf ``i`` takes its offset along each cut coordinate
    from the digits of ``i`` in the mixed radix ``splits``, most significant first, so any subset of
    leaves can be built directly (which is what lets ranks take disjoint blocks of the partition).

    The union of all leaves is exactly the ball: the outermost corners are ``batch - epsilon`` and
    ``batch + epsilon`` as the unrefined bound computes them, and neighbouring leaves share their
    common corner bit for bit.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(x_l, x_u)`` of shape ``[len(leaf_ids) * B, ...]``, the
            corners of leaf ``leaf_ids[0]`` for every sample first.
    """
    _check_grid(batch, dims, splits)
    batchsize = batch.size(0)
    dims, splits = list(dims), list(splits)
    lower = (batch - epsilon).reshape(batchsize, -1).repeat(len(leaf_ids), 1)
    upper = (batch + epsilon).reshape(batchsize, -1).repeat(len(leaf_ids), 1)

    if dims:
        radix = torch.tensor(splits, device=batch.device)
        places = torch.tensor([math.prod(splits[p + 1 :]) for p in range(len(splits))], device=batch.device)
        offsets = (leaf_ids.to(batch.device)[:, None] // places) % radix
        offsets = offsets.repeat_interleave(batchsize, dim=0)
        # interior corners are scaled in float64 then cast, so a float32 grid is as even as it can be
        widths = torch.tensor([2.0 * epsilon / s for s in splits], dtype=torch.float64, device=batch.device)
        ball_l, ball_u = lower[:, dims], upper[:, dims]
        cut_l = torch.minimum(ball_l + (offsets * widths).to(batch.dtype), ball_u)
        cut_u = torch.minimum(ball_l + ((offsets + 1) * widths).to(batch.dtype), ball_u)
        # ball_l + 2 * epsilon can round below ball_u: pin the top corner so that no sliver of the
        # ball is left uncovered
        lower[:, dims] = cut_l
        upper[:, dims] = torch.where(offsets + 1 == radix, ball_u, cut_u)

    shape = (len(leaf_ids) * batchsize, *batch.shape[1:])
    return lower.view(shape), upper.view(shape)


def leaf_block(n_leaves: int, rank: int, world_size: int) -> tuple[int, int]:
    """The contiguous ``[start, stop)`` block of leaf ids that ``rank`` of ``world_size`` propagates.
    Blocks differ in size by at most one and are empty when there are more ranks than leaves."""
    return rank * n_leaves // world_size, (rank + 1) * n_leaves // world_size


# ======================================================================================
# The refined bound
# ======================================================================================


def refined_bound_backward_combined(
    bounded_model: BoundedModel,
    batch: torch.Tensor,
    labels: torch.Tensor,
    loss: BoundedLoss,
    *,
    epsilon: float,
    dims: Sequence[int],
    splits: Sequence[int],
    leaf_chunk: int,
    label_k_poison: int = 0,
    label_epsilon: float = 0.0,
    poison_target_idx: int = -1,
    shard: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Per-sample parameter-gradient bounds under the feature-poisoning adversary, refined over a grid
    of input sub-boxes (see ``leaf_boxes`` for the grid ``dims``/``splits`` describe).

    Drop-in tightening of ``bounded_model.bound_backward_combined(batch - epsilon, batch + epsilon,
    labels, loss, ...)``: the return shape is identical (per-sample bounds in flat parameter order)
    and the bounds are never wider. With nothing cut it is exactly that call.

    Leaves are propagated in groups of ``leaf_chunk``, stacked into the batch dimension so each
    group is one batched bounding call, then hulled at the per-sample gradient. The grouping does
    not change the result.

    With ``shard``, each rank of the default ``torch.distributed`` process group propagates only its
    ``leaf_block`` of the partition and the per-rank hulls are combined with a MIN/MAX all-reduce, so
    every rank returns the full hull. The caller must give every rank the same model, batch and
    ``dims`` (see ``broadcast_parameters``, ``broadcast_batch`` and ``broadcast_split_dims``).
    """
    batchsize = batch.size(0)
    chunk_size = max(1, leaf_chunk)
    n_leaves = math.prod(splits)
    start, stop = 0, n_leaves
    if shard:
        start, stop = leaf_block(n_leaves, dist.get_rank(), dist.get_world_size())

    hull_l: list[torch.Tensor] | None = None
    hull_u: list[torch.Tensor] | None = None
    for chunk_start in range(start, stop, chunk_size):
        leaf_ids = torch.arange(chunk_start, min(chunk_start + chunk_size, stop))
        n_chunk = len(leaf_ids)
        x_l, x_u = leaf_boxes(batch, epsilon, dims, splits, leaf_ids)
        grads_l, grads_u = bounded_model.bound_backward_combined(
            x_l,
            x_u,
            labels.repeat(n_chunk, *([1] * (labels.dim() - 1))),
            loss,
            label_k_poison=label_k_poison,
            label_epsilon=label_epsilon,
            poison_target_idx=poison_target_idx,
        )
        # unstack the leaf-major batch dimension ([n_chunk * B, ...] -> [n_chunk, B, ...]) and hull
        # over the leaf axis, leaving per-sample bounds
        grads_l = [g.view(n_chunk, batchsize, *g.shape[1:]).amin(dim=0) for g in grads_l]
        grads_u = [g.view(n_chunk, batchsize, *g.shape[1:]).amax(dim=0) for g in grads_u]
        if hull_l is None or hull_u is None:
            hull_l, hull_u = grads_l, grads_u
        else:
            hull_l = [torch.minimum(a, b) for a, b in zip(hull_l, grads_l)]
            hull_u = [torch.maximum(a, b) for a, b in zip(hull_u, grads_u)]

    if shard:
        return _all_reduce_hull(bounded_model, batchsize, hull_l, hull_u)
    assert hull_l is not None and hull_u is not None
    return hull_l, hull_u


def resolve_leaf_chunk(cfg: InputRefinementConfig, fragsize: int, n_rows: int) -> int:
    """Number of leaves to stack into one bounding call: ``cfg.leaf_chunk`` when set, else as many
    as keep the stacked batch within ``fragsize`` rows."""
    if cfg.leaf_chunk is not None:
        return cfg.leaf_chunk
    return max(1, fragsize // max(1, n_rows))


def validate_input_refinement(bounded_model: BoundedModel, config: AGTConfig) -> list[int]:
    """
    Check the assumptions of the soundness argument (docs/input_refinement_soundness.md) against the
    concrete model, and refuse configurations that would be unsound or unexpectedly expensive.

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
                f"A1 (docs/input_refinement_soundness.md)."
            )
    if getattr(config.get_bounded_loss_fn(), "reduction", "none") != "none":
        raise ValueError("input_refinement requires a loss with reduction='none' (assumption A1).")

    # A2 -- clamp clipping commutes with the hull; norm clipping stays sound but strictly looser.
    if config.clip_method == "norm":
        LOGGER.warning(
            "input_refinement with clip_method='norm' is sound but strictly looser than clipping "
            "inside each leaf (assumption A2, docs/input_refinement_soundness.md)."
        )

    if cfg.shard_leaves and not (dist.is_available() and dist.is_initialized()):
        raise ValueError(
            "input_refinement.shard_leaves requires an initialised torch.distributed process group "
            "(e.g. launch with torchrun and call torch.distributed.init_process_group)."
        )

    if cfg.n_leaves > cfg.max_leaves:
        raise ValueError(
            f"input_refinement would propagate {cfg.n_leaves} leaves per fragment (splits "
            f"{cfg.splits}), above max_leaves={cfg.max_leaves}. Cut fewer coordinates, cut them into "
            f"fewer pieces, or raise max_leaves."
        )
    return select_split_dims(bounded_model, config.epsilon, cfg)


# ======================================================================================
# Sharding the leaves over a torch.distributed process group
# ======================================================================================


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
