"""
Tests for training_utils.propagate_clipping: the clipped interval contains the clipped value of
every point of the input interval, for parameter gradients of any rank (weights [batch, out, in] and
biases [batch, out]).

Key external dependencies: pytest, torch.
"""

import pytest
import torch

from abstract_gradient_training import training_utils


def _clip(x: torch.Tensor, gamma: float, method: str) -> torch.Tensor:
    if method == "clamp":
        return x.clamp(-gamma, gamma)
    norms = x.flatten(1).norm(2, dim=1)
    factor = (gamma / (norms + 1e-6)).clamp(max=1.0)
    return x * factor.view(-1, *([1] * (x.dim() - 1)))


@pytest.mark.parametrize("method", ["clamp", "norm"])
@pytest.mark.parametrize("gamma", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("shape", [(8, 5, 3), (8, 5), (8, 2, 3, 3)])
def test_clipped_interval_contains_clipped_points(method, gamma, shape):
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(shape, generator=generator, dtype=torch.float64)
    radius = 0.3 * torch.rand(shape, generator=generator, dtype=torch.float64)
    x_l, x_u = x - radius, x + radius

    (clip_l,), (clip_n,), (clip_u,) = training_utils.propagate_clipping(
        [x_l.clone()], [x.clone()], [x_u.clone()], gamma, method
    )
    assert clip_n.shape == x.shape
    assert torch.allclose(clip_n, _clip(x, gamma, method))

    for trial in range(200):
        u = torch.rand(shape, generator=generator, dtype=torch.float64)
        if trial % 2 == 0:
            u = (u > 0.5).double()  # a vertex of the interval
        clipped = _clip(x_l + u * (x_u - x_l), gamma, method)
        assert (clipped >= clip_l - 1e-9).all()
        assert (clipped <= clip_u + 1e-9).all()
