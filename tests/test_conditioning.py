"""Numerical checks of the scoped offline spectral intervention."""

import pytest
import torch

from src.core.conditioning import floor_square_blocks, matched_random


@pytest.mark.parametrize("alpha", [0.01, 0.1, 1.0])
def test_floor_preserves_norm_and_bounds_global_block_condition(alpha):
    g = torch.Generator().manual_seed(44)
    w = torch.randn(3, 4, 4, generator=g, dtype=torch.float64)
    w[0] *= 0.001
    actual = floor_square_blocks(w, alpha)
    s = torch.linalg.svdvals(actual)
    torch.testing.assert_close(actual.norm(), w.norm(), rtol=1e-12, atol=1e-12)
    assert s.max() / s.min() <= (1 / alpha) * (1 + 1e-12)


def test_polar_product_spectrum_with_rectangular_factor_and_permutation():
    g = torch.Generator().manual_seed(81)
    w = torch.randn(2, 4, 4, generator=g, dtype=torch.float64)
    q = torch.block_diag(*floor_square_blocks(w, 1).unbind())
    r = torch.randn(8, 15, generator=g, dtype=torch.float64)
    order = torch.randperm(8, generator=g)
    actual = torch.linalg.svdvals(q @ r[order])
    scale = w.norm() / (8**0.5)
    torch.testing.assert_close(actual, scale * torch.linalg.svdvals(r), rtol=1e-12, atol=1e-12)


def test_reconstruction_and_matched_noise_preserve_stated_controls():
    w = torch.randn(2, 4, 4, generator=torch.Generator().manual_seed(9))
    torch.testing.assert_close(floor_square_blocks(w, 0), w, rtol=0, atol=1e-6)
    target = floor_square_blocks(w, 0.5)
    noisy = matched_random(w, target, 27)
    torch.testing.assert_close(
        (noisy - w).norm(dim=(-2, -1)), (target - w).norm(dim=(-2, -1)), rtol=1e-6, atol=1e-6
    )
    assert torch.equal(noisy, matched_random(w, target, 27))
