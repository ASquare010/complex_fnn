"""Independent numerical contracts for the unregistered overcomplete module."""

import pytest
import torch
from torch.nn import functional as F

from src.multihead_ffn.overcomplete import OvercompleteHeadwiseFFN


def dense_factors(x, module):
    first = torch.block_diag(*module.first.weight.unbind(0))
    second = torch.block_diag(*module.second.weight.unbind(0))
    d, m, g = first.shape[0], second.shape[0], module.groups
    middle_order = torch.arange(d).reshape(g, -1).T.flatten()
    output_order = torch.arange(m).reshape(m // g, g).T.flatten()
    return F.linear(F.linear(x, first)[..., middle_order], second)[..., output_order]


def test_overcomplete_matches_dense_loop_and_every_derivative():
    module = OvercompleteHeadwiseFFN(12, 16, 4, 4, 8).double()
    module.initialize(19, "test", 0.3)
    x = torch.randn(2, 3, 12, dtype=torch.float64, requires_grad=True)
    q = dense_factors(x, module.input_projection).reshape(2, 3, 4, 4)
    pieces = []
    for h in range(4):
        z = q[:, :, h]
        pieces.append((F.silu(z @ module.gate[h]) * (z @ module.value[h])) @ module.down[h])
    expected = dense_factors(torch.cat(pieces, -1), module.output_projection)
    actual = module(x)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    probe = torch.randn_like(actual)
    variables = (x, *module.parameters())
    a = torch.autograd.grad((actual * probe).sum(), variables)
    b = torch.autograd.grad((expected * probe).sum(), variables)
    for left, right in zip(a, b, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-12)
    assert torch.autograd.gradcheck(module, (x,), fast_mode=True)


def test_overcomplete_isometries_determinism_and_count():
    module = OvercompleteHeadwiseFFN(24, 32, 4, 4, 16).double()
    module.initialize(17, "test", 0.25)
    a = module.input_projection(torch.eye(24, dtype=torch.float64)).T
    b = module.output_projection(torch.eye(32, dtype=torch.float64)).T
    torch.testing.assert_close(a.T @ a, torch.eye(24, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        b @ b.T, 0.25**2 * torch.eye(24, dtype=torch.float64), atol=1e-12, rtol=1e-12
    )
    saved = {n: p.clone() for n, p in module.named_parameters()}
    module.initialize(17, "test", 0.25)
    assert all(torch.equal(p, saved[n]) for n, p in module.named_parameters())
    full = OvercompleteHeadwiseFFN(384, 512, 16, 8, 200)
    assert sum(p.numel() for p in full.parameters()) == full.parameter_count == 350208
    assert not hasattr(full, "router")


def test_overcomplete_rejects_invalid_geometry():
    for dims in (
        (12, 12, 4, 4, 8),
        (12, 16, 0, 4, 8),
        (12, 16, 4, 3, 8),
        (12, 16, 4, 4, 0),
        (12, 17, 4, 1, 8),
    ):
        with pytest.raises(ValueError):
            OvercompleteHeadwiseFFN(*dims)
    module = OvercompleteHeadwiseFFN(12, 16, 4, 4, 8)
    with pytest.raises(ValueError):
        module.initialize(17, "test", float("nan"))
    with pytest.raises(ValueError):
        module(torch.randn(3, 11))
