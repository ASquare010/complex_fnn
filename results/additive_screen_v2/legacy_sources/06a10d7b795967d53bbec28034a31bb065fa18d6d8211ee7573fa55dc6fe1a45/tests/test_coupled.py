"""Check structural claims independently of the training result."""

from dataclasses import replace

import pytest
import torch

from src.core.config import ModelConfig
from src.core.transformer import Transformer
from src.cross_group_ffn import CoupledStructuredFFN, mix_groups
from src.grouped_ffn import StructuredFFN


def test_group_mixing_singular_bound_and_zero_identity():
    eye = torch.eye(12, dtype=torch.float64)
    matrix = mix_groups(eye, 4)
    sigma = torch.linalg.svdvals(matrix)
    assert sigma.min() >= 0.5 / 1.25**0.5 - 1e-14
    assert sigma.max() <= 1.5 / 1.25**0.5 + 1e-14
    torch.testing.assert_close(mix_groups(eye, 4, 0), eye, atol=0, rtol=0)
    with pytest.raises(ValueError):
        mix_groups(eye, 4, 1)


def test_cross_group_mixed_derivative_is_restored():
    ordinary = StructuredFFN(4, 4, 4, gated=True).double()
    coupled = CoupledStructuredFFN(4, 4, 4).double()
    with torch.no_grad():
        for p in ordinary.parameters():
            p.fill_(1)
        coupled.load_state_dict(ordinary.state_dict())
    x = torch.zeros(4, dtype=torch.float64)
    first = torch.autograd.functional.hessian(lambda z: ordinary(z)[0], x)
    second = torch.autograd.functional.hessian(lambda z: coupled(z)[0], x)
    assert first[0, 3] == 0
    torch.testing.assert_close(second[0, 3], torch.tensor(0.25 / 1.25**0.5, dtype=x.dtype))


def test_coupled_input_gradcheck_and_identical_weights():
    f = CoupledStructuredFFN(4, 8, 2).double()
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (x,))
    cfg = ModelConfig(variant="structured_swiglu_coupled", groups=4)
    assert cfg.unique_ffn_parameters == 294912
    candidate = Transformer(cfg)
    control = Transformer(replace(cfg, variant="structured_swiglu"))
    for name, p in candidate.named_parameters():
        torch.testing.assert_close(p, dict(control.named_parameters())[name], atol=0, rtol=0)


def test_explicit_cross_group_polynomial_construction():
    # Independent algebraic witness, including the actual hidden shuffle.
    f = CoupledStructuredFFN(4, 16, 4).double()
    with torch.no_grad():
        for p in f.parameters():
            p.zero_()
        f.up.weight[0, 0, 0] = 1
        f.up.weight[0, 1, 0] = -1
        f.gate.weight[0, :2, 0] = 1
        f.down.weight[0, 0, 0] = 1.25**0.5
        f.down.weight[1, 0, 0] = -(1.25**0.5)
    x = torch.randn(50, 4, dtype=torch.float64)
    expected = x[:, 0] * (x[:, 0] + 0.5 * x[:, 3])
    torch.testing.assert_close(f(x).sum(-1), expected, atol=1e-12, rtol=1e-12)
