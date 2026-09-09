"""Test exact equivalences and analytical stability limits, not just shapes."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from src.bezier_ffn import BezierActivation
from src.core.config import ModelConfig
from src.core.transformer import Transformer


def test_initial_equivalence_to_narrow_transformer():
    cfg = ModelConfig(variant="gelu_narrow", vocab_size=32, width=24, heads=3, layers=2, context=8)
    baseline = Transformer(cfg)
    candidate = Transformer(replace(cfg, variant="bezier_grouped"))
    x = torch.randint(32, (2, 8))
    torch.testing.assert_close(baseline(x), candidate(x), atol=0, rtol=0)


def test_gradcheck_input_and_control_parameters():
    activation = BezierActivation(8, 2).double()
    x = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    theta = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)

    def function(x, theta):
        return torch.func.functional_call(activation, {"theta": theta}, (x,))

    assert torch.autograd.gradcheck(function, (x, theta))


def test_live_control_gradients_at_zero():
    activation = BezierActivation(8, 2)
    activation(torch.ones(3, 8)).sum().backward()
    assert (activation.theta.grad.abs() > 0).all()


def test_value_and_derivative_bounds():
    activation = BezierActivation(8, 2).double()
    with torch.no_grad():
        activation.theta.copy_(torch.tensor([[20.0, -20.0], [-20.0, 20.0]]))
    x = (
        torch.linspace(-30, 30, 1001, dtype=torch.float64)[:, None]
        .expand(-1, 8)
        .clone()
        .requires_grad_()
    )
    residual = activation(x) - F.gelu(x)
    (derivative,) = torch.autograd.grad(residual.sum(), x)
    assert residual.abs().max() <= 0.375 + 1e-12
    assert derivative.abs().max() <= 0.75 + 1e-12


def test_static_bank_collapses():
    generator = torch.Generator().manual_seed(17)
    t = torch.linspace(0, 1, 201, dtype=torch.float64)
    basis = torch.stack(((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3), dim=1)
    controls = torch.randn(16, 4, generator=generator, dtype=torch.float64)
    weights = torch.randn(7, 16, generator=generator, dtype=torch.float64).softmax(-1)
    bank = basis @ controls.T @ weights.T
    collapsed = basis @ (weights @ controls).T
    torch.testing.assert_close(bank, collapsed, atol=1e-14, rtol=1e-14)


def test_bottleneck_invariance():
    a = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]], dtype=torch.float64)
    b = torch.randn(4, 2, dtype=torch.float64)
    v = torch.tensor([-2.0, 1.0, -1.0], dtype=torch.float64)
    x = torch.randn(3, dtype=torch.float64)
    torch.testing.assert_close(b @ torch.sin(a @ x), b @ torch.sin(a @ (x + v)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_bf16_extreme_values_and_backward():
    activation = BezierActivation(8, 2).cuda()
    x = torch.tensor(
        [-1e4, -100, -10, -1, 1, 10, 100, 1e4],
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = activation(x)
    y.float().sum().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    assert torch.isfinite(activation.theta.grad).all()
