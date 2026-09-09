"""Tests of mathematical claims and controlled initial comparisons."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from src.bezier_ffn import QuadraticBezierActivation, QuadraticBezierFFN, quadratic_bezier
from src.core.config import ModelConfig, TrainConfig
from src.core.optimization import initialize_dense_width, parameter_groups
from src.core.transformer import Transformer


def test_bernstein_endpoints_and_local_silu_match():
    c1, c2 = torch.tensor(-1.0), torch.tensor(2.0)
    assert quadratic_bezier(torch.tensor(0.0), c1, c2) == 0
    assert quadratic_bezier(torch.tensor(1.0), c1, c2) == c2
    a = QuadraticBezierActivation(1).double()
    x = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    first = torch.autograd.grad(a(x).sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x)[0]
    torch.testing.assert_close(a(x), F.silu(x), atol=0, rtol=0)
    assert first.item() == 0.5 and second.item() == 0.5
    grid = torch.linspace(-20, 20, 501, dtype=torch.float64)[:, None]
    u = (grid / 2).tanh()
    torch.testing.assert_close(a(grid), u + u.square(), atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("residual", [False, True])
def test_input_and_control_first_and_second_derivatives(residual):
    a = QuadraticBezierActivation(4, 2, residual=residual).double()
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    theta = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)

    def function(x, theta):
        return torch.func.functional_call(a, {"theta": theta}, (x,))

    assert torch.autograd.gradcheck(function, (x, theta))
    assert torch.autograd.gradgradcheck(function, (x, theta))


@pytest.mark.parametrize("residual", [False, True])
def test_uniform_value_derivative_bounds_and_vanishing_tails(residual):
    a = QuadraticBezierActivation(4, 4, residual=residual).double()
    with torch.no_grad():
        a.theta.copy_(torch.tensor([[30.0, 30.0], [-30.0, 30.0], [30.0, -30.0], [-30.0, -30.0]]))
    x = (
        torch.linspace(-40, 40, 1001, dtype=torch.float64)[:, None]
        .expand(-1, 4)
        .clone()
        .requires_grad_()
    )
    q = a(x) - F.silu(x) if residual else a(x)
    slope = torch.autograd.grad(q.sum(), x)[0]
    assert q.abs().max() <= (0.5 if residual else 2.5) + 1e-12
    assert slope.abs().max() <= (0.5 if residual else 2.0) + 1e-12
    assert slope[[0, -1]].abs().max() < 1e-12


def test_branch_containment_cross_partial_and_spectral_bound():
    plain = QuadraticBezierFFN(3, 3, 3).double()
    mixed = QuadraticBezierFFN(3, 3, 3, mixed=True).double()
    z = torch.randn(2, 3, dtype=torch.float64)
    torch.testing.assert_close(plain.features(z), mixed.features(z), atol=0, rtol=0)
    with torch.no_grad():
        mixed.mix_theta.fill_(0.7)
    point = torch.zeros(3, dtype=torch.float64)
    hessian = torch.autograd.functional.hessian(lambda x: mixed.features(x)[0], point)
    expected = 0.5 * mixed.mix_theta[0].tanh() / 4
    torch.testing.assert_close(hessian[1, 2], expected)
    assert hessian[1, 2].abs() > 0
    with torch.no_grad():
        mixed.curve.theta.copy_(torch.tensor([[30.0, -30.0], [-30.0, 30.0], [30.0, 30.0]]))
        mixed.mix_theta.copy_(torch.tensor([30.0, -30.0, 30.0]))
    for point in (
        torch.zeros(3, dtype=torch.float64),
        torch.tensor([-1.0, 2.0, 4.0], dtype=torch.float64),
    ):
        jac = torch.autograd.functional.jacobian(mixed.features, point)
        assert torch.linalg.matrix_norm(jac, 2) <= 7
        assert mixed.features(point).abs().max() <= 5.625
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    eta = torch.randn(3, dtype=torch.float64, requires_grad=True)

    def function(x, eta):
        return torch.func.functional_call(mixed, {"mix_theta": eta}, (x,))

    assert torch.autograd.gradcheck(function, (x, eta))
    assert torch.autograd.gradgradcheck(function, (x, eta))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_residual_matches_calibrated_reference_and_common_gradients(device):
    cfg = ModelConfig(
        variant="swiglu_narrow", hidden=16, width=24, heads=3, layers=2, vocab_size=32, context=8
    )
    tc = TrainConfig(
        ffn_width_init_mode="fan_in", ffn_width_lr_mode="fan_in", ffn_decay_mode="product"
    )
    reference = Transformer(cfg).to(device)
    candidate = Transformer(replace(cfg, variant="swiglu_quadratic")).to(device)
    for model in (reference, candidate):
        initialize_dense_width(model, tc)
    x = torch.randint(32, (2, 8), device=device)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        a, b = reference(x), candidate(x)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    a.float().square().sum().backward()
    b.float().square().sum().backward()
    named = dict(candidate.named_parameters())
    for name, p in reference.named_parameters():
        torch.testing.assert_close(p.grad, named[name].grad, atol=0, rtol=0)
    for block in candidate.blocks:
        assert torch.isfinite(block.ffn.curve.theta.grad).all()
        assert block.ffn.curve.theta.grad.abs().sum() > 0
    lookup = {id(p): g for g in parameter_groups(candidate, tc) for p in g["params"]}
    for block in candidate.blocks:
        assert lookup[id(block.ffn.curve.theta)]["weight_decay"] == 0
        assert lookup[id(block.ffn.down.weight)]["lr_scale"] == 4


def test_screen_counts():
    for variant, hidden, groups, expected in (
        ("quadratic_bezier", 228, 3, 350232),
        ("quadratic_bezier_mixed", 228, 3, 350244),
        ("swiglu_quadratic", 152, 1, 350216),
        ("swiglu_quadratic", 152, 8, 350272),
    ):
        cfg = ModelConfig(variant=variant, hidden=hidden, groups=groups)
        m = Transformer(cfg)
        assert cfg.unique_ffn_parameters == expected
        assert sum(p.numel() for b in m.blocks for p in b.ffn.parameters()) == expected
        assert 1 - expected / 1179648 > 0.7
