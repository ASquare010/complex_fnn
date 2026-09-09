"""Independent formula, gradient, initialization and recomputation checks."""

from dataclasses import replace

import pytest
import torch
from torch.func import functional_call
from torch.nn import functional as F

from src.core.benchmark import forward_flops
from src.core.config import ModelConfig, TrainConfig
from src.core.optimization import initialize_dense_width, parameter_groups
from src.core.transformer import Transformer
from src.rational_blockshuffle_ffn import (
    RationalBlockShuffleFFN,
    RationalResidualActivation,
)


@pytest.mark.parametrize("cls", [RationalResidualActivation])
def test_independent_formula_and_parameter_gradcheck(cls):
    torch.manual_seed(123)
    curve = cls(4, 2).double()
    with torch.no_grad():
        for p in curve.parameters():
            p.uniform_(-0.3, 0.3)
    z = torch.tensor([[-4.0, 0.0, 0.7, 6.0]], dtype=torch.float64, requires_grad=True)
    grouped = z.reshape(1, 2, 2)
    expected = torch.zeros_like(grouped)
    for g in range(2):
        a = curve.theta_coefficients[g].tanh()
        beta = 1 + 0.75 * curve.theta_denominator[g].tanh()
        q = grouped[:, g]
        expected[:, g] = 0.25 * sum(a[j] * q**j for j in range(4)) / (1 + beta * q**2)
    expected = F.silu(z) + expected.reshape_as(z)
    actual = curve(z)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    params = dict(curve.named_parameters())
    expected_grads = torch.autograd.grad(expected.sum(), (z, *params.values()), retain_graph=True)
    actual_grads = torch.autograd.grad(actual.sum(), (z, *params.values()))
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)

    def call(x, *values):
        return functional_call(curve, dict(zip(params, values)), (x,))

    assert torch.autograd.gradcheck(call, (z, *params.values()), fast_mode=True)


@pytest.mark.parametrize("family,extra", [("rational", 40)])
@pytest.mark.parametrize("base,hidden", [("blockshuffle_swiglu", 2048)])
def test_large_actual_budget_and_matrix_work(family, extra, base, hidden):
    config = ModelConfig(variant=base, width=384, layers=8, hidden=hidden)
    adaptive = replace(config, variant=f"{base}_{family}")
    # One layer is enough to inspect its real allocation without a full checkpoint.
    from src.core.transformer import make_ffn

    ffn = make_ffn(adaptive)
    assert sum(p.numel() for p in ffn.parameters()) == config.ffn_parameters + extra
    assert adaptive.unique_ffn_parameters == config.unique_ffn_parameters + 8 * extra
    assert 1 - adaptive.unique_ffn_parameters / 9437184 >= 0.70
    assert forward_flops(adaptive) == forward_flops(config)


@pytest.mark.parametrize("family", ["rational"])
@pytest.mark.parametrize("base", ["blockshuffle_swiglu"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_initial_common_parameters_forward_backward_and_optimizer(family, base, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = ModelConfig(
        variant=base, width=24, layers=2, heads=3, hidden=48, groups=3, vocab_size=32, context=8
    )
    tc = TrainConfig(ffn_lr_mode="fan_in", ffn_decay_mode="parameter")
    baseline = Transformer(config, 17).to(device)
    adaptive = Transformer(
        replace(config, variant=f"{base}_{family}"),
        17,
    ).to(device)
    for model in (baseline, adaptive):
        initialize_dense_width(model, tc)
        if base.startswith("blockshuffle"):
            for block in model.blocks:
                block.ffn.recompute_gate = True
                block.ffn.gate_recompute_method = "checkpoint"
    bp, ap = dict(baseline.named_parameters()), dict(adaptive.named_parameters())
    assert all(torch.equal(p, ap[n]) for n, p in bp.items())
    assert all(torch.count_nonzero(p) == 0 for n, p in ap.items() if n not in bp)
    x = torch.arange(16, device=device).reshape(2, 8)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        a, b = baseline(x), adaptive(x)
        la, lb = baseline.loss(x, x + 1), adaptive.loss(x, x + 1)
    assert torch.equal(a, b) and torch.equal(la, lb)
    la.backward()
    lb.backward()
    assert all(torch.equal(p.grad, ap[n].grad) for n, p in bp.items())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ap.values())
    original_groups = {
        id(p): (g["weight_decay"], g["lr_scale"])
        for g in parameter_groups(baseline, tc)
        for p in g["params"]
    }
    adaptive_groups = {
        id(p): (g["weight_decay"], g["lr_scale"])
        for g in parameter_groups(adaptive, tc)
        for p in g["params"]
    }
    assert all(original_groups[id(p)] == adaptive_groups[id(ap[n])] for n, p in bp.items())
    assert all(adaptive_groups[id(p)] == (0.0, 1.0) for n, p in ap.items() if n not in bp)


@pytest.mark.parametrize("family", ["rational"])
def test_recomputation_preserves_all_nonzero_activation_gradients(family):
    torch.manual_seed(88)
    ffn = RationalBlockShuffleFFN(12, 24, 3).double()
    with torch.no_grad():
        for p in ffn.curve.parameters():
            p.uniform_(-0.5, 0.5)
    z = torch.randn(2, 3, 12, dtype=torch.float64, requires_grad=True)
    outputs, gradients = [], []
    for enabled in (False, True):
        ffn.recompute_gate = enabled
        value = ffn(z)
        outputs.append(value)
        gradients.append(torch.autograd.grad(value.square().sum(), (z, *ffn.parameters())))
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("cls,bound", [(RationalResidualActivation, 3.29)])
def test_bounded_shapes_derivatives_and_nontrivial_learning(cls, bound):
    torch.manual_seed(95)
    curve = cls(8, 2).double()
    # At zero controls the coordinate gradients vanish; controls must still learn.
    x = torch.linspace(-4, 4, 512, dtype=torch.float64).reshape(64, 8)
    target = F.silu(x) + 0.08 * torch.sin(x)
    opt = torch.optim.Adam(curve.parameters(), lr=0.02)
    initial = (curve(x) - target).square().mean().item()
    for _ in range(60):
        opt.zero_grad()
        loss = (curve(x) - target).square().mean()
        loss.backward()
        opt.step()
    assert (curve(x) - target).square().mean().item() < initial * 0.6
    with torch.no_grad():
        for p in curve.parameters():
            p.uniform_(-8, 8)
    z = torch.linspace(-100, 100, 8000, dtype=torch.float64).reshape(-1, 8).requires_grad_(True)
    value = curve(z)
    derivative = torch.autograd.grad(value.sum(), z)[0]
    assert derivative.abs().max() < bound
    large = torch.tensor([-1e20, 1e20] * 4).reshape(1, 8)
    assert torch.isfinite(curve.float()(large)).all()


def test_reject_invalid_dimensions_and_unsupported_native_recomputation():
    with pytest.raises(ValueError):
        RationalResidualActivation(7, 2)
    with pytest.raises(ValueError):
        ModelConfig(variant="blockshuffle_swiglu_rational", hidden=17).validate()
    model = RationalBlockShuffleFFN(12, 24, 3)
    model.recompute_gate = True
    model.gate_recompute_method = "native"
    with pytest.raises(ValueError, match="checkpoint"):
        model(torch.ones(1, 12))
