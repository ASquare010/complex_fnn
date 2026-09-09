"""Independently check containment, strict separation and bounded value mixing."""

import math
from dataclasses import replace

import torch

from src.adaptive_cross_group_ffn import AdaptiveCoupledFFN
from src.core.config import ModelConfig
from src.core.transformer import Transformer


def test_zero_coupling_preserves_whole_model_and_parameter_budget():
    cfg = ModelConfig(variant="structured_swiglu_adaptive", groups=4)
    assert cfg.unique_ffn_parameters == 296960
    assert cfg.total_parameters == 1674944
    adaptive = Transformer(cfg)
    original = Transformer(replace(cfg, variant="structured_swiglu"))
    tokens = torch.randint(cfg.vocab_size, (1, 8))
    torch.testing.assert_close(adaptive(tokens), original(tokens), atol=0, rtol=0)
    for name, p in original.named_parameters():
        torch.testing.assert_close(p, dict(adaptive.named_parameters())[name], atol=0, rtol=0)


def test_pure_cross_product_construction():
    f = AdaptiveCoupledFFN(4, 16, 4).double()
    with torch.no_grad():
        for p in f.parameters():
            p.zero_()
        f.up.weight[0, 0, 0] = 1
        f.up.weight[0, 1, 0] = -1
        f.gate.weight[3, :2, 0] = 1
        f.mix_theta[:2] = math.atanh(2 / 3)
        f.down.weight[0, 0, 0] = 2 * 1.25**0.5
        f.down.weight[1, 0, 0] = -2 * 1.25**0.5
    x = torch.randn(40, 4, dtype=torch.float64)
    torch.testing.assert_close(f(x).sum(-1), x[:, 0] * x[:, 3], atol=1e-12, rtol=1e-12)


def test_variable_coefficient_mixer_singular_bounds():
    p = torch.eye(16, dtype=torch.float64).reshape(16, 4, 4).roll(1, dims=1).reshape(16, 16).T
    raw = torch.linspace(-12, 12, 16, dtype=torch.float64)
    mix = 0.75 * raw.tanh()
    matrix = torch.diag(torch.rsqrt(1 + mix.square())) @ (torch.eye(16) + torch.diag(mix) @ p)
    sigma = torch.linalg.svdvals(matrix)
    assert sigma.min() >= 0.2 - 1e-14
    assert sigma.max() <= 1.75 + 1e-14


def test_coupling_has_trainable_gradients_at_zero():
    f = AdaptiveCoupledFFN(4, 8, 2).double()
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (x,))
    gradient = torch.autograd.grad(f(x).square().sum(), f.mix_theta)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    # Finite differences check the new learned coefficients, not only input gradients.
    from torch.func import functional_call

    parameters = dict(f.named_parameters())
    assert torch.autograd.gradcheck(
        lambda theta: functional_call(f, {**parameters, "mix_theta": theta}, (x,)),
        (f.mix_theta,),
    )
