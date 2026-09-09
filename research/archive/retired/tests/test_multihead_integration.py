"""Initialization, count and diagnostic contracts for registered multi-head FFNs."""

import math

import pytest
import torch

from src.core.benchmark import forward_flops
from src.core.config import ModelConfig
from src.core.diagnostics import inspect_layers
from src.core.transformer import Transformer
from src.multihead_ffn import ParallelMultiheadFFN, normalized_sigmoid


@pytest.mark.parametrize("variant", ["multihead_swiglu", "multihead_swiglu_calibrated"])
def test_registered_forward_gradients_and_diagnostics_preserve_state(variant):
    config = ModelConfig(
        variant=variant, width=24, layers=2, heads=3, context=8, vocab_size=32, hidden=24, groups=3
    )
    model = Transformer(config, 17)
    again = Transformer(config, 17)
    full = Transformer(
        ModelConfig(width=24, layers=2, heads=3, context=8, vocab_size=32, variant="swiglu"), 17
    )
    assert all(torch.equal(t, again.state_dict()[n]) for n, t in model.state_dict().items())
    assert all(
        torch.equal(p, full.state_dict()[n])
        for n, p in model.named_parameters()
        if ".ffn." not in n
    )
    x = torch.arange(16).reshape(2, 8)
    before = model(x).detach()
    diagnostics = inspect_layers(model, x, "cpu", "fp32")
    assert torch.equal(before, model(x))
    assert all(torch.equal(t, again.state_dict()[n]) for n, t in model.state_dict().items())
    for i in range(2):
        row = diagnostics[f"layer_{i}.ffn"]
        assert row["routing_finite"] and 0 < row["routing_weight_sum_mean"] <= 1
        assert 0 <= row["routing_normalized_entropy_mean"] <= math.log(2) + 1e-6
    model.loss(x, x + 1).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.count_nonzero(b.ffn.router.grad) > 0 for b in model.blocks)


def test_calibrated_mixer_norms_and_uniform_routing():
    module = ParallelMultiheadFFN(24, 3, 2, 24).double()
    module.initialize(17, "blocks.0.ffn", 0.25, True)
    x = torch.randn(11, 24, dtype=torch.float64)
    q = module.input_projection(x)
    torch.testing.assert_close(q.square().sum(-1), x.square().sum(-1), rtol=1e-12, atol=1e-12)
    y = module.output_projection(x)
    torch.testing.assert_close(
        y.square().sum(-1), x.square().sum(-1) * 0.25**2, rtol=1e-12, atol=1e-12
    )
    assert torch.count_nonzero(module.router) == 0
    logits = torch.einsum("mhd,hde->mhe", q.reshape(11, 3, 8), module.router)
    weights = normalized_sigmoid(logits, module.epsilon)
    torch.testing.assert_close(
        weights, torch.full_like(weights, 1 / (2 + 2 * module.epsilon)), rtol=1e-14, atol=1e-14
    )


def test_small_budget_counts_and_independent_attention_heads():
    config = ModelConfig(
        variant="multihead_swiglu", width=384, layers=8, heads=6, groups=48, vocab_size=4096
    )
    config.validate()
    assert (
        config.ffn_width == 24
        and config.unique_ffn_parameters == 2807808
        and config.total_parameters == 9105792
    )
    assert forward_flops(config)["ffn_matrix_forward_flops_per_token"] == 2 * 2807808
    with pytest.raises(ValueError, match="heads must divide"):
        ModelConfig(variant="multihead_swiglu", width=24, heads=3, groups=5).validate()
