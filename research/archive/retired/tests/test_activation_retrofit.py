"""Trained-weight retrofits preserve the function and leave the source untouched."""

import pytest
import torch

from src.core.activation_retrofit import add_learnable_activations
from src.core.config import ModelConfig, TrainConfig
from src.core.optimization import initialize_dense_width
from src.core.transformer import Transformer


@pytest.mark.parametrize("base", ["swiglu_narrow", "blockshuffle_swiglu"])
@pytest.mark.parametrize("family", ["shifted", "rational", "affine"])
def test_retrofit_keeps_trained_calibrated_weights_and_source(base, family):
    torch.manual_seed(909)
    config = ModelConfig(
        variant=base, width=24, layers=2, heads=3, context=8, hidden=48, groups=3, vocab_size=32
    )
    source = Transformer(config)
    if base == "swiglu_narrow":
        initialize_dense_width(source, TrainConfig(ffn_width_init_mode="fan_in"))
    x = torch.arange(16).reshape(2, 8)
    optimizer = torch.optim.Adam(source.parameters(), lr=0.003)
    for _ in range(3):
        optimizer.zero_grad()
        source.loss(x, x + 1).backward()
        optimizer.step()
    source.eval()
    before = {name: p.detach().clone() for name, p in source.named_parameters()}
    adapted = add_learnable_activations(source, family)
    assert not adapted.training
    assert torch.equal(source(x), adapted(x))
    assert all(torch.equal(p, before[name]) for name, p in source.named_parameters())
    assert all(
        p is not dict(adapted.named_parameters())[name] for name, p in source.named_parameters()
    )
    assert all(
        torch.count_nonzero(p) == 0 for name, p in adapted.named_parameters() if ".curve." in name
    )
    adapted.train()
    adapted.loss(x, x + 1).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapted.parameters())


def test_reject_repeated_retrofit_and_unknown_family():
    source = Transformer(
        ModelConfig(variant="swiglu_narrow", width=24, layers=1, heads=3, vocab_size=32, context=8)
    )
    with pytest.raises(ValueError, match="Choose"):
        add_learnable_activations(source, "unknown")
    adapted = add_learnable_activations(source)
    with pytest.raises(ValueError, match="unmodified"):
        add_learnable_activations(adapted)
