"""Zero-correction removal preserves learned common weights and the function."""

from dataclasses import replace

import pytest
import torch

from src.core.activation_retrofit import remove_zero_learnable_activations
from src.core.config import ModelConfig
from src.core.transformer import Transformer


@pytest.mark.parametrize("base", ["swiglu_narrow", "blockshuffle_swiglu"])
def test_removal_copies_trained_common_parameters_and_rejects_nonzero_shape(base):
    config = ModelConfig(
        variant=base + "_affine_activation",
        width=24,
        hidden=48,
        layers=2,
        heads=3,
        vocab_size=32,
        context=8,
        groups=3,
    )
    model = Transformer(config)
    x = torch.arange(16).reshape(2, 8)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    for _ in range(3):
        opt.zero_grad()
        model.loss(x, x + 1).backward()
        opt.step()
    with pytest.raises(ValueError, match="Reset"):
        remove_zero_learnable_activations(model)
    with torch.no_grad():
        for block in model.blocks:
            for p in block.ffn.curve.parameters():
                p.zero_()
    model.eval()
    copied = remove_zero_learnable_activations(model)
    assert copied.config == replace(config, variant=base)
    assert not copied.training
    assert torch.equal(model(x), copied(x))
    source = dict(model.named_parameters())
    for name, parameter in copied.named_parameters():
        assert torch.equal(parameter, source[name]) and parameter is not source[name]
    assert not any("curve" in name for name in copied.state_dict())
