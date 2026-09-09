"""Whole-decoder geometry, initialization, gradients and checkpoint contracts."""

import io
import math
from dataclasses import replace

import pytest
import torch

from src.core.benchmark import forward_flops
from src.core.config import ModelConfig
from src.core.diagnostics import inspect_layers
from src.core.transformer import Transformer
from src.multihead_ffn.overcomplete import OvercompleteHeadwiseFFN


def test_registered_overcomplete_budget_and_invalid_geometry():
    c = ModelConfig(variant="overcomplete_headwise_swiglu", width=384, layers=8, heads=6, groups=8)
    c.validate()
    assert (
        c.ffn_width == 200 and c.unique_ffn_parameters == 2801664 and c.total_parameters == 9099648
    )
    assert forward_flops(c)["ffn_matrix_forward_flops_per_token"] == 5603328
    assert replace(c, hidden=192).ffn_width == 192
    for invalid in (replace(c, groups=0), replace(c, groups=7), replace(c, width=400, heads=5)):
        with pytest.raises(ValueError):
            invalid.validate()


def test_parent_calibration_common_weights_gradients_and_checkpoint_roundtrip():
    c = ModelConfig(
        variant="overcomplete_headwise_swiglu",
        width=24,
        layers=2,
        heads=3,
        context=8,
        vocab_size=32,
        groups=2,
        hidden=16,
    )
    model = Transformer(c, 17)
    reference = Transformer(replace(c, variant="swiglu", hidden=0), 17)
    common = dict(reference.named_parameters())
    assert all(torch.equal(p, common[n]) for n, p in model.named_parameters() if ".ffn." not in n)
    for layer, block in enumerate(model.blocks):
        standalone = OvercompleteHeadwiseFFN(24, 32, 4, 2, 16)
        standalone.initialize(17, f"blocks.{layer}.ffn", 1 / math.sqrt(2 * c.layers))
        assert all(
            torch.equal(p, standalone.state_dict()[n]) for n, p in block.ffn.state_dict().items()
        )
    x = torch.arange(16).reshape(2, 8)
    before = model(x).detach()
    diagnostics = inspect_layers(model, x, "cpu", "fp32")
    assert all(r["finite"] for r in diagnostics.values())
    assert torch.equal(before, model(x))
    model.loss(x, x + 1).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
        for p in model.parameters()
    )
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = Transformer(c, 29)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.equal(model(x), restored(x))
