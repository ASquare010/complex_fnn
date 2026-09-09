"""Shared tensors must be truly shared, counted once, and accumulate all gradients."""

import io
from dataclasses import replace

import torch

from src.core.config import ModelConfig
from src.core.transformer import Transformer
from src.shared_ffn import shared_ffn_for_layer


def test_sharing_identity_and_boundaries():
    cfg = ModelConfig(variant="shared_gelu", width=24, heads=3, layers=6, context=8, vocab_size=32)
    model = Transformer(cfg)
    assert model.blocks[0].ffn is model.blocks[3].ffn
    assert model.blocks[4].ffn is model.blocks[5].ffn
    assert model.blocks[0].ffn is not model.blocks[4].ffn
    assert model.blocks[0].attn is not model.blocks[1].attn
    assert sum(p.numel() for p in model.parameters()) == cfg.total_parameters
    assert cfg.unique_ffn_parameters == 2 * cfg.ffn_parameters


def test_exact_seventy_five_percent_unique_ffn_reduction():
    for variant in ("shared_gelu", "shared_swiglu"):
        cfg = ModelConfig(variant=variant)
        assert cfg.unique_ffn_parameters == 294912
        assert cfg.total_parameters == 1672896
        assert cfg.unique_ffn_parameters / (cfg.layers * cfg.ffn_parameters) == 0.25


def test_shared_gradients_equal_sum_of_separate_uses():
    bank = {}
    f = shared_ffn_for_layer(bank, 0, 4, 8, "gelu").double()
    assert f is shared_ffn_for_layer(bank, 3, 4, 8, "gelu")
    x = torch.randn(3, 4, dtype=torch.float64)
    y = torch.randn(2, 4, dtype=torch.float64)
    params = tuple(f.parameters())
    first = torch.autograd.grad(f(x).square().sum(), params)
    second = torch.autograd.grad(f(y).square().sum(), params)
    combined = torch.autograd.grad(f(x).square().sum() + f(y).square().sum(), params)
    for a, b, c in zip(first, second, combined, strict=True):
        torch.testing.assert_close(a + b, c, atol=1e-12, rtol=1e-12)


def test_shared_checkpoint_preserves_identity_and_logits():
    cfg = ModelConfig(
        variant="shared_swiglu", width=24, heads=3, layers=4, context=8, vocab_size=32
    )
    model = Transformer(cfg)
    x = torch.randint(32, (2, 8))
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = Transformer(cfg)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    assert restored.blocks[0].ffn is restored.blocks[3].ffn
    torch.testing.assert_close(model(x), restored(x), atol=0, rtol=0)
    baseline = Transformer(replace(cfg, variant="swiglu"))
    for name, p in model.named_parameters():
        if ".ffn." not in name:
            torch.testing.assert_close(p, dict(baseline.named_parameters())[name], atol=0, rtol=0)


def test_shared_activation_diagnostics_follow_each_invocation():
    from src.core.diagnostics import inspect_layers

    cfg = ModelConfig(variant="shared_gelu", width=24, heads=3, layers=4, context=8, vocab_size=32)
    model = Transformer(cfg)
    x = torch.randint(32, (2, 8))
    observed = []
    handle = model.blocks[0].ffn.register_forward_hook(
        lambda module, inputs, output: observed.append(output.float().std(unbiased=False).item())
    )
    try:
        records = inspect_layers(model, x, "cpu", "fp32")
    finally:
        handle.remove()
    assert len(observed) == 4
    for i, value in enumerate(observed):
        assert records[f"layer_{i}.ffn"]["std"] == value
