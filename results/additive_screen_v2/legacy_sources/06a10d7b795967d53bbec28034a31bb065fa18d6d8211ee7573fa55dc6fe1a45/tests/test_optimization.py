"""Protect historical optimizer behavior and declared factor-specific rates."""

from dataclasses import replace

import pytest
import torch

from src.core.config import ModelConfig, TrainConfig
from src.core.optimization import parameter_groups
from src.core.transformer import Transformer


def test_default_optimizer_matches_historical_update():
    cfg = ModelConfig(width=24, heads=3, layers=2, vocab_size=32, context=8)
    a, b = Transformer(cfg), Transformer(cfg)
    tc = TrainConfig()
    old = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in a.named_parameters() if p.ndim >= 2 and "theta" not in n],
                "weight_decay": tc.weight_decay,
            },
            {
                "params": [p for n, p in a.named_parameters() if p.ndim < 2 or "theta" in n],
                "weight_decay": 0.0,
            },
        ],
        lr=tc.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    new = torch.optim.AdamW(
        parameter_groups(b, tc), lr=tc.learning_rate, betas=(0.9, 0.95), eps=1e-8
    )
    x = torch.arange(16).reshape(2, 8)
    for _ in range(3):
        for model, opt in ((a, old), (b, new)):
            opt.zero_grad(set_to_none=True)
            model.loss(x, x).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb, atol=0, rtol=0)


def test_fan_in_rates_and_unique_parameter_membership():
    cfg = ModelConfig(variant="blockshuffle_swiglu", hidden=832, groups=8)
    model = Transformer(cfg)
    groups = parameter_groups(model, TrainConfig(ffn_lr_mode="fan_in"))
    lookup = {id(p): g["lr_scale"] for g in groups for p in g["params"]}
    assert len(lookup) == sum(len(g["params"]) for g in groups) == len(list(model.parameters()))
    f = model.blocks[0].ffn
    assert lookup[id(f.up.first.weight)] == 4
    assert lookup[id(f.up.second.weight)] == 4
    assert lookup[id(f.down.first.weight)] == 4
    assert lookup[id(f.down.second.weight)] == pytest.approx(832 / 48)
    assert lookup[id(model.embedding.weight)] == 1
    dense = Transformer(replace(cfg, variant="swiglu"))
    assert all(
        g["lr_scale"] == 1 for g in parameter_groups(dense, TrainConfig(ffn_lr_mode="fan_in"))
    )


def test_product_decay_matches_dense_shrinkage_to_second_order():
    cfg = ModelConfig(
        variant="blockshuffle_swiglu",
        width=24,
        heads=3,
        layers=2,
        vocab_size=32,
        context=8,
        groups=4,
    )
    model = Transformer(cfg).double()
    tc = TrainConfig(ffn_lr_mode="fan_in", ffn_decay_mode="product", learning_rate=0.01)
    projection = model.blocks[0].ffn.up
    eye = torch.eye(24, dtype=torch.float64)
    before = projection(eye).detach().clone()
    optimizer = torch.optim.AdamW(parameter_groups(model, tc), lr=tc.learning_rate)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()
    expected = (1 - tc.learning_rate * tc.weight_decay / 2) ** 2
    torch.testing.assert_close(projection(eye), before * expected, atol=1e-13, rtol=1e-13)
    assert expected - (1 - tc.learning_rate * tc.weight_decay) == pytest.approx(
        (tc.learning_rate * tc.weight_decay) ** 2 / 4
    )
