"""Scientific correctness gates, including an actual memorization test."""

from dataclasses import replace

import pytest
import torch

from src.core.config import VARIANTS, ModelConfig, TrainConfig
from src.core.trainer import learning_rate
from src.core.transformer import Transformer
from src.dense_ffn import DenseFFN

torch.set_num_threads(2)


@pytest.mark.parametrize("variant", VARIANTS)
def test_analytic_counts_and_gradients(variant):
    config = ModelConfig(
        variant=variant,
        vocab_size=32,
        width=24,
        heads=3,
        layers=2,
        context=8,
        groups=3 if variant == "quadratic_bezier_mixed" else 8,
    )
    model = Transformer(config)
    assert sum(p.numel() for p in model.parameters()) == config.total_parameters
    x = torch.randint(32, (2, 8))
    model.loss(x, torch.roll(x, -1, 1)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_attention_causality():
    model = Transformer(ModelConfig(vocab_size=32, width=24, heads=3, layers=2, context=8)).eval()
    x = torch.randint(32, (2, 8))
    changed = x.clone()
    changed[:, 5:] = (changed[:, 5:] + 1) % 32
    torch.testing.assert_close(model(x)[:, :5], model(changed)[:, :5], atol=1e-6, rtol=1e-6)


def test_shared_initialization():
    config = ModelConfig(vocab_size=32, width=24, heads=3, layers=2, context=8)
    a = dict(Transformer(config, 29).named_parameters())
    b = dict(Transformer(replace(config, variant="swiglu"), 29).named_parameters())
    for key in a:
        if ".ffn." not in key:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)


@pytest.mark.parametrize(
    "variant",
    [
        "gelu",
        "swiglu",
        "paired_antipodal_swiglu",
        "paired_reciprocal_swiglu",
        "quadratic_bezier",
        "quadratic_bezier_mixed",
        "swiglu_quadratic",
    ],
)
def test_tiny_overfit(variant):
    torch.manual_seed(5)
    config = ModelConfig(
        variant=variant,
        vocab_size=16,
        width=24,
        heads=3,
        layers=2,
        context=8,
        groups=3 if variant == "quadratic_bezier_mixed" else 8,
    )
    model = Transformer(config)
    x = torch.randint(16, (2, 8))
    y = (x + 1) % 16
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial = model.loss(x, y).item()
    for _ in range(80):
        optimizer.zero_grad()
        loss = model.loss(x, y)
        loss.backward()
        optimizer.step()
    assert model.loss(x, y).item() < 0.1 * initial


def test_swiglu_reference_equation():
    model = DenseFFN(8, 12, "swiglu").double()
    x = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(x @ model.up.weight.T) * (x @ model.gate.weight.T),
        model.down.weight,
    )
    torch.testing.assert_close(model(x), expected)
    assert torch.autograd.gradcheck(model, (x,))


def test_schedule_boundaries():
    config = TrainConfig(steps=100)
    assert learning_rate(9, config) == pytest.approx(config.learning_rate)
    assert learning_rate(99, config) == pytest.approx(0.1 * config.learning_rate)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_backward():
    for variant in ("gelu", "swiglu"):
        model = Transformer(
            ModelConfig(variant=variant, vocab_size=32, width=24, heads=3, layers=2, context=8)
        ).cuda()
        x = torch.randint(32, (2, 8), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(x, x)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
