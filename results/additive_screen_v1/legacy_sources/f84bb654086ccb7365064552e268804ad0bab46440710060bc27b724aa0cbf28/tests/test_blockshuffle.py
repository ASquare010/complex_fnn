"""Check the structured map independently against explicit dense matrices."""

import math

import pytest
import torch

from src.blockshuffle_ffn import BlockShuffleLinear, unshuffle_channels
from src.core.benchmark import forward_flops
from src.core.config import ModelConfig
from src.core.transformer import Transformer
from src.grouped_ffn import shuffle_channels


@pytest.mark.parametrize("widths", [(8, 16), (16, 8)])
def test_blockshuffle_dense_equivalence_and_gradients(widths):
    n, m = widths
    projection = BlockShuffleLinear(n, m, 2).double()
    projection.initialize(17, "check")
    x = torch.randn(3, n, dtype=torch.float64, requires_grad=True)
    first = torch.block_diag(*projection.first.weight.unbind())
    second = torch.block_diag(*projection.second.weight.unbind())
    mid_order = torch.arange(min(n, m)).reshape(2, -1).T.flatten()
    out_order = torch.arange(m).reshape(-1, 2).T.flatten()
    expected = ((x @ first.T)[:, mid_order] @ second.T)[:, out_order]
    torch.testing.assert_close(projection(x), expected, atol=1e-13, rtol=1e-13)
    assert torch.autograd.gradcheck(projection, (x,))
    dense = projection(torch.eye(n, dtype=torch.float64)).T
    scale = 0.02 * math.sqrt(max(n, m))
    torch.testing.assert_close(
        torch.linalg.svdvals(dense),
        torch.full((min(n, m),), scale, dtype=torch.float64),
        atol=1e-13,
        rtol=1e-13,
    )


def test_shuffle_inverse_and_full_ffn_budget():
    x = torch.randn(2, 24)
    torch.testing.assert_close(unshuffle_channels(shuffle_channels(x, 8), 8), x, atol=0, rtol=0)
    config = ModelConfig(variant="blockshuffle_swiglu", hidden=832, groups=8)
    model = Transformer(config)
    assert sum(p.numel() for p in model.parameters()) == 1672896
    assert config.unique_ffn_parameters == 294912
    assert forward_flops(config) == forward_flops(ModelConfig(variant="swiglu_narrow"))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("method", ["checkpoint", "native"])
def test_gate_recomputation_preserves_forward_and_parameter_gradients(device, method):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from contextlib import nullcontext

    cfg = ModelConfig(
        variant="blockshuffle_swiglu",
        width=24,
        heads=3,
        layers=2,
        context=8,
        vocab_size=32,
        groups=4,
    )
    a, b = Transformer(cfg).to(device), Transformer(cfg).to(device)
    for block in b.blocks:
        block.ffn.recompute_gate = True
        block.ffn.gate_recompute_method = method
    x = torch.arange(16, device=device).reshape(2, 8)
    losses = []
    for model in (a, b):
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
        )
        with context:
            loss = model.loss(x, x)
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(losses[0], losses[1], atol=0, rtol=0)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa.grad, pb.grad, atol=0, rtol=0)


def test_native_recomputed_gate_gradcheck_and_second_derivatives():
    from src.blockshuffle_ffn import RecomputedSwiGLU

    u = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(RecomputedSwiGLU.apply, (u, v))
    assert torch.autograd.gradgradcheck(RecomputedSwiGLU.apply, (u, v))
