"""Recomputation preserves the function and derivatives; setup rules are explicit."""

from dataclasses import replace

import pytest
import torch

from src.core.config import ModelConfig, TrainConfig
from src.core.trainer import configure_gate_recomputation
from src.core.transformer import Transformer
from src.multihead_ffn.overcomplete import OvercompleteHeadwiseFFN


def test_overcomplete_native_recompute_forward_all_gradients_and_gradcheck():
    m = OvercompleteHeadwiseFFN(12, 16, 4, 4, 8).double()
    m.initialize(17, "test", 0.3)
    x = torch.randn(2, 3, 12, dtype=torch.float64, requires_grad=True)
    probe = torch.randn_like(x)
    eager = m(x)
    variables = (x, *m.parameters())
    expected = torch.autograd.grad((eager * probe).sum(), variables)
    m.recompute_gate = True
    actual = m(x)
    assert torch.equal(actual, eager)
    gradients = torch.autograd.grad((actual * probe).sum(), variables)
    for a, b in zip(gradients, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-11)
    assert torch.autograd.gradcheck(m, (x,), fast_mode=True)
    with torch.no_grad():
        assert torch.equal(m(x), eager)


def test_shared_gate_setup_accepts_native_and_preserves_older_constraints():
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
    m = Transformer(c, 17)
    original = {n: p.clone() for n, p in m.named_parameters()}
    configure_gate_recomputation(m, TrainConfig())
    assert all(not b.ffn.recompute_gate for b in m.blocks)
    native = TrainConfig(recompute_gate=True, gate_recompute_method="native")
    configure_gate_recomputation(m, native)
    assert all(b.ffn.recompute_gate and b.ffn.gate_recompute_method == "native" for b in m.blocks)
    assert all(torch.equal(p, original[n]) for n, p in m.named_parameters())
    with pytest.raises(ValueError, match="native"):
        configure_gate_recomputation(m, replace(native, gate_recompute_method="checkpoint"))
    dense = Transformer(replace(c, variant="swiglu", hidden=0), 17)
    with pytest.raises(ValueError, match="Gate recomputation"):
        configure_gate_recomputation(dense, native)
    block = Transformer(replace(c, variant="blockshuffle_swiglu", groups=8, hidden=64), 17)
    configure_gate_recomputation(block, replace(native, gate_recompute_method="checkpoint"))
    assert all(b.ffn.gate_recompute_method == "checkpoint" for b in block.blocks)
    learned = Transformer(
        replace(c, variant="blockshuffle_swiglu_affine_activation", groups=8, hidden=64), 17
    )
    with pytest.raises(ValueError, match="Learnable activations require checkpoint"):
        configure_gate_recomputation(learned, native)
