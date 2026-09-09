"""Caching must preserve parameters and be explicit about extra storage."""

import pytest
import torch

from src.blockshuffle_ffn import BlockShuffleLinear
from src.core.serving import CachedProjection


def test_cached_dense_matches_factorized_fp64_and_keeps_original_parameters():
    original = BlockShuffleLinear(8, 16, 2).double()
    original.initialize(17, "cache")
    cached = CachedProjection(original, torch.float64).eval()
    x = torch.randn(3, 8, dtype=torch.float64)
    torch.testing.assert_close(cached(x), original(x), atol=1e-13, rtol=1e-13)
    assert {id(p) for p in cached.parameters()} == {id(p) for p in original.parameters()}
    assert cached.weight.numel() * cached.weight.element_size() == 8 * 16 * 8
    cached.train()
    with pytest.raises(RuntimeError, match="inference only"):
        cached(x)


def test_packed_factors_preserve_ffn_function_and_learned_parameters():
    from src.core.config import ModelConfig
    from src.core.serving import PackedGateFFN
    from src.core.transformer import Transformer

    model = Transformer(
        ModelConfig(
            variant="blockshuffle_swiglu",
            width=24,
            heads=3,
            layers=2,
            vocab_size=32,
            context=8,
            groups=4,
        )
    )
    base = model.blocks[0].ffn.double().eval()
    packed = PackedGateFFN(base, torch.float64).eval()
    x = torch.randn(2, 5, 24, dtype=torch.float64)
    torch.testing.assert_close(packed(x), base(x), atol=1e-13, rtol=1e-13)
    assert {id(p) for p in packed.parameters()} == {id(p) for p in base.parameters()}
