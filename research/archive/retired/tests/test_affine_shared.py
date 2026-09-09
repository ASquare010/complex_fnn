"""Affine specialization must preserve its strict-sharing baseline at zero."""

from dataclasses import replace

import torch

from src.affine_shared_ffn import AffineSharedFFN
from src.core.config import ModelConfig
from src.core.transformer import Transformer
from src.dense_ffn import DenseFFN


def test_counts_and_sharing():
    cfg = ModelConfig(variant="shared_swiglu_affine")
    assert cfg.unique_ffn_parameters == 301056
    assert cfg.total_parameters == 1679040
    model = Transformer(cfg)
    assert model.blocks[0].ffn.base is model.blocks[3].ffn.base
    assert model.blocks[0].ffn.scale_delta is not model.blocks[1].ffn.scale_delta


def test_exact_initial_logits_and_live_parameters():
    cfg = ModelConfig(
        variant="shared_swiglu_affine", width=24, heads=3, layers=4, context=8, vocab_size=32
    )
    candidate = Transformer(cfg)
    control = Transformer(replace(cfg, variant="shared_swiglu"))
    x = torch.randint(32, (2, 8))
    torch.testing.assert_close(candidate(x), control(x), atol=0, rtol=0)
    candidate.loss(x, x).backward()
    for name, p in candidate.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        if "delta" in name:
            assert p.grad.abs().sum() > 0


def test_input_gradcheck():
    model = AffineSharedFFN(DenseFFN(4, 8, "swiglu")).double()
    with torch.no_grad():
        model.scale_delta.normal_(0, 0.1)
        model.shift_delta.normal_(0, 0.1)
        model.output_delta.normal_(0, 0.1)
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(model, (x,))
