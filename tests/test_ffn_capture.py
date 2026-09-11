"""Capture is equivalent to real execution and cleans up on exceptional exits."""

import pytest
import torch
from torch import nn

from src.core.config import ModelConfig
from src.core.ffn_capture import capture_ffn_pairs
from src.core.transformer import Transformer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA capture parity")


def tiny_model():
    return Transformer(
        ModelConfig(width=24, hidden=48, layers=2, heads=3, vocab_size=41, context=8)
    ).cuda()


def test_capture_uneven_batches_matches_full_decoder():
    model = tiny_model().eval()
    windows = torch.randint(41, (5, 8))
    wanted = []
    handle = model.blocks[1].ffn.register_forward_hook(
        lambda module, args, output: wanted.append((args[0].float().cpu(), output.float().cpu()))
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in windows.split(2):
            model(batch.cuda())
    handle.remove()
    model.train()
    pairs = capture_ffn_pairs(model, windows, [0, 1], batch_size=2)
    for j, key in enumerate(("x", "y")):
        torch.testing.assert_close(
            pairs[1][key], torch.cat([v[j].flatten(0, 1) for v in wanted]), atol=0, rtol=0
        )
    assert pairs[0]["x"].shape == (40, 24)
    assert model.training and not model.blocks[1].ffn._forward_hooks


def test_capture_restores_state_after_forward_failure():
    class Broken(nn.Module):
        def forward(self, x):
            raise RuntimeError("deliberate capture probe failure")

    model = tiny_model().train()
    model.blocks[1].ffn = Broken()
    with pytest.raises(RuntimeError, match="deliberate capture"):
        capture_ffn_pairs(model, torch.ones(2, 8, dtype=torch.long), [0, 1])
    assert model.training
    assert not model.blocks[0].ffn._forward_hooks and not model.blocks[1].ffn._forward_hooks


def test_capture_rejects_ambiguous_duplicate_layers():
    model = tiny_model()
    with pytest.raises(ValueError, match="distinct"):
        capture_ffn_pairs(model, torch.ones(2, 8, dtype=torch.long), [0, 0])
