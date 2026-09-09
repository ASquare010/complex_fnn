"""Per-invocation diagnostics must preserve caller state and shared-use identity."""

import torch

from src.core.config import ModelConfig
from src.core.model_audit import gradient_flow
from src.core.transformer import Transformer


def test_shared_invocation_gradients_preserve_existing_parameter_gradients_and_mode():
    model = Transformer(
        ModelConfig(variant="shared_swiglu", width=24, heads=3, layers=4, context=8, vocab_size=32)
    ).eval()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    x = torch.arange(16).reshape(2, 8)
    result = gradient_flow(model, x, x)
    assert len(result["records"]) == 12
    assert all(row["finite"] for row in result["records"].values())
    assert len({result["records"][f"layer_{i}.output"]["gradient_rms"] for i in range(4)}) == 4
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())
    assert not model.training
    assert not model.blocks[0].ffn._forward_hooks
