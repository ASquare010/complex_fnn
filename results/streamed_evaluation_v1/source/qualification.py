"""Independent double-precision/mask and state-preservation evaluation checks."""

import torch

from results.streamed_evaluation_v1.source.evaluation import POLICIES, batch_loss, tensor_hash
from src.core.config import ModelConfig
from src.core.transformer import Transformer


def run():
    checks = []
    for batch, context, chunk in ((3, 7, 10), (5, 5, 12), (2, 11, 4)):
        cfg = ModelConfig(
            variant="gelu_narrow",
            width=24,
            hidden=48,
            layers=2,
            heads=3,
            groups=8,
            vocab_size=32,
            context=context,
        )
        model = Transformer(cfg, 17).double().eval()
        x = torch.arange(batch * context).reshape(batch, context) % 31
        for mask in ("none", "partial", "one_sequence", "all"):
            y = (x + 1).clone()
            if mask == "partial":
                y.flatten()[::3] = -100
            elif mask == "one_sequence":
                y[0] = -100
            elif mask == "all":
                y[:] = -100
            before = {k: tensor_hash(v) for k, v in model.state_dict().items()}
            rng = tensor_hash(torch.random.get_rng_state())
            native = batch_loss(model, x, y, "native", chunk)
            for policy in POLICIES:
                value = batch_loss(model, x, y, policy, chunk)
                torch.testing.assert_close(value, native, rtol=1e-10, atol=1e-10, equal_nan=True)
                assert not value.requires_grad and all(p.grad is None for p in model.parameters())
                assert {k: tensor_hash(v) for k, v in model.state_dict().items()} == before
                assert tensor_hash(torch.random.get_rng_state()) == rng
                checks.append(
                    {
                        "batch": batch,
                        "context": context,
                        "chunk": chunk,
                        "mask": mask,
                        "policy": policy,
                        "matches": True,
                    }
                )
    return checks
