"""Qualify scaling, gradients, optimizer identity and existing raw JVPs."""

import copy

import torch
from torch.nn import functional as F

from results.sobolev_learning_v1.source.model import TrainingAdapter
from results.sobolev_selection_v1.source.features import PrunedFFN, value_jvp


def run():
    checks = []
    for kind in ("gelu", "swiglu"):
        torch.manual_seed(109)
        raw = PrunedFFN(5, 7, kind).double()
        model = TrainingAdapter(copy.deepcopy(raw), 2.7)
        x, target, direction = [torch.randn(9, 5, dtype=torch.float64) for _ in range(3)]
        loss = F.mse_loss(model(x), target / 2.7)
        direct = F.mse_loss(raw(x), target) / 2.7**2
        torch.testing.assert_close(loss, direct, atol=1e-14, rtol=1e-14)
        loss.backward()
        direct.backward()
        for a, b in zip(raw.parameters(), model.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-14, rtol=1e-14)
        for base in (raw, model):
            optimizer = torch.optim.AdamW(
                base.parameters(), lr=0.001, betas=(0.9, 0.95), weight_decay=0
            )
            torch.nn.utils.clip_grad_norm_(base.parameters(), 1)
            optimizer.step()
        for a, b in zip(raw.parameters(), model.parameters()):
            torch.testing.assert_close(a, b, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(model(x) * 2.7, model.raw(x), atol=1e-14, rtol=1e-14)
        value, derivative = value_jvp(model.raw, x, direction)
        auto_value, auto_derivative = torch.autograd.functional.jvp(model.raw, x, direction)
        torch.testing.assert_close(value, auto_value, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(derivative, auto_derivative, atol=1e-13, rtol=1e-13)
        checks.append(
            {
                "activation": kind,
                "loss_gradient_optimizer_scaling": True,
                "raw_identity": True,
                "jvp": True,
            }
        )
        hidden = 448 if kind == "gelu" else 298
        counted = PrunedFFN(384, hidden, kind)
        count = sum(p.numel() for p in counted.parameters())
        assert count == (2 if kind == "gelu" else 3) * 384 * hidden + 384
        assert 1 - count / 1179648 >= 0.7
        checks.append({"activation": kind, "parameters": count, "seventy_percent_reduction": True})
    return checks
