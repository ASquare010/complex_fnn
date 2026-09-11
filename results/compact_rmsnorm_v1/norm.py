"""Analytical RMSNorm VJP using owned scratch buffers; first derivatives only."""

import types

import torch
from torch.autograd.function import once_differentiable

from src.core.transformer import RMSNorm


class CompactRMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        assert x.dtype in (torch.float32, torch.float64) and w.dtype == x.dtype
        r = torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
        ctx.save_for_backward(x, w, r)
        return (x * r) * w

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        x, w, r = ctx.saved_tensors
        dw = (g * x).mul_(r).sum(tuple(range(x.ndim - 1)))
        dx = g * w
        correction = (dx * x).mean(-1, keepdim=True)
        correction.mul_(r.square()).mul_(r)
        dx.mul_(r).addcmul_(x, correction, value=-1)
        return dx, dw


def install(model):
    identifiers = {k: id(p) for k, p in model.named_parameters()}

    def forward(this, x):
        return CompactRMS.apply(x, this.weight)

    count = 0
    for module in model.modules():
        if isinstance(module, RMSNorm):
            assert "forward" not in module.__dict__
            module.forward = types.MethodType(forward, module)
            count += 1
    assert identifiers == {k: id(p) for k, p in model.named_parameters()}
    return count
