"""Fused first-order reconstruction; all reference arms retain H148 implementations."""

import torch
from torch.autograd.function import once_differentiable

from results.fused_reconstruction_v1.kernel import local
from results.reversible_training_v1.model import Model as Reference
from results.reversible_training_v1.model import Reversible, stack

ARMS = (
    "full_gelu:eager",
    "full_swiglu:eager",
    "full_gelu:checkpoint",
    "full_swiglu:checkpoint",
    "budget_gelu:checkpoint",
    "learned:checkpoint",
    "learned:reconstruct",
    "learned:fused",
    "fixed:reconstruct",
    "affine:reconstruct",
)


class FusedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, theta, bias, q, rho):
        y = stack(x, theta, bias, q, rho, "learned")
        ctx.save_for_backward(y, theta, bias, q)
        ctx.rho = rho
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        x, theta, bias, q = ctx.saved_tensors
        gt, gb = torch.empty_like(theta), torch.empty_like(bias)
        for j in reversed(range(len(q))):
            pre, dz = local(x, g.contiguous(), theta[j], bias[j], ctx.rho, gt[j], gb[j], bool(j))
            g = dz @ q[j]
            if j:
                x = pre @ q[j]
            del pre, dz
        return g, gt, gb, None, None


class Fused(Reversible):
    def forward(self, x):
        if torch.is_grad_enabled():
            return FusedFunction.apply(x, self.theta, self.bias, self.q, self.rho)
        return super().forward(x)


def Model(arm, d=384, seed=223, depth=8):
    assert arm in ARMS
    return (
        Fused("learned:reconstruct", d=d, seed=seed, depth=depth)
        if arm == "learned:fused"
        else Reference(arm, d=d, seed=seed, depth=depth)
    )
