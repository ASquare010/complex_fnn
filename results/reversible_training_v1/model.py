"""First-order reconstruction boundary with dense mixing buffers counted explicitly."""

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.utils.checkpoint import checkpoint

from results.modulation_depth_v1.model import Model as DenseStack
from results.reversible_softsign_v1.model import inverse, phi

ARMS = (
    "full_gelu:eager",
    "full_swiglu:eager",
    "full_gelu:checkpoint",
    "full_swiglu:checkpoint",
    "budget_gelu:checkpoint",
    "learned:eager",
    "learned:checkpoint",
    "learned:reconstruct",
    "fixed:reconstruct",
    "affine:reconstruct",
)


def stack(x, theta, bias, q, rho, kind):
    for j in range(len(q)):
        x = x @ q[j].T + bias[j]
        a = rho * theta[j].tanh()
        x = x * (1 + a) if kind == "affine" else phi(x, a)
    return x


class Reconstruct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, theta, bias, q, rho, kind):
        y = stack(x, theta, bias, q, rho, kind)
        ctx.save_for_backward(y, theta, bias, q)
        ctx.rho, ctx.kind = rho, kind
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        y, theta, bias, q = ctx.saved_tensors
        gt, gb = torch.empty_like(theta), torch.empty_like(bias)
        x = y
        for j in reversed(range(len(q))):
            t = theta[j].tanh()
            a = ctx.rho * t
            z = x / (1 + a) if ctx.kind == "affine" else inverse(x, a)
            basis = z if ctx.kind == "affine" else z / (1 + z.abs())
            slope = 1 + a if ctx.kind == "affine" else 1 + a / (1 + z.abs()).square()
            gt[j] = (g * basis).sum(0) * ctx.rho * (1 - t.square())
            dz = g * slope
            gb[j] = dz.sum(0)
            g = dz @ q[j]
            if j:
                x = (z - bias[j]) @ q[j]
        return g, gt if ctx.needs_input_grad[1] else None, gb, None, None, None


class Reversible(nn.Module):
    def __init__(self, arm, d=384, seed=173, depth=8):
        super().__init__()
        self.kind, self.mode = arm.split(":")
        self.rho = 2 / depth
        assert 0 < self.rho < 1
        gen = torch.Generator().manual_seed(seed + depth)
        q = torch.linalg.qr(torch.randn(depth, d, d, generator=gen, dtype=torch.float64)).Q.float()
        theta = (
            torch.empty(depth, d, dtype=torch.float64).uniform_(-1.5, 1.5, generator=gen).float()
        )
        bias = (0.1 * torch.randn(depth, d, generator=gen, dtype=torch.float64)).float()
        self.register_buffer("q", q)
        if self.kind == "fixed":
            self.register_buffer("theta", theta)
        else:
            self.theta = nn.Parameter(theta)
        self.bias = nn.Parameter(bias)
        self.macs = depth * d * d

    def forward(self, x):
        if self.mode == "reconstruct" and torch.is_grad_enabled():
            return Reconstruct.apply(x, self.theta, self.bias, self.q, self.rho, self.kind)
        if self.mode == "checkpoint" and torch.is_grad_enabled():
            for j in range(len(self.q)):

                def layer(t, j=j):
                    return phi(t @ self.q[j].T + self.bias[j], self.rho * self.theta[j].tanh())

                x = checkpoint(layer, x, use_reentrant=False, preserve_rng_state=False)
            return x
        return stack(x, self.theta, self.bias, self.q, self.rho, self.kind)


def Model(arm, d=384, seed=173, depth=8):
    assert arm in ARMS
    if arm.split(":")[0] in ("full_gelu", "full_swiglu", "budget_gelu"):
        return DenseStack(arm, d=d, seed=seed, depth=depth)
    return Reversible(arm, d=d, seed=seed, depth=depth)
