"""H067 isolated residual token-conditioned activation; no active model registration."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.blockshuffle_ffn import BlockShuffleFFN


def mixed_product(u, v, alpha):
    return F.silu(u) * (v + alpha.to(v.dtype) * F.silu(v))


class ResidualActivationFFN(BlockShuffleFFN):
    amplitude = 0.25

    def __init__(self, width, hidden, groups, adaptive):
        super().__init__(width, hidden, groups)
        self.adaptive = adaptive
        if adaptive:
            self.router = nn.Linear(width, 1, bias=True)
            nn.init.zeros_(self.router.weight)
            nn.init.zeros_(self.router.bias)
        else:
            self.gate_bias = nn.Parameter(torch.zeros(1))

    def mixing_weight(self, x):
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(x.device.type, enabled=False):
            if self.adaptive:
                routing = F.linear(
                    x.to(dtype), self.router.weight.to(dtype), self.router.bias.to(dtype)
                )
            else:
                routing = self.gate_bias.to(dtype)
            return self.amplitude * routing.tanh()

    def forward(self, x):
        u, v = self.up(x), self.gate(x)
        alpha = self.mixing_weight(x)
        if self.recompute_gate and self.training and torch.is_grad_enabled():
            if self.gate_recompute_method != "checkpoint":
                raise ValueError("This qualification uses native product checkpointing")
            z = checkpoint(
                mixed_product, u, v, alpha, use_reentrant=False, preserve_rng_state=False
            )
        else:
            z = mixed_product(u, v, alpha)
        return self.down(z)
