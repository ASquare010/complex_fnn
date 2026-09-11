"""Eight residual modules with equally available checkpoint controls."""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from results.split_modulation_v1.model import Model as Single

BASE_ARMS = ("full_gelu", "full_swiglu", "budget_gelu", "shared_gate", "split_gate")
ARMS = tuple(f"{a}:{mode}" for mode in ("eager", "checkpoint") for a in BASE_ARMS)


class Model(nn.Module):
    def __init__(self, arm, d=384, seed=83, depth=8):
        super().__init__()
        base, mode = arm.split(":")
        assert base in BASE_ARMS and mode in ("eager", "checkpoint")
        self.mode = mode
        self.blocks = nn.ModuleList(Single(base, d=d, seed=seed + 101 * i) for i in range(depth))
        self.scale = 1 / math.sqrt(depth)
        self.macs = sum(b.macs for b in self.blocks)

    def forward(self, x):
        for block in self.blocks:

            def residual(t, block=block):
                return t + self.scale * block(t)

            x = (
                checkpoint(residual, x, use_reentrant=False, preserve_rng_state=False)
                if self.mode == "checkpoint" and torch.is_grad_enabled()
                else residual(x)
            )
        return x
