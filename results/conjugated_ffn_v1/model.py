"""Experimental two-use shared FFN; no active factory/default changes."""

import math

import torch
from torch import nn
from torch.nn import functional as F

ARMS = (
    "full_gelu",
    "full_swiglu",
    "narrow_gelu",
    "shared_same",
    "shared_permuted",
    "untied_permuted",
)


class Branch(nn.Module):
    def __init__(self, d, h, gated=False):
        super().__init__()
        self.up = nn.Linear(d, h)
        self.gate = nn.Linear(d, h) if gated else None
        self.down = nn.Linear(h, d)

    def forward(self, x):
        u = self.up(x)
        z = F.gelu(u) if self.gate is None else F.silu(u) * self.gate(x)
        return self.down(z)


class Model(nn.Module):
    def __init__(self, arm, d=384, seed=17):
        super().__init__()
        assert arm in ARMS and d % 6 == 0
        self.arm = arm
        torch.manual_seed(seed)
        h = d if arm == "full_gelu" else 2 * d // 3 if arm == "full_swiglu" else d // 2
        self.first = Branch(d, h, arm == "full_swiglu")
        self.second = Branch(d, h) if arm == "untied_permuted" else None
        permutation = torch.randperm(d, generator=torch.Generator().manual_seed(881))
        if arm == "shared_same":
            permutation = torch.arange(d)
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse", permutation.argsort())
        self.mac_per_token = (3 * d * h if arm == "full_swiglu" else 2 * d * h) * (
            2 if "shared" in arm or arm == "untied_permuted" else 1
        )

    def forward(self, x):
        if self.arm in ("full_gelu", "full_swiglu", "narrow_gelu"):
            return self.first(x)
        z = x + self.first(x) / math.sqrt(2)
        branch = self.first if self.second is None else self.second
        z = z + branch(z[..., self.permutation])[..., self.inverse] / math.sqrt(2)
        return z - x
