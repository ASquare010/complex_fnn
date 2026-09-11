"""Matched-budget diagonal, shared and split observation controls."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

ARMS = ("full_gelu", "full_swiglu", "budget_gelu", "static_diagonal", "shared_gate", "split_gate")


class Model(nn.Module):
    def __init__(self, arm, d=384, seed=53):
        super().__init__()
        assert arm in ARMS and d % 12 == 0
        self.arm = arm
        h = (
            d
            if arm == "full_gelu"
            else 2 * d // 3
            if arm == "full_swiglu"
            else d // 2
            if arm == "shared_gate"
            else 3 * d // 4
        )
        base_h = d // 2 if arm == "split_gate" else h
        self.base_h = base_h
        self.up = nn.Linear(d, h)
        self.down = nn.Linear(base_h, d)
        self.gate = None
        if arm == "full_swiglu":
            self.gate = nn.Linear(d, h)
        elif arm in ("shared_gate", "split_gate"):
            self.gate = nn.Linear(h if arm == "shared_gate" else d // 4, d)
        if arm == "static_diagonal":
            self.diagonal = nn.Parameter(torch.zeros(d))

        def initialize(layer, tag, fan):
            digest = hashlib.sha256(f"{seed}:{tag}".encode()).digest()
            generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
            master = torch.empty(d, d).uniform_(
                -1 / math.sqrt(fan), 1 / math.sqrt(fan), generator=generator
            )
            with torch.no_grad():
                layer.weight.copy_(master[: layer.out_features, : layer.in_features])
                layer.bias.zero_()

        initialize(self.up, "up", d)
        initialize(self.down, "down", base_h)
        if self.gate is not None:
            if arm == "full_swiglu":
                initialize(self.gate, "swiglu_gate", d)
            else:
                nn.init.zeros_(self.gate.weight)
                nn.init.zeros_(self.gate.bias)
        self.macs = (
            3 * d * h if arm == "full_swiglu" else 3 * d * h if arm == "shared_gate" else 2 * d * h
        )

    def forward(self, x):
        u = self.up(x)
        if self.arm == "full_swiglu":
            return self.down(F.silu(u) * self.gate(x))
        z = F.gelu(u)
        if self.arm == "split_gate":
            return self.down(z[..., : self.base_h]) + x * self.gate(z[..., self.base_h :])
        result = self.down(z)
        if self.arm == "shared_gate":
            result = result + x * self.gate(z)
        elif self.arm == "static_diagonal":
            result = result + x * self.diagonal
        return result
