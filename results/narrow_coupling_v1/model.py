"""Narrow learned additive coupling; established mechanism, experimental budget."""

import torch
from torch import nn

from results.readout_learning_v1.model import Block

ARMS = (
    "relu",
    "leaky_relu",
    "prelu",
    "gelu",
    "silu",
    "swiglu",
    "budget_gelu",
    "coupling",
    "fixed_coupling",
    "affine_coupling",
)


class Coupling(nn.Module):
    def __init__(self, arm, seed, d=32, depth=4):
        super().__init__()
        self.arm, self.depth = arm, depth
        h = d // 4
        self.up = nn.ModuleList(nn.Linear(d // 2, h) for _ in range(depth))
        self.down = nn.ModuleList(nn.Linear(h, d // 2) for _ in range(depth))
        gen = torch.Generator().manual_seed(seed + 15400)
        self.register_buffer(
            "permutations", torch.stack([torch.randperm(d, generator=gen) for _ in range(depth)])
        )
        self.register_buffer("inverse_permutations", self.permutations.argsort(dim=1))
        self.readout = nn.Linear(d, d)
        for layer in [*self.up, *self.down]:
            nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.readout.weight.copy_(torch.eye(d))
            self.readout.bias.zero_()
        if arm == "fixed_coupling":
            for layer in self.up:
                for name in ("weight", "bias"):
                    value = getattr(layer, name).detach()
                    delattr(layer, name)
                    layer.register_buffer(name, value)

    def update(self, b, j):
        z = self.up[j](b)
        if self.arm != "affine_coupling":
            z = torch.nn.functional.gelu(z)
        return self.down[j](z) / self.depth**0.5

    def core(self, x):
        for j in range(self.depth):
            a, b = x[:, self.permutations[j]].chunk(2, dim=-1)
            x = torch.cat((a + self.update(b, j), b), dim=-1)[:, self.inverse_permutations[j]]
        return x

    def inverse_core(self, y):
        for j in reversed(range(self.depth)):
            a, b = y[:, self.permutations[j]].chunk(2, dim=-1)
            y = torch.cat((a - self.update(b, j), b), dim=-1)[:, self.inverse_permutations[j]]
        return y

    def forward(self, x):
        return self.readout(self.core(x))


class Dense(nn.Module):
    def __init__(self, arm):
        super().__init__()
        h = 8 if arm == "budget_gelu" else 21 if arm == "swiglu" else 32
        self.blocks = nn.Sequential(*(Block(arm, h) for _ in range(4)))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.blocks(x)


def Model(arm, seed):
    assert arm in ARMS
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Coupling(arm, seed) if "coupling" in arm else Dense(arm)
