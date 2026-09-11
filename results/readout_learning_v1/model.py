"""Small readout-extension pilot with mandatory conventional activation controls."""

import torch
from torch import nn

from results.fused_reconstruction_v1.model import Fused
from results.reversible_training_v1.model import Reversible, stack

ARMS = (
    "relu",
    "leaky_relu",
    "prelu",
    "gelu",
    "silu",
    "swiglu",
    "budget_gelu",
    "linear",
    "learned",
    "fixed",
    "affine",
)


class Readout(nn.Module):
    def __init__(self, arm, seed=263):
        super().__init__()
        self.core = (
            Fused("learned:reconstruct", d=32, seed=seed, depth=4)
            if arm == "learned"
            else Reversible(arm + ":reconstruct", d=32, seed=seed, depth=4)
        )
        self.readout = nn.Linear(32, 32)
        with torch.no_grad():
            self.core.bias.zero_()
            A = torch.eye(32)
            for q in self.core.q:
                A = q @ A
            self.readout.weight.copy_(A.T)
            self.readout.bias.zero_()

    def forward(self, x):
        h = (
            self.core(x)
            if x.is_cuda
            else stack(
                x, self.core.theta, self.core.bias, self.core.q, self.core.rho, self.core.kind
            )
        )
        return self.readout(h)


class Block(nn.Module):
    def __init__(self, arm, h):
        super().__init__()
        self.up = nn.Linear(32, h)
        self.down = nn.Linear(h, 32)
        self.act = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "prelu": nn.PReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "swiglu": nn.SiLU,
            "budget_gelu": nn.GELU,
        }[arm]()
        self.gate = nn.Linear(32, h) if arm == "swiglu" else None

    def forward(self, x):
        z = self.act(self.up(x))
        if self.gate is not None:
            z = z * self.gate(x)
        return x + self.down(z) / 2


class Dense(nn.Module):
    def __init__(self, arm):
        super().__init__()
        h = 4 if arm == "budget_gelu" else 21 if arm == "swiglu" else 32
        self.blocks = nn.Sequential(*(Block(arm, h) for _ in range(4)))

    def forward(self, x):
        return self.blocks(x)


def Model(arm, seed):
    assert arm in ARMS
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if arm in ("learned", "fixed", "affine"):
            return Readout(arm, seed)
        if arm == "linear":
            return nn.Linear(32, 32)
        m = Dense(arm)
        for module in m.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)
        return m
