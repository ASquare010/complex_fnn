"""Small BTT shape ablation; equations follow established two-core BTT prior art."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.blockshuffle_ffn import BlockShuffleLinear

ARMS = ("wide", "narrow33", "narrow40", "narrow27", "blockshuffle", "greedy", "balanced")


def factors(value, balanced):
    if balanced:
        a = math.isqrt(value)
        while value % a:
            a -= 1
        return a, value // a
    parts, divisor = [1, 1], 2
    while divisor * divisor <= value:
        while value % divisor == 0:
            i = 0 if parts[0] <= parts[1] else 1
            parts[i] *= divisor
            value //= divisor
        divisor += 1
    if value > 1:
        parts[0 if parts[0] <= parts[1] else 1] *= value
    return tuple(sorted(parts))


class BTT(nn.Module):
    def __init__(self, din, dout, balanced):
        super().__init__()
        self.m0, self.m1 = factors(din, balanced)
        self.n0, self.n1 = factors(dout, balanced)
        self.right = nn.Parameter(torch.empty(self.m1, self.m0, self.n0))
        self.left = nn.Parameter(torch.empty(self.n0, self.m1, self.n1))
        self.gains = nn.Parameter(torch.ones(2))
        self.scales = (
            math.sqrt(min(self.m0, self.n0)) / self.m0,
            math.sqrt(min(self.m1, self.n1)) / self.m1,
        )
        nn.init.normal_(self.right, std=self.scales[0])
        nn.init.normal_(self.left, std=self.scales[1])
        self.right.lr_scale = din / self.m0 / 2
        self.left.lr_scale = din / self.m1 / 2

    def cores(self):
        return tuple(
            g * w / (torch.sqrt(w.square().mean() + 1e-8) / scale).clamp_min(1)
            for g, w, scale in zip(self.gains, (self.right, self.left), self.scales, strict=True)
        )

    def forward(self, x):
        right, left = self.cores()
        leading = x.shape[:-1]
        x = x.reshape(-1, self.m1, self.m0).transpose(0, 1)
        middle = torch.bmm(x, right).permute(2, 1, 0).contiguous()
        y = torch.bmm(middle, left).transpose(0, 1)
        return y.reshape(*leading, self.n0 * self.n1)

    def dense(self):
        right, left = self.cores()
        return torch.einsum("gib,bga->giba", right, left).reshape(
            self.m1 * self.m0, self.n0 * self.n1
        )


class Model(nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        torch.manual_seed(seed)
        hidden = int(arm[6:]) if arm.startswith("narrow") else 152
        self.arm = arm
        if arm in ("greedy", "balanced"):
            self.up = BTT(64, hidden, arm == "balanced")
            self.down = BTT(hidden, 64, arm == "balanced")
        elif arm == "blockshuffle":
            self.up = BlockShuffleLinear(64, hidden, 8)
            self.down = BlockShuffleLinear(hidden, 64, 8)
            for name, module in (("up", self.up), ("down", self.down)):
                module.initialize(seed, name)
                for layer in (module.first, module.second):
                    layer.weight.lr_scale = module.input_width / layer.weight.shape[-1] / 2
        else:
            self.up = nn.Linear(64, hidden, bias=False)
            self.down = nn.Linear(hidden, 64, bias=False)
            nn.init.normal_(self.up.weight, std=64**-0.5)
            nn.init.normal_(self.down.weight, std=hidden**-0.5)
        self.bias_up = nn.Parameter(torch.zeros(hidden))
        self.bias_down = nn.Parameter(torch.zeros(64))

    def forward(self, x):
        return self.down(F.gelu(self.up(x) + self.bias_up)) + self.bias_down

    def groups(self, lr):
        return [dict(params=[p], lr=lr * getattr(p, "lr_scale", 1.0)) for p in self.parameters()]
