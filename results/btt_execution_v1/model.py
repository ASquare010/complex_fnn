"""BTT execution adapters; retained H164 cores and normalization stay unchanged."""

import torch
from torch import nn
from torch.nn import functional as F

from results.btt_balance_v1.model import BTT


class Model(nn.Module):
    def __init__(self, width, hidden, seed, arm):
        super().__init__()
        torch.manual_seed(seed)
        self.arm = arm
        if arm == "dense":
            self.up = nn.Linear(width, hidden, bias=False)
            self.down = nn.Linear(hidden, width, bias=False)
            nn.init.normal_(self.up.weight, std=width**-0.5)
            nn.init.normal_(self.down.weight, std=hidden**-0.5)
        else:
            self.up = BTT(width, hidden, True)
            self.down = BTT(hidden, width, True)
        self.bias_up = nn.Parameter(torch.zeros(hidden))
        self.bias_down = nn.Parameter(torch.zeros(width))

    def project(self, layer, x):
        return x @ layer.dense() if self.arm == "materialized" else layer(x)

    def forward(self, x):
        return (
            self.project(self.down, F.gelu(self.project(self.up, x) + self.bias_up))
            + self.bias_down
        )

    def groups(self):
        return [dict(params=[p], lr=0.003 * getattr(p, "lr_scale", 1.0)) for p in self.parameters()]
