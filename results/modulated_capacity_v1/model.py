"""Isolated output-modulated shared-feature candidate; no active factory change."""

from torch import nn
from torch.nn import functional as F


class ModulatedFFN(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.up = nn.Linear(d, h)
        self.base = nn.Linear(h, d)
        self.gate = nn.Linear(h, d)

    def forward(self, x):
        z = F.gelu(self.up(x))
        return self.base(z) + x * self.gate(z)
