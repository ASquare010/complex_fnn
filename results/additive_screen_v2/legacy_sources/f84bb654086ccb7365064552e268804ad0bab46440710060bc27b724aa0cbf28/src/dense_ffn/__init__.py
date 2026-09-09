"""Dense reference FFNs with shared projection names for controlled comparisons."""

import torch
from torch import nn
from torch.nn import functional as F


class DenseFFN(nn.Module):
    def __init__(self, width: int, hidden: int, activation: str = "gelu") -> None:
        super().__init__()
        if activation not in ("gelu", "relu", "silu", "swiglu"):
            raise ValueError(activation)
        self.activation = activation
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.gate = nn.Linear(width, hidden, bias=False) if activation == "swiglu" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.up(x)
        if self.gate is not None:
            z = F.silu(z) * self.gate(x)
        else:
            z = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}[self.activation](z)
        return self.down(z)
