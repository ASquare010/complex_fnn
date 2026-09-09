"""Wide nonlinear features with grouped projection matrices."""

import torch
from torch import nn
from torch.nn import functional as F


class GroupedLinear(nn.Module):
    """Bias-free block diagonal projection using native batched GEMM."""

    def __init__(self, input_width: int, output_width: int, groups: int) -> None:
        super().__init__()
        if groups <= 0 or input_width % groups or output_width % groups:
            raise ValueError("Groups must divide both projection dimensions")
        self.input_width, self.output_width, self.groups = input_width, output_width, groups
        self.weight = nn.Parameter(
            torch.empty(groups, output_width // groups, input_width // groups)
        )
        nn.init.normal_(self.weight, std=0.02 * groups**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_width:
            raise ValueError("Grouped projection input dimension mismatch")
        grouped = x.reshape(-1, self.groups, self.input_width // self.groups).transpose(0, 1)
        y = torch.bmm(grouped, self.weight.transpose(1, 2)).transpose(0, 1)
        return y.reshape(*x.shape[:-1], self.output_width)


def shuffle_channels(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Interleave old groups before the following grouped projection."""
    return x.reshape(*x.shape[:-1], groups, x.shape[-1] // groups).transpose(-2, -1).reshape_as(x)


class StructuredFFN(nn.Module):
    def __init__(self, width: int, hidden: int, groups: int, gated: bool = False) -> None:
        super().__init__()
        self.groups = groups
        self.up = GroupedLinear(width, hidden, groups)
        self.down = GroupedLinear(hidden, width, groups)
        self.gate = GroupedLinear(width, hidden, groups) if gated else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.up(x)
        z = F.silu(z) * self.gate(x) if self.gate is not None else F.gelu(z)
        return self.down(shuffle_channels(z, self.groups))
