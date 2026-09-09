"""Fixed cross-group coupling inside a compressed multiplicative FFN."""

import math

import torch
from torch.nn import functional as F

from src.grouped_ffn import StructuredFFN, shuffle_channels


def mix_groups(x: torch.Tensor, groups: int, coefficient: float = 0.5) -> torch.Tensor:
    """Invertible cyclic mixing; the singular-value bound is in the model notes."""
    if groups <= 0 or x.shape[-1] % groups or abs(coefficient) >= 1:
        raise ValueError("Divisible groups and a mixing coefficient of magnitude below 1 required")
    grouped = x.reshape(*x.shape[:-1], groups, x.shape[-1] // groups)
    neighbor = grouped.roll(1, dims=-2).reshape_as(x)
    return (x + coefficient * neighbor) / math.sqrt(1 + coefficient**2)


class CoupledStructuredFFN(StructuredFFN):
    """Same matrices and count as grouped SwiGLU, with cross-group value inputs."""

    def __init__(self, width: int, hidden: int, groups: int) -> None:
        super().__init__(width, hidden, groups, gated=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.silu(self.up(x)) * self.gate(mix_groups(x, self.groups))
        return self.down(shuffle_channels(z, self.groups))
