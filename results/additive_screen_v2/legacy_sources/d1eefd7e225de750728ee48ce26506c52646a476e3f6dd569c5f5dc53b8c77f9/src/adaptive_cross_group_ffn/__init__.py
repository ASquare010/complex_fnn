"""Optional cross-group products using already-projected value features."""

import torch
from torch import nn
from torch.nn import functional as F

from src.grouped_ffn import StructuredFFN, shuffle_channels


class AdaptiveCoupledFFN(StructuredFFN):
    """Retain the uncoupled function class at zero; add per-channel neighbor values."""

    def __init__(self, width: int, hidden: int, groups: int) -> None:
        super().__init__(width, hidden, groups, gated=True)
        self.mix_theta = nn.Parameter(torch.zeros(hidden))

    def coefficients(self) -> torch.Tensor:
        return 0.75 * self.mix_theta.tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.gate(x)
        neighbor = (
            value.reshape(*value.shape[:-1], self.groups, -1).roll(1, dims=-2).reshape_as(value)
        )
        mix = self.coefficients().to(value.dtype)
        coupled = (value + mix * neighbor) * torch.rsqrt(1 + mix.square())
        z = F.silu(self.up(x)) * coupled
        return self.down(shuffle_channels(z, self.groups))

    @torch.no_grad()
    def curves(self, grid: torch.Tensor) -> torch.Tensor:
        """Neighbor-to-value response with own value fixed to zero; a linear slice."""
        mix = self.coefficients()
        return grid[:, None] * (mix * torch.rsqrt(1 + mix.square()))
