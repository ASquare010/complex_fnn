"""Layer-specific affine scalar gates over a shared dense feature bank."""

import torch
from torch import nn
from torch.nn import functional as F

from src.dense_ffn import DenseFFN


class AffineSharedFFN(nn.Module):
    """Three bounded vectors specialize shared SwiGLU features at each depth."""

    def __init__(self, base: DenseFFN) -> None:
        super().__init__()
        if base.gate is None:
            raise ValueError("This candidate requires the SwiGLU control")
        self.base = base
        hidden = base.up.out_features
        self.scale_delta = nn.Parameter(torch.zeros(hidden))
        self.shift_delta = nn.Parameter(torch.zeros(hidden))
        self.output_delta = nn.Parameter(torch.zeros(hidden))

    @property
    def up(self):
        return self.base.up

    def scalar_gate(self, z: torch.Tensor) -> torch.Tensor:
        a = (0.5 * self.scale_delta.tanh()).to(z.dtype)
        b = self.shift_delta.tanh().to(z.dtype)
        return F.silu(z + a * z + b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.base.up(x)
        v = self.base.gate(x)
        y = self.scalar_gate(u) * v
        c = (0.5 * self.output_delta.tanh()).to(y.dtype)
        return self.base.down(y + c * y)

    @torch.no_grad()
    def curves(self, grid: torch.Tensor) -> torch.Tensor:
        """Scalar gate curves with the value projection fixed to one."""
        z = grid[:, None].expand(-1, self.scale_delta.numel())
        gate = self.scalar_gate(z)
        return gate * (1 + 0.5 * self.output_delta.tanh())
