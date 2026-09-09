"""Quadratic and cubic Bezier activation FFNs with bounded learned controls."""

import torch
from torch import nn
from torch.nn import functional as F

from .quadratic import QuadraticBezierActivation, QuadraticBezierFFN, quadratic_bezier

__all__ = [
    "BezierActivation",
    "BezierFFN",
    "QuadraticBezierActivation",
    "QuadraticBezierFFN",
    "quadratic_bezier",
]


class BezierActivation(nn.Module):
    """GELU plus a bounded cubic deformation, initialized exactly at GELU.

    Fixed X(t)=t; t=sigmoid(x). Each group owns two interior Y deltas.
    No root solving, gates, or extra projections. See research/theory.md.
    """

    def __init__(self, hidden: int, groups: int = 8, bound: float = 0.5) -> None:
        super().__init__()
        if hidden <= 0 or groups <= 0 or hidden % groups:
            raise ValueError("Positive group count must divide hidden width")
        if not 0 < bound <= 0.5:
            raise ValueError("The tested derivative bound requires 0 < bound <= .5")
        self.hidden, self.groups, self.bound = hidden, groups, bound
        self.theta = nn.Parameter(torch.zeros(groups, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.hidden:
            raise ValueError("Activation width mismatch")
        # Preserve FP64 for gradcheck and FP32 for stable curve arithmetic under AMP.
        z = x if x.dtype == torch.float64 else x.float()
        t = z.reshape(*z.shape[:-1], self.groups, self.hidden // self.groups).sigmoid()
        delta = self.bound * self.theta.tanh()
        d1, d2 = delta[:, 0, None], delta[:, 1, None]
        residual = 3 * t * (1 - t) * ((1 - t) * d1 + t * d2)
        return (F.gelu(x).to(z.dtype) + residual.reshape_as(z)).to(x.dtype)

    @torch.no_grad()
    def curves(self, grid: torch.Tensor) -> torch.Tensor:
        x = grid[:, None].expand(-1, self.hidden)
        return self(x)[:, :: self.hidden // self.groups]


class BezierFFN(nn.Module):
    def __init__(self, width: int, hidden: int, groups: int = 8) -> None:
        super().__init__()
        self.up = nn.Linear(width, hidden, bias=False)
        self.activation = BezierActivation(hidden, groups)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.activation(self.up(x)))
