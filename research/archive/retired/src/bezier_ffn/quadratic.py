"""Direct quadratic Bezier activations and a controlled three-branch interaction."""

import torch
from torch import nn
from torch.nn import functional as F

from src.dense_ffn import DenseFFN


def quadratic_bezier(t: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    """B(t) with P0=0. The geometric interval is 0 <= t <= 1."""
    return 2 * (1 - t) * t * c1 + t.square() * c2


class QuadraticBezierActivation(nn.Module):
    """One learned quadratic per channel group, with t=sigmoid(preactivation).

    Direct mode starts at 4t^2-2t. Residual mode starts exactly at SiLU.
    The sigmoid domain and bounded controls give an upper derivative bound,
    not a positive lower bound: direct-mode tail gradients vanish.
    """

    def __init__(self, hidden: int, groups: int = 1, residual: bool = False) -> None:
        super().__init__()
        if hidden <= 0 or groups <= 0 or hidden % groups:
            raise ValueError("Positive groups must divide hidden width")
        self.hidden, self.groups, self.residual = hidden, groups, residual
        self.theta = nn.Parameter(torch.zeros(groups, 2))

    def controls(self) -> torch.Tensor:
        delta = 0.5 * self.theta.tanh()
        return delta if self.residual else delta + delta.new_tensor([-1.0, 2.0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.hidden:
            raise ValueError("Activation width mismatch")
        # FP32 avoids cancellation around zero in direct mode under BF16 AMP.
        z = x if x.dtype == torch.float64 else x.float()
        t = z.reshape(*z.shape[:-1], self.groups, self.hidden // self.groups).sigmoid()
        controls = self.controls()
        q = quadratic_bezier(t, controls[:, 0, None], controls[:, 1, None])
        q = q.reshape_as(x).to(x.dtype)
        return F.silu(x) + q if self.residual else q

    @torch.no_grad()
    def curves(self, grid: torch.Tensor) -> torch.Tensor:
        return self(grid[:, None].expand(-1, self.hidden))[:, :: self.hidden // self.groups]


class QuadraticBezierFFN(DenseFFN):
    """Dense projections with a direct curve, optional branch products, or SwiGLU correction.

    With three groups, branch i can add lambda_i * q_j * q_k at matching
    feature indices. Zero initial lambdas give exact unmixed containment.
    Projection names preserve common initialization and dense width calibration.
    """

    def __init__(
        self, width: int, hidden: int, groups: int = 1, gated: bool = False, mixed: bool = False
    ) -> None:
        if mixed and (groups != 3 or gated):
            raise ValueError("Product mixing requires three ungated branches")
        super().__init__(width, hidden, "swiglu" if gated else "gelu")
        self.curve = QuadraticBezierActivation(hidden, groups, residual=gated)
        self.mix_theta = nn.Parameter(torch.zeros(3)) if mixed else None

    def features(self, z: torch.Tensor) -> torch.Tensor:
        q = self.curve(z)
        if self.mix_theta is None:
            return q
        branches = q.reshape(*q.shape[:-1], 3, self.curve.hidden // 3)
        coefficient = (0.5 * self.mix_theta.tanh()).to(q.dtype)[:, None]
        products = branches.roll(1, -2) * branches.roll(-1, -2)
        return (branches + coefficient * products).reshape_as(q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(self.up(x))
        if self.gate is not None:
            z = z * self.gate(x)
        return self.down(z)
