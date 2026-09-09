"""Native rational residual activation on the retained BlockShuffle candidate."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.blockshuffle_ffn import BlockShuffleFFN


class LearnableActivation(nn.Module):
    """Group-shared shapes; FP32 pointwise work with the input dtype restored."""

    amplitude = 0.25

    def __init__(self, hidden: int, groups: int) -> None:
        super().__init__()
        if hidden <= 0 or groups <= 0 or hidden % groups:
            raise ValueError("Positive activation groups must divide hidden width")
        self.hidden, self.groups = hidden, groups

    def residual(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.hidden:
            raise ValueError("Activation input must match hidden width")
        work = z if z.dtype == torch.float64 else z.float()
        work = work.reshape(*z.shape[:-1], self.groups, self.hidden // self.groups)
        correction = self.residual(work).reshape_as(z).to(z.dtype)
        # Casting before addition preserves baseline BF16 activation/product rounding.
        return F.silu(z) + correction

    def shape_parameters(self) -> dict:
        raise NotImplementedError


class RationalResidualActivation(LearnableActivation):
    """Bounded coefficients over (1, z, z^2, z^3)/(1 + beta*z^2)."""

    def __init__(self, hidden: int, groups: int = 8) -> None:
        super().__init__(hidden, groups)
        self.theta_coefficients = nn.Parameter(torch.zeros(groups, 4))
        self.theta_denominator = nn.Parameter(torch.zeros(groups))

    def residual(self, z):
        beta = (1 + 0.75 * self.theta_denominator.tanh()).unsqueeze(-1)
        a = self.theta_coefficients.tanh().unbind(-1)
        # Homogeneous rescaling avoids forming large z^2 or z^3 intermediates.
        w = (1 + z.abs()).reciprocal()
        u = z * w
        denominator = w.square() + beta * u.square()
        r0 = w.square() / denominator
        r1 = u * w / denominator
        r2 = u.square() / denominator
        r3 = z * r2
        return self.amplitude * sum(c.unsqueeze(-1) * r for c, r in zip(a, (r0, r1, r2, r3)))

    def shape_parameters(self):
        return {
            "family": "rational_residual",
            "amplitude": self.amplitude,
            "coefficients": self.theta_coefficients.tanh().detach().tolist(),
            "denominator_coefficient": (1 + 0.75 * self.theta_denominator.tanh()).detach().tolist(),
        }


class RationalBlockShuffleFFN(BlockShuffleFFN):
    """Preserve factor initialization; recompute only activation and product."""

    def __init__(self, width: int, hidden: int, groups: int) -> None:
        super().__init__(width, hidden, groups)
        self.curve = RationalResidualActivation(hidden, groups)

    def product(self, u, v):
        return self.curve(u) * v

    def forward(self, x):
        u, v = self.up(x), self.gate(x)
        if self.recompute_gate and self.training and torch.is_grad_enabled():
            if self.gate_recompute_method != "checkpoint":
                raise ValueError("Learnable activations require checkpoint gate recomputation")
            z = checkpoint(self.product, u, v, use_reentrant=False, preserve_rng_state=False)
        else:
            z = self.product(u, v)
        return self.down(z)
