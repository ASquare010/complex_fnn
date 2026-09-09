"""Small learned residual activations on calibrated dense and BlockShuffle FFNs."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.blockshuffle_ffn import BlockShuffleFFN
from src.dense_ffn import DenseFFN


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


class ShiftedBezierActivation(LearnableActivation):
    """Four distinct sigmoid coordinates, with bounded quadratic control points."""

    knots = 4

    def __init__(self, hidden: int, groups: int = 8) -> None:
        super().__init__(hidden, groups)
        self.theta_controls = nn.Parameter(torch.zeros(groups, self.knots, 2))
        self.theta_centers = nn.Parameter(torch.zeros(self.knots))
        self.theta_slopes = nn.Parameter(torch.zeros(self.knots))
        self.register_buffer("anchors", torch.linspace(-2, 2, self.knots), persistent=False)

    def coordinates(self):
        centers = self.anchors + 0.5 * self.theta_centers.tanh()
        slopes = 1 + 0.5 * self.theta_slopes.tanh()
        return centers, slopes

    def residual(self, z):
        centers, slopes = self.coordinates()
        t = ((z.unsqueeze(-1) - centers) * slopes).sigmoid()
        a, b = self.theta_controls.tanh().unbind(-1)
        curves = 2 * (1 - t) * t * a.unsqueeze(-2) + t.square() * b.unsqueeze(-2)
        return self.amplitude * curves.mean(-1)

    def shape_parameters(self):
        centers, slopes = self.coordinates()
        return {
            "family": "shifted_bezier",
            "amplitude": self.amplitude,
            "controls": self.theta_controls.tanh().detach().tolist(),
            "centers": centers.detach().tolist(),
            "slopes": slopes.detach().tolist(),
        }


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


class AffineResidualActivation(LearnableActivation):
    """A learned linear and constant correction, shared within each channel group."""

    def __init__(self, hidden: int, groups: int = 8) -> None:
        super().__init__(hidden, groups)
        self.theta_gain = nn.Parameter(torch.zeros(groups))
        self.theta_bias = nn.Parameter(torch.zeros(groups))

    def residual(self, z):
        gain = self.theta_gain.tanh().unsqueeze(-1)
        bias = self.theta_bias.tanh().unsqueeze(-1)
        return self.amplitude * (gain * z + bias)

    def shape_parameters(self):
        return {
            "family": "affine_residual",
            "amplitude": self.amplitude,
            "gain": (self.amplitude * self.theta_gain.tanh()).detach().tolist(),
            "bias": (self.amplitude * self.theta_bias.tanh()).detach().tolist(),
        }


def activation(hidden: int, groups: int, family: str) -> LearnableActivation:
    if family == "affine":
        return AffineResidualActivation(hidden, groups)
    if family == "shifted":
        return ShiftedBezierActivation(hidden, groups)
    if family == "rational":
        return RationalResidualActivation(hidden, groups)
    raise ValueError(f"Unknown learnable activation family: {family}")


class LearnableDenseFFN(DenseFFN):
    """Preserve dense parameter names and width calibration; adapt the gate shape."""

    def __init__(self, width: int, hidden: int, groups: int, family: str) -> None:
        super().__init__(width, hidden, "swiglu")
        self.curve = activation(hidden, groups, family)

    def forward(self, x):
        return self.down(self.curve(self.up(x)) * self.gate(x))


class LearnableBlockShuffleFFN(BlockShuffleFFN):
    """Preserve factor initialization; recompute only activation and product."""

    def __init__(self, width: int, hidden: int, groups: int, family: str) -> None:
        super().__init__(width, hidden, groups)
        self.curve = activation(hidden, groups, family)

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
