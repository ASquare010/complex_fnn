"""H088: coupled pairs and minimal shared scalar geometry at a fixed FFN budget."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

FORMS = (
    "full_gelu",
    "full_relu",
    "full_swiglu",
    "narrow_gelu",
    "narrow_relu",
    "narrow_swiglu",
    "groupsort",
    "gelu_offset",
    "twist_fixed",
    "twist_learned",
    "twist_identity",
    "bezier_1p",
    "bump_2p",
    "linear",
)
CANDIDATES = ("twist_learned", "twist_identity", "bezier_1p", "bump_2p")
CONTROLS = ("narrow_gelu", "narrow_relu", "narrow_swiglu", "groupsort", "gelu_offset")
EXTRAS = {
    name: (
        8
        if name == "bump_2p"
        else 1
        if name == "bezier_1p"
        else 4
        if name in ("twist_learned", "twist_identity", "gelu_offset")
        else 0
    )
    for name in FORMS
}
HIDDEN = {
    name: (
        1024
        if name == "full_swiglu"
        else 1536
        if name.startswith("full_")
        else 304
        if name == "narrow_swiglu"
        else 456
    )
    for name in FORMS
}
COUNTS = {name: (1179648 if name.startswith("full_") else 350208) + EXTRAS[name] for name in FORMS}


def twist(z, theta, center=1.0):
    """Rational rotation about (center,0); theta's length determines sharing."""
    groups = theta.numel()
    if z.ndim == 0 or z.shape[-1] % (2 * groups):
        raise ValueError("The feature width must be divisible by twice the group count")
    work = z if z.dtype == torch.float64 else z.float()
    pairs = work.reshape(*z.shape[:-1], groups, -1, 2)
    q0, q1 = pairs[..., 0] - center, pairs[..., 1]
    radius_inverse = (1 + q0.square() + q1.square()).reciprocal()
    t = 2 * theta.tanh().unsqueeze(-1) * (1 - radius_inverse)
    scale = 2 * t / (1 + t.square())
    correction = torch.stack((-scale * (t * q0 + q1), scale * (q0 - t * q1)), -1)
    return z + correction.reshape_as(z).to(z.dtype)


class PairTwist(nn.Module):
    def __init__(self, groups=4, *, learned=True, identity=False, center=1.0):
        super().__init__()
        if groups <= 0:
            raise ValueError("Positive groups required")
        theta = torch.full((groups,), 0.0 if identity else math.atanh(0.5))
        if learned:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer("theta", theta)
        self.center = center

    def forward(self, z):
        return twist(z, self.theta, self.center)

    def shape_parameters(self):
        return {
            "family": "pair_twist",
            "amplitude": (2 * self.theta.tanh()).detach().tolist(),
            "center": self.center,
        }


class LocalCurve(nn.Module):
    def __init__(self, family, groups):
        super().__init__()
        if family not in ("bezier", "bump") or groups <= 0:
            raise ValueError((family, groups))
        self.family, self.groups = family, groups
        self.theta = nn.Parameter(torch.zeros(groups, 1 if family == "bezier" else 2))

    def forward(self, z):
        if z.shape[-1] % self.groups:
            raise ValueError("Groups must divide feature width")
        work = z if z.dtype == torch.float64 else z.float()
        work = work.reshape(*z.shape[:-1], self.groups, -1)
        u = work.clamp(0, 1)  # Bound only the residual basis; preserve the ReLU tail.
        basis = u * (1 - u)
        controls = self.theta.tanh()
        if self.family == "bezier":
            correction = controls[:, 0, None] * basis
        else:
            correction = (
                2 * basis.square() * (controls[:, 0, None] + controls[:, 1, None] * (2 * u - 1))
            )
        correction = torch.where((work > 0) & (work < 1), correction, 0)
        return F.relu(z) + correction.reshape_as(z).to(z.dtype)

    def shape_parameters(self):
        factor = 1 if self.family == "bezier" else 2
        return {
            "family": self.family,
            "coefficients": (factor * self.theta.tanh()).detach().tolist(),
        }


class GroupSort(nn.Module):
    def forward(self, z):
        pairs = z.reshape(*z.shape[:-1], -1, 2)
        return torch.stack((pairs.min(-1).values, pairs.max(-1).values), -1).reshape_as(z)


class GeluOffset(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(4))

    def forward(self, z):
        grouped = z.reshape(*z.shape[:-1], 4, -1)
        shift = (2 * self.theta.tanh()).unsqueeze(-1).to(z.dtype)
        return F.gelu(grouped + shift).reshape_as(z)

    def shape_parameters(self):
        return {"family": "gelu_offset", "offset": (2 * self.theta.tanh()).detach().tolist()}


class GeometryFFN(nn.Module):
    def __init__(self, form, seed=17):
        super().__init__()
        if form not in FORMS:
            raise ValueError(form)
        self.form, self.hidden = form, HIDDEN[form]
        self.up = nn.Linear(384, self.hidden, bias=False)
        self.down = nn.Linear(self.hidden, 384, bias=False)
        self.gate = nn.Linear(384, self.hidden, bias=False) if "swiglu" in form else None
        if form.startswith("twist_"):
            self.curve = PairTwist(learned=form != "twist_fixed", identity=form == "twist_identity")
        elif form == "bezier_1p":
            self.curve = LocalCurve("bezier", 1)
        elif form == "bump_2p":
            self.curve = LocalCurve("bump", 4)
        elif form == "groupsort":
            self.curve = GroupSort()
        elif form == "gelu_offset":
            self.curve = GeluOffset()
        elif "gelu" in form:
            self.curve = nn.GELU()
        elif "relu" in form:
            self.curve = nn.ReLU()
        else:
            self.curve = nn.Identity()
        with torch.no_grad():
            for name in ("up", "gate", "down"):
                projection = getattr(self, name)
                if projection is None:
                    continue
                digest = hashlib.sha256(f"{seed}:{name}.weight".encode()).digest()
                generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
                projection.weight.normal_(
                    0, 1 / math.sqrt(projection.in_features), generator=generator
                )
        assert sum(p.numel() for p in self.parameters()) == COUNTS[form]

    def features(self, x):
        z = self.up(x)
        return (z, F.silu(z) * self.gate(x)) if self.gate is not None else (z, self.curve(z))

    def forward(self, x):
        return self.down(self.features(x)[1])

    def optimizer_groups(self, base_rate):
        multiplier = (
            1.0
            if self.form.startswith("full_")
            else ((1024 if self.gate is not None else 1536) / self.hidden)
        )
        return [
            {
                "params": [p],
                "lr": base_rate * (multiplier if name == "down.weight" else 1),
                "parameter_name": name,
            }
            for name, p in self.named_parameters()
        ]
