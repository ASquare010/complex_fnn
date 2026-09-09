"""H099: biased conventional controls and four shared parity-mixing parameters."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

TASKS = ("quadratic", "cubic", "product", "piecewise")
FORMS = (
    "full_gelu",
    "full_swiglu",
    "full_relu2",
    "narrow_gelu",
    "narrow_swiglu",
    "narrow_relu2",
    "starrelu",
    "even_rational",
    "fixed_mix",
    "learned_mix",
    "feature_aware",
)
CONTROLS = (
    "narrow_gelu",
    "narrow_swiglu",
    "narrow_relu2",
    "starrelu",
    "even_rational",
    "fixed_mix",
)


def target_features(q, task):
    if task == "quadratic":
        return (q.square() - 1) / math.sqrt(2)
    if task == "cubic":
        return q * (q.square() - 3) / math.sqrt(6)
    if task == "product":
        return q * torch.roll(q, -1, -1)
    if task == "piecewise":
        return (F.relu(q).square() - 0.5 - math.sqrt(2 / math.pi) * q) / math.sqrt(
            1.25 - 2 / math.pi
        )
    raise ValueError(task)


def rational(z, theta):
    work = z if z.dtype == torch.float64 else z.float()
    groups = theta.numel()
    q = work.reshape(*z.shape[:-1], groups, -1)
    s = q / (1 + q.abs())
    a = 2 * theta.tanh().unsqueeze(-1)
    return (q * s * (1 + a * s)).reshape_as(z).to(z.dtype)


class Curve(nn.Module):
    def __init__(self, form):
        super().__init__()
        self.form = form
        if form == "learned_mix":
            self.theta = nn.Parameter(torch.full((4,), math.atanh(0.5)))
        elif form in ("even_rational", "fixed_mix"):
            self.register_buffer(
                "theta", torch.full((1,), 0.0 if form == "even_rational" else math.atanh(0.5))
            )
        elif form == "starrelu":
            self.scale = nn.Parameter(torch.ones(()))
            self.offset = nn.Parameter(torch.zeros(()))

    def forward(self, z):
        work = z if z.dtype == torch.float64 else z.float()
        if hasattr(self, "theta"):
            return rational(z, self.theta)
        if self.form == "starrelu":
            value = self.scale * F.relu(work).square() + self.offset
        elif "relu2" in self.form:
            value = F.relu(work).square()
        elif "gelu" in self.form:
            value = F.gelu(work)
        else:
            value = work
        return value.to(z.dtype)

    def shape_parameters(self):
        if hasattr(self, "theta"):
            return {"family": self.form, "alpha": (2 * self.theta.tanh()).detach().cpu().tolist()}
        if self.form == "starrelu":
            return {"family": self.form, "scale": self.scale.item(), "offset": self.offset.item()}
        return {"family": self.form}


class ResidualTaskFFN(nn.Module):
    def __init__(self, form, seed=17, task="quadratic"):
        super().__init__()
        if form not in FORMS or task not in TASKS:
            raise ValueError((form, task))
        self.form, self.task = form, task
        self.hidden = (
            16
            if form == "feature_aware"
            else (
                1024
                if form == "full_swiglu"
                else 1536
                if form.startswith("full_")
                else 304
                if form == "narrow_swiglu"
                else 456
            )
        )
        self.up = None if form == "feature_aware" else nn.Linear(384, self.hidden)
        self.gate = nn.Linear(384, self.hidden) if "swiglu" in form else None
        self.down = nn.Linear(self.hidden, 384, bias=form != "feature_aware")
        self.curve = Curve(form)
        if form == "feature_aware":
            self.register_buffer(
                "indices", torch.randperm(384, generator=torch.Generator().manual_seed(9861))[:16]
            )
        with torch.no_grad():
            for name in ("up", "gate"):
                projection = getattr(self, name)
                if projection is None:
                    continue
                digest = hashlib.sha256(f"{seed}:{name}.weight".encode()).digest()
                generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
                projection.weight.normal_(0, 1 / math.sqrt(384), generator=generator)
                bias_digest = hashlib.sha256(f"{seed}:{name}.bias".encode()).digest()
                bias_generator = torch.Generator().manual_seed(
                    int.from_bytes(bias_digest[:8], "little")
                )
                projection.bias.uniform_(-1, 1, generator=bias_generator)
            self.down.weight.zero_()
            if self.down.bias is not None:
                self.down.bias.zero_()

    def features(self, x):
        if self.up is None:
            z = x[..., self.indices]
            return z, target_features(z, self.task)
        z = self.up(x)
        if self.gate is not None:
            work = z if z.dtype == torch.float64 else z.float()
            value = (F.silu(work) * self.gate(x).to(work.dtype)).to(z.dtype)
        else:
            value = self.curve(z)
        return z, value

    def forward(self, x):
        return self.down(self.features(x)[1])

    def optimizer_groups(self, rate):
        multiplier = (
            1
            if self.form.startswith("full_") or self.form == "feature_aware"
            else ((1024 if self.gate is not None else 1536) / self.hidden)
        )
        return [
            {
                "params": [p],
                "lr": rate * (multiplier if name == "down.weight" else 1),
                "parameter_name": name,
            }
            for name, p in self.named_parameters()
        ]


def counts(form):
    model = ResidualTaskFFN(form)
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "activation_parameters": sum(p.numel() for p in model.curve.parameters()),
        "matrix_parameters": sum(p.numel() for p in model.parameters() if p.ndim == 2),
        "bias_parameters": sum(
            p.numel() for name, p in model.named_parameters() if name.endswith(".bias")
        ),
    }
