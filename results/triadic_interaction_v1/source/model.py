"""H100: three learned projections with raw or bounded multiplicative interaction."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from results.nonlinear_residual_v1.source.model import TASKS, Curve, target_features

FORMS = (
    "full_gelu",
    "full_swiglu",
    "full_relu2",
    "narrow_gelu",
    "narrow_swiglu",
    "narrow_relu2",
    "starrelu",
    "fixed_mix",
    "cubic_ridge",
    "cp2",
    "cp3",
    "bounded_cp3",
    "feature_aware",
)
CANDIDATES = ("cp3", "bounded_cp3")
CONTROLS = (
    "narrow_gelu",
    "narrow_swiglu",
    "narrow_relu2",
    "starrelu",
    "fixed_mix",
    "cubic_ridge",
    "cp2",
)


def input_basis():
    return torch.linalg.qr(
        torch.randn(384, 384, dtype=torch.float64, generator=torch.Generator().manual_seed(9901))
    ).Q[:, :16]


def interaction(form, u, v=None, w=None):
    dtype = u.dtype
    u = u if dtype == torch.float64 else u.float()
    v = v if v is None or dtype == torch.float64 else v.float()
    w = w if w is None or dtype == torch.float64 else w.float()
    if form == "cubic_ridge":
        value = u * (u.square() - 3) / math.sqrt(6)
    elif form == "cp2":
        value = u * v
    elif form == "cp3":
        value = u * v * w
    elif form == "bounded_cp3":
        value = 4 * u * v * w / (1 + u.square() + v.square() + w.square())
    else:
        raise ValueError(form)
    return value.to(dtype)


class InteractionFFN(nn.Module):
    def __init__(self, form, seed=17, task="quadratic"):
        super().__init__()
        if form not in FORMS or task not in TASKS:
            raise ValueError((form, task))
        self.form, self.task = form, task
        self.hidden = (
            16
            if form == "feature_aware"
            else 1024
            if form == "full_swiglu"
            else 1536
            if form.startswith("full_")
            else 304
            if form in ("cp2", "narrow_swiglu")
            else 228
            if form in CANDIDATES
            else 456
        )
        self.up = None if form == "feature_aware" else nn.Linear(384, self.hidden)
        self.gate = (
            nn.Linear(384, self.hidden)
            if form in ("cp2", "cp3", "bounded_cp3", "full_swiglu", "narrow_swiglu")
            else None
        )
        self.third = nn.Linear(384, self.hidden) if form in CANDIDATES else None
        self.down = nn.Linear(self.hidden, 384, bias=form != "feature_aware")
        self.curve = Curve(form)
        if form == "feature_aware":
            self.register_buffer("basis", input_basis())
        with torch.no_grad():
            for name in ("up", "gate", "third"):
                projection = getattr(self, name)
                if projection is None:
                    continue
                for suffix, normal in (("weight", True), ("bias", False)):
                    digest = hashlib.sha256(f"{seed}:{name}.{suffix}".encode()).digest()
                    gen = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
                    value = getattr(projection, suffix)
                    if normal:
                        value.normal_(0, 1 / math.sqrt(384), generator=gen)
                    else:
                        value.uniform_(-1, 1, generator=gen)
            self.down.weight.zero_()
            if self.down.bias is not None:
                self.down.bias.zero_()

    def features(self, x):
        if self.form == "feature_aware":
            q = x @ self.basis.to(x.dtype)
            return q, target_features(q, self.task)
        u = self.up(x)
        if self.form in (*CANDIDATES, "cp2", "cubic_ridge"):
            v = self.gate(x) if self.gate is not None else None
            w = self.third(x) if self.third is not None else None
            return u, interaction(self.form, u, v, w)
        if "swiglu" in self.form:
            v = self.gate(x)
            work_u, work_v = (u, v) if u.dtype == torch.float64 else (u.float(), v.float())
            return u, (F.silu(work_u) * work_v).to(u.dtype)
        return u, self.curve(u)

    def forward(self, x):
        return self.down(self.features(x)[1])

    def optimizer_groups(self, rate):
        reference = 1024 if "swiglu" in self.form else 1536
        return [
            {
                "params": [p],
                "parameter_name": name,
                "lr": rate
                * (
                    reference / self.hidden
                    if name == "down.weight"
                    and not self.form.startswith("full_")
                    and self.form != "feature_aware"
                    else 1
                ),
            }
            for name, p in self.named_parameters()
        ]


def counts(form):
    model = InteractionFFN(form)
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "matrix_parameters": sum(p.numel() for p in model.parameters() if p.ndim == 2),
        "bias_parameters": sum(
            p.numel() for name, p in model.named_parameters() if name.endswith("bias")
        ),
        "activation_parameters": sum(p.numel() for p in model.curve.parameters()),
        "fixed_buffer_entries": sum(p.numel() for p in model.buffers()),
    }
