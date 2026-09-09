"""H091: extra nonlinear features from part of a full-width projection."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

KINDS = ("duplicate", "identity", "rational", "hermite2", "hermite3", "sine")
FORMS = ("core_gelu", *KINDS)


def basis(z, kind):
    work = z if z.dtype == torch.float64 else z.float()
    if kind == "duplicate":
        value = F.gelu(work)
    elif kind == "identity":
        value = work
    elif kind == "rational":
        value = work * (work / (1 + work.abs()))
    elif kind == "hermite2":
        value = (work.square() - 1) / math.sqrt(2)
    elif kind == "hermite3":
        value = work * (work.square() - 3) / math.sqrt(6)
    elif kind == "sine":
        value = torch.sin(2 * work)
    else:
        raise ValueError(kind)
    return value.to(z.dtype)


class PartialCurve(nn.Module):
    def __init__(self, kind, expanded=144):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(kind)
        self.kind, self.expanded = kind, expanded

    def forward(self, z):
        return torch.cat((F.gelu(z), basis(z[..., :self.expanded], self.kind)), -1)


class PartialFFN(nn.Module):
    def __init__(self, form, seed=17):
        super().__init__()
        if form not in FORMS:
            raise ValueError(form)
        self.form, self.hidden = form, 384
        self.expanded = 0 if form == "core_gelu" else 144
        self.up = nn.Linear(384, 384, bias=False)
        self.down = nn.Linear(384 + self.expanded, 384, bias=False)
        self.curve = nn.GELU() if not self.expanded else PartialCurve(form)
        with torch.no_grad():
            for name in ("up", "down"):
                digest = hashlib.sha256(f"{seed}:{name}.weight".encode()).digest()
                generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
                initial = torch.empty(384, 384).normal_(0, 1/math.sqrt(384), generator=generator)
                projection = getattr(self, name).weight
                projection.zero_()
                projection[:, :384].copy_(initial)

    def features(self, x):
        z = self.up(x)
        return z, self.curve(z)

    def forward(self, x):
        return self.down(self.features(x)[1])
