"""H092: retain input coordinates while learning a compact nonlinear basis."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

PAIRS = ("rational", "hermite", "trig", "duplicate", "linear", "antipodal")
FORMS = (*PAIRS, "raw_gelu", "core_linear")


def pair_basis(z, kind):
    w = z if z.dtype == torch.float64 else z.float()
    if kind in ("rational", "duplicate"):
        s = w / (1 + w.abs())
        even = w * s
        odd = w * s.square() if kind == "rational" else even
    elif kind == "hermite":
        even = (w.square() - 1) / math.sqrt(2)
        odd = w * (w.square() - 3) / math.sqrt(6)
    elif kind == "trig":
        even, odd = 1 - w.cos(), w.sin() - w
    elif kind == "linear":
        even, odd = w, w
    elif kind == "antipodal":
        even, odd = F.gelu(w), F.gelu(-w)
    else:
        raise ValueError(kind)
    return torch.cat((even, odd), -1).to(z.dtype)


class InputBasisFFN(nn.Module):
    def __init__(self, form, seed=17):
        super().__init__()
        if form not in FORMS:
            raise ValueError(form)
        self.form = form
        self.hidden = 0 if form == "core_linear" else 264 if form == "raw_gelu" else 176
        self.extra = 0 if not self.hidden else self.hidden * (1 if form == "raw_gelu" else 2)
        self.up = nn.Linear(384, self.hidden, bias=False) if self.hidden else None
        self.down = nn.Linear(384 + self.extra, 384, bias=False)
        with torch.no_grad():
            for name in ("up", "down"):
                projection = getattr(self, name)
                if projection is None:
                    continue
                digest = hashlib.sha256(f"{seed}:{name}.weight".encode()).digest()
                generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
                rows = self.hidden if name == "up" else 384
                initial = torch.empty(rows, 384).normal_(0, 1 / math.sqrt(384), generator=generator)
                projection.weight.zero_()
                projection.weight[:, :384].copy_(initial)

    def features(self, x):
        if self.up is None:
            return x, x
        z = self.up(x)
        if self.form == "raw_gelu":
            work = z if z.dtype == torch.float64 else z.float()
            extra = F.gelu(work).to(z.dtype)
        else:
            extra = pair_basis(z, self.form)
        return z, torch.cat((x.to(z.dtype), extra), -1)

    def forward(self, x):
        return self.down(self.features(x)[1])
