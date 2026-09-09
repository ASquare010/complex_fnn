"""H082: identity-initialized, group-shared curves inside BlockShuffle factors."""

import math
from types import MethodType

import torch
from torch import nn

from results.ungated_fit_v1.source import study as base
from src.blockshuffle_ffn import unshuffle_channels
from src.core.structured_linear import shuffle_channels

ORIGINAL_MAKE_MODEL = base.make_model
FORMS = ("plain", "latent_affine", "latent_tanh", "latent_sine", "narrow_swiglu",
         "narrow_gelu", "full_swiglu", "full_gelu")
HIDDEN = dict(zip(FORMS, (2048, 2048, 2048, 2048, 304, 456, 1024, 1536)))
COUNTS = dict(zip(FORMS, (350208, 350256, 350256, 350256, 350208, 350208, 1179648, 1179648)))


def curve(z, theta_a, theta_b, family):
    groups = theta_a.numel()
    work = z if z.dtype == torch.float64 else z.float()
    work = work.reshape(*z.shape[:-1], groups, z.shape[-1] // groups)
    a = 0.5 * theta_a.tanh().unsqueeze(-1)
    if family == "affine":
        correction = a * work + 0.5 * theta_b.tanh().unsqueeze(-1)
    else:
        scale = (math.log(4) * theta_b.tanh()).exp().unsqueeze(-1)
        argument = work / scale
        q = argument.tanh() if family == "tanh" else argument.sin()
        correction = a * scale * q.square()
    return z + correction.reshape_as(z).to(z.dtype)


def slope(z, theta_a, theta_b, family):
    groups = theta_a.numel()
    work = z.reshape(*z.shape[:-1], groups, z.shape[-1] // groups)
    a = 0.5 * theta_a.tanh().unsqueeze(-1)
    if family == "affine":
        result = torch.ones_like(work) * (1 + a)
    else:
        scale = (math.log(4) * theta_b.tanh()).exp().unsqueeze(-1)
        u = work / scale
        if family == "tanh":
            t = u.tanh()
            result = 1 + 2 * a * t * (1 - t.square())
        else:
            result = 1 + a * (2 * u).sin()
    return result.reshape_as(z)


class LatentCurve(nn.Module):
    def __init__(self, width=384, groups=8, family="tanh"):
        super().__init__()
        if width <= 0 or groups <= 0 or width % groups or family not in ("affine", "tanh", "sine"):
            raise ValueError("Valid family and positive dividing groups are required")
        self.width, self.groups, self.family = width, groups, family
        self.theta_a = nn.Parameter(torch.zeros(groups))
        self.theta_b = nn.Parameter(torch.zeros(groups))

    def forward(self, z):
        if z.ndim < 1 or z.shape[-1] != self.width:
            raise ValueError("Curve input must match its intermediate width")
        return curve(z, self.theta_a, self.theta_b, self.family)


def projected_forward(self, x):
    latent = self.curve(self.first(x))
    return unshuffle_channels(self.second(shuffle_channels(latent, self.groups)), self.groups)


def make_model(form, seed):
    model = ORIGINAL_MAKE_MODEL("plain" if form.startswith("latent_") else form, seed)
    if form.startswith("latent_"):
        for projection in (model.up, model.gate, model.down):
            projection.curve = LatentCurve(family=form.removeprefix("latent_"))
            projection.forward = MethodType(projected_forward, projection)
    assert sum(p.numel() for p in model.parameters()) == COUNTS[form]
    return model
