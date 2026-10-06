"""Selected Curve-Wide Transformer: full-width mixing and three learned self-gated curves."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.components import Attention, RMSNorm, counts, initialize, parameter_generator


class CurveFFN(nn.Module):
    """Mix features, apply learned curves, then mix features again."""

    def __init__(self, config):
        super().__init__()
        self.width = config.width
        self.up = nn.Linear(config.width, config.width, bias=False)
        self.down = nn.Linear(config.width, config.width, bias=False)
        self.initial_slopes = [0.5, 1.0, 1.5]
        self.initial_offsets = [-1.0, 0.0, 1.0]
        templates = torch.tensor(self.initial_slopes)[:, None].expand(3, config.width).clone()
        offsets = torch.tensor(self.initial_offsets)[:, None].expand(3, config.width).clone()
        # Keep parameter names compatible with the measured checkpoints.
        self.a = nn.Parameter(templates)
        self.b = nn.Parameter(offsets)
        self.c = nn.Parameter(torch.ones(3, config.width))
        self.e = nn.Parameter(torch.zeros(3, config.width))
        self.initial_mean, self.scale = calibration(config.width)
        self.projection_parameters = 2 * config.width**2

    def forward(self, x):
        z = self.up(x)  # Mix the token's features: width -> width.
        output_dtype = z.dtype
        # Keep curve arithmetic in FP32 during mixed-precision training.
        z = z if z.dtype == torch.float64 else z.float()

        value = torch.zeros_like(z)
        for branch in range(3):
            response = F.silu(self.a[branch] * z + self.b[branch])
            multiplier = self.c[branch] * z + self.e[branch]
            value = value + response * multiplier

        value = self.scale * (value / math.sqrt(3) - self.initial_mean)
        return self.down(value.to(output_dtype))  # Mix features back into the output.

    @torch.no_grad()
    def initialize(self, seed, prefix, residual_scale):
        self.up.weight.normal_(0, 0.02, generator=parameter_generator(seed, f"{prefix}.up.weight"))
        self.down.weight.normal_(
            0, 0.02 * residual_scale, generator=parameter_generator(seed, f"{prefix}.down.weight")
        )


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.width)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.width)
        self.ffn = CurveFFN(config)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class Model(nn.Module):
    model_name = "channel_curve_transformer"

    def __init__(self, config, seed=17):
        super().__init__()
        config.validate()
        if config.name != self.model_name:
            raise ValueError("Wrong model name")
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = RMSNorm(config.width)
        initialize(self, seed)

    def forward(self, tokens):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config.context:
            raise ValueError("Invalid token shape")
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        return F.linear(self.norm(x), self.embedding.weight)

    def counts(self):
        return counts(self)


def calibration(width, nodes=128):
    """Fixed starting mean and scale; preserve the measured initialization."""
    points, weights = np.polynomial.hermite.hermgauss(nodes)
    z = torch.from_numpy(points) * math.sqrt(2 * width * 0.02**2)
    w = torch.from_numpy(weights) / math.sqrt(math.pi)
    slopes = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    offsets = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    branches = F.silu(slopes[:, None] * z + offsets[:, None])
    variance = width * 0.02**2
    target = (1376 / 512) * variance * float((w * F.silu(z).square()).sum())
    raw = z * branches.sum(0) / math.sqrt(3)
    mean = float((raw * w).sum())
    initial_variance = float(((raw - mean).square() * w).sum())
    return mean, math.sqrt(target / initial_variance)
