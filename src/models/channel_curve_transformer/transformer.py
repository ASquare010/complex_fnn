"""Selected Curve-Wide Transformer: full-width mixing and three learned self-gated curves."""

import math

import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

from models.channel_curve_transformer.kernels import backward as cuda_backward
from models.channel_curve_transformer.kernels import forward as cuda_forward
from models.components import Attention, RMSNorm, counts, initialize, parameter_generator


def calibration(width, nodes=128):
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


def ordinary_curves(z, a, b, c, e, mean, scale):
    work_dtype = torch.float64 if z.dtype == torch.float64 else torch.float32
    z = z.to(work_dtype)
    value = torch.zeros_like(z)
    for k in range(3):
        value = value + F.silu(a[k] * z + b[k]) * (c[k] * z + e[k])
    return scale * (value / math.sqrt(3) - mean)


class Curves(Function):
    @staticmethod
    def forward(ctx, z, a, b, c, e, mean, scale):
        ctx.save_for_backward(z, a, b, c, e)
        ctx.scale = scale
        if z.is_cuda and z.dtype != torch.float64:
            return cuda_forward(z, a, b, c, e, True, mean, scale)
        return ordinary_curves(z, a, b, c, e, mean, scale).to(z.dtype)

    @staticmethod
    def backward(ctx, gradient):
        original, a, b, c, e = ctx.saved_tensors
        if original.is_cuda and original.dtype != torch.float64:
            return (
                *cuda_backward(original, gradient, a, b, c, e, True, ctx.scale),
                None,
                None,
            )
        dtype = torch.float64 if original.dtype == torch.float64 else torch.float32
        z = original.to(dtype)
        t = gradient.to(dtype) * (ctx.scale / math.sqrt(3))
        dz = torch.zeros_like(z)
        parameter_gradients = [[] for _ in range(4)]
        reduce_axes = tuple(range(z.ndim - 1))
        for k in range(3):
            u = a[k] * z + b[k]
            gate = c[k] * z + e[k]
            sigmoid = u.sigmoid()
            feature = u * sigmoid
            derivative = sigmoid * (1 + u * (1 - sigmoid))
            local = t * a[k] * derivative * gate
            remote = t * feature * c[k]
            dz.add_(local + remote)
            for collection, value in zip(
                parameter_gradients,
                (
                    t * z * derivative * gate,
                    t * derivative * gate,
                    t * feature * z,
                    t * feature,
                ),
            ):
                collection.append(value.sum(reduce_axes))
        return (
            dz.to(original.dtype),
            *(torch.stack(v) for v in parameter_gradients),
            None,
            None,
        )


class CurveFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.width = config.width
        self.up = nn.Linear(config.width, config.width, bias=False)
        self.down = nn.Linear(config.width, config.width, bias=False)
        self.initial_slopes = [0.5, 1.0, 1.5]
        self.initial_offsets = [-1.0, 0.0, 1.0]
        templates = torch.tensor(self.initial_slopes)[:, None].expand(3, config.width).clone()
        offsets = torch.tensor(self.initial_offsets)[:, None].expand(3, config.width).clone()
        self.a = nn.Parameter(templates)
        self.b = nn.Parameter(offsets)
        self.c = nn.Parameter(torch.ones(3, config.width))
        self.e = nn.Parameter(torch.zeros(3, config.width))
        self.initial_mean, self.scale = calibration(config.width)
        self.projection_parameters = 2 * config.width**2

    def forward(self, x):
        z = self.up(x)
        value = Curves.apply(z, self.a, self.b, self.c, self.e, self.initial_mean, self.scale)
        return self.down(value)

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
