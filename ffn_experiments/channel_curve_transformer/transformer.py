"""Complete Transformer with three coupled learned curves per channel."""

import math
from dataclasses import replace

import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

from ffn_experiments.channel_curve_transformer.kernels import backward as cuda_backward
from ffn_experiments.channel_curve_transformer.kernels import forward as cuda_forward
from ffn_experiments.components import (
    Attention,
    DenseFFN,
    RMSNorm,
    counts,
    initialize,
    parameter_generator,
)


def calibration(width, self_gate=False, nodes=128, wide=False):
    points, weights = np.polynomial.hermite.hermgauss(nodes)
    z = torch.from_numpy(points) * math.sqrt(2 * width * 0.02**2)
    w = torch.from_numpy(weights) / math.sqrt(math.pi)
    slopes = torch.tensor([0.5, 1.0, 1.5] if wide else [0.75, 1.0, 1.25], dtype=torch.float64)
    offsets = torch.tensor([-1.0, 0.0, 1.0] if wide else [-0.5, 0.0, 0.5], dtype=torch.float64)
    branches = F.silu(slopes[:, None] * z + offsets[:, None])
    variance = width * 0.02**2
    target = (1376 / 512) * variance * float((w * F.silu(z).square()).sum())
    if self_gate:
        raw = z * branches.sum(0) / math.sqrt(3)
        mean = float((raw * w).sum())
        initial_variance = float(((raw - mean).square() * w).sum())
    else:
        mean = 0.0
        initial_variance = variance / 3 * float((branches.square() * w).sum())
    return mean, math.sqrt(target / initial_variance)


def ordinary_curves(z, a, b, c, e, self_gate, mean, scale):
    work_dtype = torch.float64 if z.dtype == torch.float64 else torch.float32
    z = z.to(work_dtype)
    value = torch.zeros_like(z)
    for k, shift in enumerate((1, 17, 129)):
        neighbor = z if self_gate else torch.roll(z, -shift, -1)
        value = value + F.silu(a[k] * z + b[k]) * (c[k] * neighbor + e[k])
    return scale * (value / math.sqrt(3) - mean)


class Curves(Function):
    @staticmethod
    def forward(ctx, z, a, b, c, e, self_gate, mean, scale):
        ctx.save_for_backward(z, a, b, c, e)
        ctx.self_gate, ctx.scale = self_gate, scale
        if z.is_cuda and z.dtype != torch.float64:
            return cuda_forward(z, a, b, c, e, self_gate, mean, scale)
        return ordinary_curves(z, a, b, c, e, self_gate, mean, scale).to(z.dtype)

    @staticmethod
    def backward(ctx, gradient):
        original, a, b, c, e = ctx.saved_tensors
        if original.is_cuda and original.dtype != torch.float64:
            return (
                *cuda_backward(original, gradient, a, b, c, e, ctx.self_gate, ctx.scale),
                None,
                None,
                None,
            )
        dtype = torch.float64 if original.dtype == torch.float64 else torch.float32
        z = original.to(dtype)
        t = gradient.to(dtype) * (ctx.scale / math.sqrt(3))
        dz = torch.zeros_like(z)
        parameter_gradients = [[] for _ in range(4)]
        reduce_axes = tuple(range(z.ndim - 1))
        for k, shift in enumerate((1, 17, 129)):
            neighbor = z if ctx.self_gate else torch.roll(z, -shift, -1)
            u = a[k] * z + b[k]
            gate = c[k] * neighbor + e[k]
            sigmoid = u.sigmoid()
            feature = u * sigmoid
            derivative = sigmoid * (1 + u * (1 - sigmoid))
            local = t * a[k] * derivative * gate
            remote = t * feature * c[k]
            dz.add_(local + (remote if ctx.self_gate else torch.roll(remote, shift, -1)))
            for collection, value in zip(
                parameter_gradients,
                (
                    t * z * derivative * gate,
                    t * derivative * gate,
                    t * feature * neighbor,
                    t * feature,
                ),
            ):
                collection.append(value.sum(reduce_axes))
        return (
            dz.to(original.dtype),
            *(torch.stack(v) for v in parameter_gradients),
            None,
            None,
            None,
        )


class CurveFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.width = config.width
        self.mode = config.ffn
        self.up = nn.Linear(config.width, config.width, bias=False)
        self.down = nn.Linear(config.width, config.width, bias=False)
        wide = self.mode == "self_curve_wide"
        self.initial_slopes = [0.5, 1.0, 1.5] if wide else [0.75, 1.0, 1.25]
        self.initial_offsets = [-1.0, 0.0, 1.0] if wide else [-0.5, 0.0, 0.5]
        templates = torch.tensor(self.initial_slopes)[:, None].expand(3, config.width).clone()
        offsets = torch.tensor(self.initial_offsets)[:, None].expand(3, config.width).clone()
        if self.mode == "fixed_shape":
            self.register_buffer("a", templates)
            self.register_buffer("b", offsets)
        else:
            self.a = nn.Parameter(templates)
            self.b = nn.Parameter(offsets)
        self.c = nn.Parameter(torch.ones(3, config.width))
        self.e = nn.Parameter(torch.zeros(3, config.width))
        self.initial_mean, self.scale = calibration(
            config.width, self.mode.startswith("self_curve"), wide=wide
        )
        self.projection_parameters = 2 * config.width**2
        self.remove_exchange = self.restore_shapes = self.remove_even = False

    def forward(self, x):
        z = self.up(x)
        a, b = self.a, self.b
        if self.restore_shapes:
            a = (
                torch.tensor(self.initial_slopes, device=z.device, dtype=a.dtype)[:, None]
                .expand_as(a)
                .contiguous()
            )
            b = (
                torch.tensor(self.initial_offsets, device=z.device, dtype=b.dtype)[:, None]
                .expand_as(b)
                .contiguous()
            )
        value = Curves.apply(
            z,
            a,
            b,
            self.c,
            self.e,
            self.mode.startswith("self_curve") or self.remove_exchange,
            self.initial_mean,
            self.scale,
        )
        if self.remove_even:
            value = value.clone()
            value[..., ::2] = 0
        return self.down(value)

    @torch.no_grad()
    def initialize(self, seed, prefix, residual_scale):
        self.up.weight.normal_(0, 0.02, generator=parameter_generator(seed, f"{prefix}.up.weight"))
        self.down.weight.normal_(
            0, 0.02 * residual_scale, generator=parameter_generator(seed, f"{prefix}.down.weight")
        )


class PlainFFN(DenseFFN):
    def forward(self, x):
        return self.down(F.silu(self.up(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.width)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.width)
        self.ffn = (
            PlainFFN(replace(config, ffn="gelu")) if config.ffn == "plain" else CurveFFN(config)
        )

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
