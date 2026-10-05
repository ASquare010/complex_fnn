"""Complete Transformer: six scalar responses with grouped readouts."""

import math

import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

from ffn_experiments.components import Attention, RMSNorm, counts, initialize, parameter_generator


def calibration(width, self_gate=False, nodes=128):
    points, weights = np.polynomial.hermite.hermgauss(nodes)
    z = torch.from_numpy(points) * math.sqrt(2 * width * 0.02**2)
    w = torch.from_numpy(weights) / math.sqrt(math.pi)
    slopes = torch.tensor([0.75, 1.0, 1.25], dtype=torch.float64)
    offsets = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
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


def basis(z, a, b, groups, removal=0):
    """Group-major [groups,tokens,6*coordinates], product then scalar responses."""
    shape = z.shape
    z = z.reshape(-1, shape[-1])
    responses = F.silu(z[..., None] * a.T + b.T)
    products = z[..., None] * responses
    features = torch.cat((products, responses), -1)
    if removal == 1:
        features = features.clone()
        features[:, ::2] = 0
    elif removal == 2:
        features = torch.cat((products, torch.zeros_like(responses)), -1)
    elif removal == 3:
        features = torch.cat((torch.zeros_like(products), responses), -1)
    return features.reshape(z.shape[0], groups, -1).transpose(0, 1).contiguous()


def cached(z, a, b, matrix, mean, scale, removal=0):
    dtype = torch.float64 if z.dtype == torch.float64 else torch.float32
    with torch.autocast(z.device.type, enabled=False):
        features = basis(z.to(dtype), a, b, matrix.shape[0], removal)
        value = torch.bmm(features, matrix.transpose(1, 2))
        value = value.transpose(0, 1).reshape(z.shape)
        return (scale * (value - mean)).to(z.dtype)


class Readout(Function):
    @staticmethod
    def forward(ctx, z, a, b, matrix, mean, scale, removal):
        ctx.save_for_backward(z, a, b, matrix)
        ctx.scale, ctx.removal = scale, removal
        if z.is_cuda and z.dtype != torch.float64:
            from .kernels import forward

            with torch.autocast("cuda", enabled=False):
                features = forward(z, a, b, matrix.shape[0], removal)
                value = torch.bmm(features, matrix.transpose(1, 2)).transpose(0, 1).reshape(z.shape)
                return (scale * (value - mean)).to(z.dtype)
        return cached(z, a, b, matrix, mean, scale, removal)

    @staticmethod
    def backward(ctx, gradient):
        original, a, b, matrix = ctx.saved_tensors
        dtype = torch.float64 if original.dtype == torch.float64 else torch.float32
        groups, size, _ = matrix.shape
        with torch.autocast(original.device.type, enabled=False):
            z = original.to(dtype).reshape(-1, original.shape[-1])
            incoming = (
                (gradient.to(dtype) * ctx.scale)
                .reshape(-1, groups, size)
                .transpose(0, 1)
                .contiguous()
            )
            if original.is_cuda and original.dtype != torch.float64:
                from .kernels import forward

                features = forward(original, a, b, groups, ctx.removal)
            else:
                features = basis(z, a, b, groups, ctx.removal)
            dm = torch.bmm(incoming.transpose(1, 2), features)
            del features
            feature_gradient = torch.bmm(incoming, matrix)
            if original.is_cuda and original.dtype != torch.float64:
                from .kernels import adjoint

                dz, da, db = adjoint(
                    original,
                    feature_gradient,
                    a,
                    b,
                    ctx.removal,
                    ctx.needs_input_grad[1] or ctx.needs_input_grad[2],
                )
            else:
                t = feature_gradient.transpose(0, 1).reshape(z.shape[0], z.shape[1], 6)
                tp, tl = t[..., :3], t[..., 3:]
                if ctx.removal == 1:
                    tp = tp.clone()
                    tl = tl.clone()
                    tp[:, ::2] = 0
                    tl[:, ::2] = 0
                elif ctx.removal == 2:
                    tl = torch.zeros_like(tl)
                elif ctx.removal == 3:
                    tp = torch.zeros_like(tp)
                u = z[..., None] * a.T + b.T
                sigmoid = u.sigmoid()
                feature = u * sigmoid
                derivative = sigmoid * (1 + u * (1 - sigmoid))
                du = (tp * z[..., None] + tl) * derivative
                dz = (tp * feature + du * a.T).sum(-1).reshape(original.shape).to(original.dtype)
                da = (du * z[..., None]).sum(0).T.contiguous()
                db = du.sum(0).T.contiguous()
        return dz, da, db, dm, None, None, None


def initial_matrix(width, group_size, device=None, dtype=torch.float32):
    matrix = torch.zeros(
        width // group_size, group_size, 6 * group_size, device=device, dtype=dtype
    )
    index = torch.arange(group_size, device=device)
    for k in range(3):
        matrix[:, index, 6 * index + k] = 1 / math.sqrt(3)
    return matrix


class BasisFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mode, self.groups = config.ffn, config.groups
        self.width, self.group_size = config.width, config.width // config.groups
        self.up = nn.Linear(config.width, config.width, bias=False)
        self.down = nn.Linear(config.width, config.width, bias=False)
        a = torch.tensor([0.75, 1.0, 1.25])[:, None].expand(3, config.width).clone()
        b = torch.tensor([-0.5, 0.0, 0.5])[:, None].expand(3, config.width).clone()
        if self.mode == "learned_basis":
            self.a, self.b = nn.Parameter(a), nn.Parameter(b)
        else:
            self.register_buffer("a", a)
            self.register_buffer("b", b)
        if self.mode == "tied_basis":
            local = torch.zeros(config.width, 6)
            local[:, :3] = 1 / math.sqrt(3)
            self.local = nn.Parameter(local)
        else:
            self.matrix = nn.Parameter(initial_matrix(config.width, self.group_size))
        self.initial_mean, self.scale = calibration(config.width, True)
        self.projection_parameters = 2 * config.width**2 + 6 * config.width * self.group_size
        self.restore_readout = self.restore_shapes = False
        self.removal = 0

    def readout(self):
        if self.restore_readout:
            return initial_matrix(
                self.width, self.group_size, self.up.weight.device, self.up.weight.dtype
            )
        if self.mode != "tied_basis":
            return self.matrix
        matrix = self.local.new_zeros(self.groups, self.group_size, 6 * self.group_size)
        index = torch.arange(self.group_size, device=matrix.device)
        columns = 6 * index[:, None] + torch.arange(6, device=matrix.device)
        matrix[:, index[:, None], columns] = self.local.reshape(self.groups, self.group_size, 6)
        return matrix

    def forward(self, x):
        z = self.up(x)
        a, b = self.a, self.b
        if self.restore_shapes:
            a = a.new_tensor([0.75, 1.0, 1.25])[:, None].expand_as(a).contiguous()
            b = b.new_tensor([-0.5, 0.0, 0.5])[:, None].expand_as(b).contiguous()
        operation = cached if self.mode == "cached_basis" else Readout.apply
        value = operation(z, a, b, self.readout(), self.initial_mean, self.scale, self.removal)
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
        self.attention_norm, self.attention = RMSNorm(config.width), Attention(config)
        self.ffn_norm, self.ffn = RMSNorm(config.width), BasisFFN(config)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class Model(nn.Module):
    model_name = "basis_readout_transformer"

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
