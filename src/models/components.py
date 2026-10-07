"""Reusable normalization, exact attention, and deterministic initialization."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from settings import ModelConfig, ModelCounts


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
        return (normalized * self.weight).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads, self.head_dim = config.heads, config.width // config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.out = nn.Linear(config.width, config.width, bias=False)
        frequency = config.rope_base ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        angle = torch.outer(torch.arange(config.context), frequency)
        self.register_buffer("cos", angle.cos()[None, None], persistent=False)
        self.register_buffer("sin", angle.sin()[None, None], persistent=False)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        cos = self.cos[:, :, : x.shape[-2]].to(x.dtype)
        sin = self.sin[:, :, : x.shape[-2]].to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(self.rotate(q), self.rotate(k), v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(batch, length, width))


def parameter_generator(seed: int, name: str) -> torch.Generator:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))


@torch.no_grad()
def initialize(model, seed):
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2 or ".ffn." in name:
            continue
        generator = parameter_generator(seed, name)
        scale = (
            0.02 / math.sqrt(2 * model.config.layers)
            if name.endswith(("down.weight", "out.weight"))
            else 0.02
        )
        parameter.normal_(0, scale, generator=generator)
    for index, block in enumerate(model.blocks):
        block.ffn.initialize(seed, f"blocks.{index}.ffn", 1 / math.sqrt(2 * model.config.layers))


def counts(model) -> ModelCounts:
    ffn = sum(p.numel() for block in model.blocks for p in block.ffn.parameters())
    projections = sum(
        getattr(block.ffn, "projection_parameters", sum(p.numel() for p in block.ffn.parameters()))
        for block in model.blocks
    )
    return ModelCounts(
        sum(p.numel() for p in model.parameters()),
        ffn,
        2 * projections * model.config.loops,
        model.config.loops,
    )
