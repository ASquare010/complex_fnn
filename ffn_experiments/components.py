"""Reusable normalization, exact attention, and dense/structured FFNs."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from ffn_experiments.settings import ModelConfig, ModelCounts


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


class DenseFFN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fused = config.ffn in ("swiglu_fused", "swiglu_kernel")
        self.kernel = config.ffn == "swiglu_kernel"
        self.output_init_scale = config.ffn_output_init_scale
        self.up = self.projection(config.width, config.hidden * (2 if self.fused else 1))
        self.down = self.projection(config.hidden, config.width)
        self.gate = self.projection(config.width, config.hidden) if config.ffn == "swiglu" else None

    def projection(self, inputs: int, outputs: int) -> nn.Module:
        return nn.Linear(inputs, outputs, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.up(x)
        if self.fused:
            hidden, gate = hidden.chunk(2, dim=-1)
            if self.kernel:
                from ffn_experiments.activation_kernels import FusedSwiGLU

                hidden = FusedSwiGLU.apply(hidden, gate)
            else:
                hidden = F.silu(hidden) * gate
        else:
            hidden = F.gelu(hidden) if self.gate is None else F.silu(hidden) * self.gate(x)
        return self.down(hidden)

    @torch.no_grad()
    def initialize(self, seed: int, prefix: str, residual_scale: float):
        for name, parameter in self.named_parameters():
            scale = 0.02 * (residual_scale if name.startswith("down.") else 1)
            if name.startswith("down."):
                scale *= self.output_init_scale
            if self.fused and name == "up.weight":
                up, gate = parameter.chunk(2, dim=0)
                up.normal_(0, scale, generator=parameter_generator(seed, f"{prefix}.up.weight"))
                gate.normal_(0, scale, generator=parameter_generator(seed, f"{prefix}.gate.weight"))
            else:
                parameter.normal_(0, scale, generator=parameter_generator(seed, f"{prefix}.{name}"))


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
