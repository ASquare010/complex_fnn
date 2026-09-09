"""The shared decoder. Only the FFN factory differs between experiments."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.blockshuffle_ffn import BlockShuffleFFN, BlockShuffleLinear
from src.core.config import ModelConfig
from src.dense_ffn import DenseFFN


def make_ffn(config: ModelConfig) -> nn.Module:
    config.validate()
    if config.variant == "blockshuffle_swiglu":
        return BlockShuffleFFN(config.width, config.ffn_width, config.groups)
    return DenseFFN(config.width, config.ffn_width, config.variant.replace("_narrow", ""))


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
        return (y * self.weight).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.width // config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.out = nn.Linear(config.width, config.width, bias=False)
        freq = 1 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        angles = torch.outer(torch.arange(config.context).float(), freq)
        self.register_buffer("cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None], persistent=False)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        c, s = self.cos[:, :, : x.shape[-2]].to(x.dtype), self.sin[:, :, : x.shape[-2]].to(x.dtype)
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(self.rotate(q), self.rotate(k), v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, config: ModelConfig, ffn: nn.Module | None = None) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.width)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.width)
        self.ffn = make_ffn(config) if ffn is None else ffn
        self.recompute_scope = "none"

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        normalized = self.ffn_norm(x)
        if self.recompute_scope == "ffn" and self.training and torch.is_grad_enabled():
            y = checkpoint(self.ffn, normalized, use_reentrant=False, preserve_rng_state=True)
        else:
            y = self.ffn(normalized)
        return x + y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.recompute_scope == "block" and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward, x, use_reentrant=False, preserve_rng_state=True)
        return self._forward(x)


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig, seed: int = 17) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = RMSNorm(config.width)
        factor_parameters = {
            id(p)
            for module in self.modules()
            if isinstance(module, BlockShuffleLinear)
            for p in module.parameters()
        }
        # Name-local generators preserve shared tensors across different FFN shapes.
        with torch.no_grad():
            for name, p in self.named_parameters():
                if p.ndim < 2 or "theta" in name or id(p) in factor_parameters:
                    continue
                canonical_name = name.replace(".base.", ".")
                digest = hashlib.sha256(f"{seed}:{canonical_name}".encode()).digest()
                generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
                scale = (
                    0.02 / math.sqrt(2 * config.layers)
                    if name.endswith(("down.weight", "out.weight"))
                    else 0.02
                )
                if p.ndim == 3:
                    scale *= math.sqrt(p.shape[0])
                p.normal_(0, scale, generator=generator)
        for name, module in self.named_modules():
            if isinstance(module, BlockShuffleLinear):
                residual_scale = 1 / math.sqrt(2 * config.layers) if name.endswith(".down") else 1.0
                module.initialize(seed, name, residual_scale)
        assert sum(p.numel() for p in self.parameters()) == config.total_parameters

    def set_recompute_scope(self, scope: str) -> None:
        """Opt into native activation recomputation without changing parameter paths."""
        if scope not in ("none", "ffn", "block"):
            raise ValueError("Recomputation scope must be none, ffn or block")
        for block in self.blocks:
            block.recompute_scope = scope

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] > self.config.context:
            raise ValueError("Expected [batch, length] within configured context")
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        return F.linear(self.norm(x), self.embedding.weight)

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self(tokens).float().flatten(0, 1), targets.flatten())
