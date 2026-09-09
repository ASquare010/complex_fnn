"""Two-factor BlockShuffle projections, an established structured baseline."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.grouped_ffn import GroupedLinear, shuffle_channels


def unshuffle_channels(x: torch.Tensor, groups: int) -> torch.Tensor:
    return x.reshape(*x.shape[:-1], x.shape[-1] // groups, groups).transpose(-2, -1).reshape_as(x)


class BlockShuffleLinear(nn.Module):
    """P_out^-1 B2 P_mid B1 with intermediate width min(input, output)."""

    def __init__(self, input_width: int, output_width: int, groups: int) -> None:
        super().__init__()
        self.input_width, self.output_width, self.groups = input_width, output_width, groups
        middle = min(input_width, output_width)
        self.first = GroupedLinear(input_width, middle, groups)
        self.second = GroupedLinear(middle, output_width, groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.second(shuffle_channels(self.first(x), self.groups))
        return unshuffle_channels(y, self.groups)

    @torch.no_grad()
    def initialize(self, seed: int, name: str, residual_scale: float = 1.0) -> None:
        # Semi-orthogonal product with dense-equivalent mean row squared norm.
        gain = 0.02 * math.sqrt(max(self.input_width, self.output_width)) * residual_scale
        for label, factor in (("first", self.first), ("second", self.second)):
            digest = hashlib.sha256(f"{seed}:{name}.{label}.weight".encode()).digest()
            generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
            for block in factor.weight:
                nn.init.orthogonal_(block, gain=math.sqrt(gain), generator=generator)


class BlockShuffleFFN(nn.Module):
    def __init__(self, width: int, hidden: int, groups: int, gated: bool = True) -> None:
        super().__init__()
        self.recompute_gate = False
        self.gate_recompute_method = "checkpoint"
        self.up = BlockShuffleLinear(width, hidden, groups)
        self.gate = BlockShuffleLinear(width, hidden, groups) if gated else None
        self.down = BlockShuffleLinear(hidden, width, groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.up(x)
        if self.gate is None:
            features = F.gelu(u)
        else:
            v = self.gate(x)
            if self.recompute_gate and self.training and torch.is_grad_enabled():
                if self.gate_recompute_method == "native":
                    features = RecomputedSwiGLU.apply(u, v)
                else:
                    features = checkpoint(
                        swiglu_product, u, v, use_reentrant=False, preserve_rng_state=False
                    )
            else:
                features = swiglu_product(u, v)
        return self.down(features)


def swiglu_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Checkpointing this small expression recomputes no projection matrices."""
    return F.silu(u) * v


class RecomputedSwiGLU(torch.autograd.Function):
    """Save only gate inputs; reuse native SiLU backward with a recomputed value."""

    @staticmethod
    def forward(ctx, u, v):
        ctx.save_for_backward(u, v)
        return F.silu(u) * v

    @staticmethod
    def backward(ctx, grad_output):
        u, v = ctx.saved_tensors
        if torch.is_grad_enabled():
            # The installed native backward lacks a second derivative; retain it analytically.
            sigmoid = u.sigmoid()
            grad_u = grad_output * v * sigmoid * (1 + u * (1 - sigmoid))
        else:
            grad_u = torch.ops.aten.silu_backward(grad_output * v, u)
        grad_v = grad_output * F.silu(u)
        return grad_u, grad_v
