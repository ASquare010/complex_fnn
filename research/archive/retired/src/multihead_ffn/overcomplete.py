"""Overcomplete structured mixers with independent headwise SwiGLU maps."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.blockshuffle_ffn import BlockShuffleLinear, RecomputedSwiGLU


class OvercompleteHeadwiseFFN(nn.Module):
    """Structured rectangular mixers with independent headwise nonlinear maps."""

    def __init__(self, width: int, expanded: int, groups: int, heads: int, hidden: int):
        super().__init__()
        if min(width, expanded, groups, heads, hidden) <= 0:
            raise ValueError("Dimensions, groups and heads must be positive")
        if expanded <= width or width % groups or expanded % groups or expanded % heads:
            raise ValueError(
                "Overcomplete dimensions must respect mixer groups and head divisibility"
            )
        self.width, self.expanded, self.groups = width, expanded, groups
        self.heads, self.hidden = heads, hidden
        self.head_width = expanded // heads
        self.recompute_gate = False
        self.gate_recompute_method = "native"
        self.input_projection = BlockShuffleLinear(width, expanded, groups)
        self.output_projection = BlockShuffleLinear(expanded, width, groups)
        self.gate = nn.Parameter(torch.empty(heads, self.head_width, hidden))
        self.value = nn.Parameter(torch.empty(heads, self.head_width, hidden))
        self.down = nn.Parameter(torch.empty(heads, hidden, self.head_width))
        self.initialize(17, "overcomplete", 1.0)

    @property
    def parameter_count(self):
        return (
            2 * self.width * (self.width + self.expanded) // self.groups
            + 3 * self.expanded * self.hidden
        )

    @torch.no_grad()
    def initialize(self, seed: int, prefix: str, residual_scale: float):
        """CPU initialization, followed by .to(device), as in the shared decoder."""
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("Residual scale must be finite and positive")
        if self.gate.device.type != "cpu":
            raise ValueError("Initialize structured factors on CPU before moving to the device")
        unit = 1 / (0.02 * math.sqrt(self.expanded))
        self.input_projection.initialize(seed, prefix + ".input_projection", unit)
        self.output_projection.initialize(
            seed, prefix + ".output_projection", residual_scale * unit
        )
        for name in ("gate", "value", "down"):
            p = getattr(self, name)
            digest = hashlib.sha256(f"{seed}:{prefix}.{name}".encode()).digest()
            generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
            scale = (
                math.sqrt((8 * self.width // 3) / self.hidden)
                if name == "down"
                else math.sqrt(self.heads)
            )
            p.normal_(0, 0.02 * scale, generator=generator)

    def forward(self, x: torch.Tensor):
        if x.shape[-1] != self.width:
            raise ValueError("Input width does not match the module")
        q = self.input_projection(x).reshape(-1, self.heads, self.head_width)
        gate = torch.einsum("mhd,hdk->mhk", q, self.gate)
        value = torch.einsum("mhd,hdk->mhk", q, self.value)
        if self.recompute_gate and self.training and torch.is_grad_enabled():
            if self.gate_recompute_method != "native":
                raise ValueError(
                    "Overcomplete gate recomputation currently requires the native method"
                )
            features = RecomputedSwiGLU.apply(gate, value)
        else:
            features = F.silu(gate) * value
        y = torch.einsum("mhk,hkd->mhd", features, self.down)
        return self.output_projection(y.reshape(*x.shape[:-1], self.expanded))
