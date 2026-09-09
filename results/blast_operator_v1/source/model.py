"""Isolated BLAST comparator; no active registry or shared-trainer changes.

BLAST factorization: Lee et al., arXiv:2410.21262v1, Equation 2.
The rank-batched contraction and orthogonal calibration are local choices.
"""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.blockshuffle_ffn import unshuffle_channels


def contract(x, row, column, mix):
    """Canonical block projection with [rank, output_group, input_group] mixing."""
    blocks, _, input_chunk = column.shape
    grouped = x.reshape(-1, blocks, input_chunk).transpose(0, 1)
    encoded = torch.bmm(grouped, column.transpose(1, 2))
    coupled = torch.bmm(encoded.permute(2, 1, 0), mix.transpose(1, 2))
    decoded = torch.bmm(coupled.permute(2, 1, 0), row.transpose(1, 2))
    return decoded.transpose(0, 1).reshape(*x.shape[:-1], blocks * row.shape[1])


class BlastLinear(nn.Module):
    """BLAST with rank no larger than either block dimension for this initializer."""

    def __init__(
        self, input_width, output_width, blocks=8, rank=48, *, seed=17,
        name="projection", residual_scale=1.0, dtype=torch.float32,
    ):
        super().__init__()
        if min(input_width, output_width, blocks, rank) <= 0:
            raise ValueError("Widths, blocks and rank must be positive")
        if input_width % blocks or output_width % blocks:
            raise ValueError("Blocks must divide both widths")
        if rank > min(input_width, output_width) // blocks:
            raise ValueError("This orthogonal initializer requires rank <= either block width")
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("Residual scale must be positive and finite")
        self.input_width, self.output_width = input_width, output_width
        self.blocks, self.rank = blocks, rank
        self.row = nn.Parameter(torch.empty(blocks, output_width // blocks, rank, dtype=dtype))
        self.column = nn.Parameter(torch.empty(blocks, rank, input_width // blocks, dtype=dtype))
        self.mix = nn.Parameter(torch.empty(rank, blocks, blocks, dtype=dtype))
        self.gain = 0.02 * math.sqrt(max(input_width, output_width)) * residual_scale
        scale = self.gain ** (1 / 3)
        for label, parameter in self.named_parameters():
            digest = hashlib.sha256(f"{seed}:{name}.{label}".encode()).digest()
            generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
            for matrix in parameter:
                nn.init.orthogonal_(matrix, gain=scale, generator=generator)

    def forward(self, x):
        if x.ndim < 1 or x.shape[-1] != self.input_width:
            raise ValueError("BLAST projection input dimension mismatch")
        return contract(x, self.row, self.column, self.mix)


class BlastFFN(nn.Module):
    """Local BLAST FFN with the retained BlockShuffle outer permutations."""

    def __init__(self, form, seed=17, dtype=torch.float32):
        super().__init__()
        if form not in ("gelu", "swiglu"):
            raise ValueError("Expected gelu or swiglu")
        self.form, self.width, self.hidden = form, 384, 3200 if form == "gelu" else 1984
        self.up = BlastLinear(384, self.hidden, seed=seed, name="ffn.up", dtype=dtype)
        self.gate = (
            BlastLinear(384, self.hidden, seed=seed, name="ffn.gate", dtype=dtype)
            if form == "swiglu" else None
        )
        self.down = BlastLinear(
            self.hidden, 384, seed=seed, name="ffn.down", residual_scale=0.25, dtype=dtype,
        )

    def forward(self, x):
        u = unshuffle_channels(self.up(x), 8)
        features = F.gelu(u) if self.gate is None else F.silu(u) * unshuffle_channels(self.gate(x), 8)
        return unshuffle_channels(self.down(features), 8)
