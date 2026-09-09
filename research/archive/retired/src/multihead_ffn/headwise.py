"""Calibrated, router-free headwise SwiGLU with dense input/output mixers."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


class CalibratedHeadwiseFFN(nn.Module):
    """One private SwiGLU per head; heads interact through two learned dense mixers."""

    def __init__(self, width: int, heads: int, hidden: int):
        super().__init__()
        if min(width, heads, hidden) <= 0 or width % heads:
            raise ValueError("Positive dimensions and heads dividing width are required")
        self.width, self.heads, self.hidden = width, heads, hidden
        self.head_width = width // heads
        self.input_projection = nn.Linear(width, width, bias=False)
        self.output_projection = nn.Linear(width, width, bias=False)
        self.gate = nn.Parameter(torch.empty(heads, self.head_width, hidden))
        self.value = nn.Parameter(torch.empty(heads, self.head_width, hidden))
        self.down = nn.Parameter(torch.empty(heads, hidden, self.head_width))
        self.initialize(17, "headwise", 1.0)

    @property
    def parameter_count(self):
        return 2 * self.width**2 + 3 * self.width * self.hidden

    @torch.no_grad()
    def initialize(self, seed: int, prefix: str, residual_scale: float):
        """Orthogonal mixers and H045's private variance rule with one subnetwork."""
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("Residual scale must be finite and positive")
        for name, p in self.named_parameters():
            digest = hashlib.sha256(f"{seed}:{prefix}.{name}".encode()).digest()
            generator = torch.Generator(device=p.device).manual_seed(
                int.from_bytes(digest[:8], "little")
            )
            if name in ("input_projection.weight", "output_projection.weight"):
                q, r = torch.linalg.qr(
                    torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=generator)
                )
                q = q * torch.where(r.diag() < 0, -1.0, 1.0).unsqueeze(0)
                p.copy_(q * (residual_scale if name == "output_projection.weight" else 1.0))
            else:
                scale = (
                    math.sqrt((8 * self.width // 3) / self.hidden)
                    if name == "down"
                    else math.sqrt(self.heads)
                )
                p.normal_(0, 0.02 * scale, generator=generator)

    def forward(self, x: torch.Tensor):
        if x.shape[-1] != self.width:
            raise ValueError("Input width does not match the model")
        q = self.input_projection(x).reshape(-1, self.heads, self.head_width)
        gate = torch.einsum("mhd,hdk->mhk", q, self.gate)
        value = torch.einsum("mhd,hdk->mhk", q, self.value)
        y = torch.einsum("mhk,hkd->mhd", F.silu(gate) * value, self.down)
        return self.output_projection(y.reshape(x.shape))
