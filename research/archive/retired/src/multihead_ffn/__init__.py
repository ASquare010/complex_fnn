"""Equation-level architecture reference for gated parallel multi-head FFNs.

This implements the architecture in FlashMHF equations 7 and 11-14, not its
SRAM algorithm or reported training configuration. Registered variants expose
explicit reference and variance-calibrated initializations.
"""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


def normalized_sigmoid(logits: torch.Tensor, epsilon: float = 1e-6):
    """Compute sigmoid(z)/(sum(sigmoid(z))+epsilon), retaining the epsilon term.

    Log arithmetic avoids premature sigmoid underflow. Accumulate in FP32 for
    FP16/BF16 inputs and preserve FP64 for numerical reference checks.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("A finite positive normalization epsilon is required")
    value = logits if logits.dtype == torch.float64 else logits.float()
    log_weights = F.logsigmoid(value)
    log_sum = torch.logsumexp(log_weights, dim=-1, keepdim=True)
    denominator = torch.logaddexp(log_sum, value.new_tensor(math.log(epsilon)))
    return torch.exp(log_weights - denominator)


class ParallelMultiheadFFN(nn.Module):
    """Dense input/output mixing around private, softly weighted head subnetworks."""

    def __init__(
        self,
        width: int,
        heads: int,
        subnetworks: int,
        hidden_per_subnetwork: int,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        if min(width, heads, subnetworks, hidden_per_subnetwork) <= 0 or width % heads:
            raise ValueError("Positive dimensions and heads dividing width are required")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("A finite positive normalization epsilon is required")
        self.width, self.heads, self.subnetworks = width, heads, subnetworks
        self.head_width, self.hidden = width // heads, hidden_per_subnetwork
        self.epsilon = epsilon
        self.input_projection = nn.Linear(width, width, bias=False)
        self.output_projection = nn.Linear(width, width, bias=False)
        shape = (heads, subnetworks, self.head_width, self.hidden)
        self.gate = nn.Parameter(torch.empty(shape))
        self.value = nn.Parameter(torch.empty(shape))
        self.down = nn.Parameter(torch.empty(heads, subnetworks, self.hidden, self.head_width))
        self.router = nn.Parameter(torch.empty(heads, self.head_width, subnetworks))
        # A standalone reference initialization; no trained fairness claim is made.
        for parameter in self.parameters():
            nn.init.normal_(parameter, std=0.02)

    @torch.no_grad()
    def initialize(self, seed: int, prefix: str, residual_scale: float, calibrated: bool):
        """Name-local Transformer initialization; standalone construction stays unchanged.

        Calibration preserves input norms and approximates full SwiGLU output
        variance at uniform routing. It is not an exact distributional equality.
        """
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("Residual scale must be finite and positive")
        full_hidden = 8 * self.width // 3
        for name, p in self.named_parameters():
            digest = hashlib.sha256(f"{seed}:{prefix}.{name}".encode()).digest()
            generator = torch.Generator(device=p.device).manual_seed(
                int.from_bytes(digest[:8], "little")
            )
            if calibrated and name in ("input_projection.weight", "output_projection.weight"):
                q, r = torch.linalg.qr(
                    torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=generator)
                )
                q = q * torch.where(r.diag() < 0, -1.0, 1.0).unsqueeze(0)
                p.copy_(q * (residual_scale if name == "output_projection.weight" else 1.0))
            elif calibrated and name == "router":
                p.zero_()
            else:
                scale = 0.02
                if calibrated and name in ("gate", "value"):
                    scale *= math.sqrt(self.heads)
                elif calibrated and name == "down":
                    scale *= math.sqrt(self.subnetworks * full_hidden / self.hidden)
                elif name == "output_projection.weight":
                    scale *= residual_scale
                p.normal_(0, scale, generator=generator)

    @property
    def parameter_count(self):
        return (
            2 * self.width**2
            + 3 * self.width * self.subnetworks * self.hidden
            + self.width * self.subnetworks
        )

    def forward(self, x: torch.Tensor):
        if x.shape[-1] != self.width:
            raise ValueError("Input width does not match the model")
        shape = x.shape
        q = self.input_projection(x).reshape(-1, self.heads, self.head_width)
        routing_logits = torch.einsum("mhd,hde->mhe", q, self.router)
        weights = normalized_sigmoid(routing_logits, self.epsilon)
        gate = torch.einsum("mhd,hedk->mhek", q, self.gate)
        value = torch.einsum("mhd,hedk->mhek", q, self.value)
        subnetwork = torch.einsum("mhek,hekd->mhed", F.silu(gate) * value, self.down)
        mixed = (subnetwork * weights.unsqueeze(-1)).sum(dim=2).to(q.dtype)
        return self.output_projection(mixed.reshape(shape))
