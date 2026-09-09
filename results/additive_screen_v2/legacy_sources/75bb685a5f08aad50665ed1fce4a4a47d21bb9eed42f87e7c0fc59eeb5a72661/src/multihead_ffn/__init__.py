"""Untrained equation-level reference for gated parallel multi-head FFNs.

This implements the architecture in FlashMHF equations 7 and 11-14, not its
SRAM algorithm or reported training configuration. It is not registered in the
shared Transformer factory during the active four-recipe cohort.
"""

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
