"""Two dense projections with signed, reciprocal, or duplicate gated features."""

import torch
from torch import nn
from torch.nn import functional as F

MODES = ("antipodal", "reciprocal", "duplicate")


def paired_features(z: torch.Tensor, mode: str) -> torch.Tensor:
    """Return two features per gate/value pair without another projection."""
    if z.shape[-1] % 2:
        raise ValueError("Paired features require an even projected width")
    a, b = z.chunk(2, dim=-1)
    first = F.silu(a) * b
    if mode == "antipodal":
        second = F.silu(-a) * b
    elif mode == "reciprocal":
        second = F.silu(b) * a
    elif mode == "duplicate":
        second = first
    else:
        raise ValueError(f"Unknown pair mode: {mode}")
    return torch.cat((first, second), dim=-1)


class PairedFeatureFFN(nn.Module):
    def __init__(self, width: int, hidden: int, mode: str) -> None:
        super().__init__()
        if hidden <= 0 or hidden % 2 or mode not in MODES:
            raise ValueError("Expected an even positive hidden width and a supported pair mode")
        self.mode = mode
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(paired_features(self.up(x), self.mode))
