"""Block-diagonal plus global low-rank projections for ordinary SwiGLU."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.blockshuffle_ffn import RecomputedSwiGLU
from src.core.structured_linear import GroupedLinear


class AdditiveBlockLowRankLinear(nn.Module):
    """Evaluate Sx + L(Rx) without constructing the dense effective weight."""

    global_energy_fraction = 0.5

    def __init__(self, input_width: int, output_width: int, groups: int, rank: int):
        super().__init__()
        if min(input_width, output_width, groups, rank) <= 0:
            raise ValueError("Projection dimensions, groups and rank must be positive")
        if input_width % groups or output_width % groups:
            raise ValueError("Groups must divide both projection dimensions")
        if rank > min(input_width, output_width):
            raise ValueError("Rank cannot exceed either projection dimension")
        self.input_width, self.output_width = input_width, output_width
        self.groups, self.rank = groups, rank
        self.local = GroupedLinear(input_width, output_width, groups)
        self.right = nn.Linear(input_width, rank, bias=False)
        self.left = nn.Linear(rank, output_width, bias=False)
        self.initialize(17, "projection", 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.local(x) + self.left(self.right(x))

    @torch.no_grad()
    def initialize(self, seed: int, name: str, sigma: float) -> None:
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("Initialization sigma must be finite and positive")
        if self.local.weight.device.type != "cpu":
            raise ValueError("Initialize with name-local CPU generators before device transfer")
        self.initialization_sigma = sigma
        rho = self.global_energy_fraction
        block_gain = sigma * math.sqrt((1 - rho) * max(self.input_width, self.output_width))
        global_gain = sigma * math.sqrt(rho * self.input_width * self.output_width / self.rank)
        for label, parameter in (
            ("local", self.local.weight),
            ("right", self.right.weight),
            ("left", self.left.weight),
        ):
            digest = hashlib.sha256(f"{seed}:{name}.{label}.weight".encode()).digest()
            generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
            if label == "local":
                for block in parameter:
                    nn.init.orthogonal_(block, gain=block_gain, generator=generator)
            else:
                nn.init.orthogonal_(parameter, gain=math.sqrt(global_gain), generator=generator)

    def parameter_lr_multipliers(self) -> dict[int, float]:
        """Frozen independent-perturbation calibration; not Adam invariance."""
        rho = self.global_energy_fraction
        gain = self.initialization_sigma * math.sqrt(
            rho * self.input_width * self.output_width / self.rank
        )
        return {
            id(self.local.weight): math.sqrt((1 - rho) * self.groups),
            id(self.left.weight): math.sqrt(rho * self.input_width / (2 * self.rank * gain)),
            id(self.right.weight): math.sqrt(rho * self.output_width / (2 * self.rank * gain)),
        }


class AdditiveBlockLowRankFFN(nn.Module):
    """Full hidden-width SwiGLU with a local and global path in each projection."""

    def __init__(
        self, width: int, hidden: int | None = None, groups: int = 8, rank: int | None = None
    ):
        super().__init__()
        if width <= 0 or width % 24:
            raise ValueError("This candidate requires model width divisible by 24")
        hidden = 8 * width // 3 if hidden is None else hidden
        rank = width // 8 if rank is None else rank
        self.width, self.hidden, self.groups, self.rank = width, hidden, groups, rank
        self.up = AdditiveBlockLowRankLinear(width, hidden, groups, rank)
        self.down = AdditiveBlockLowRankLinear(hidden, width, groups, rank)
        self.gate = AdditiveBlockLowRankLinear(width, hidden, groups, rank)
        self.recompute_gate = False
        self.gate_recompute_method = "native"

    @torch.no_grad()
    def initialize(self, seed: int, name: str, layers: int) -> None:
        if layers <= 0:
            raise ValueError("Layer count must be positive")
        for label in ("up", "down", "gate"):
            sigma = 0.02 / math.sqrt(2 * layers) if label == "down" else 0.02
            getattr(self, label).initialize(seed, f"{name}.{label}", sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.up(x), self.gate(x)
        if self.recompute_gate and self.training and torch.is_grad_enabled():
            if self.gate_recompute_method != "native":
                raise ValueError("Additive gate recomputation requires the native method")
            z = RecomputedSwiGLU.apply(u, v)
        else:
            z = F.silu(u) * v
        return self.down(z)
