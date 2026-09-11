"""H104 factorized neurons and equally initialized dense controls."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

FORMS = (
    "full_gelu_random",
    "full_swiglu_random",
    "full_gelu_bounded",
    "full_swiglu_bounded",
    "narrow_gelu_bounded",
    "narrow_swiglu_bounded",
    "narrow_cubic_bounded",
    "tiny_gelu_bounded",
    "tiny_swiglu_bounded",
    "tiny_cubic_bounded",
    "tiny_relu2_bounded",
    "factor_gelu_random",
    "factor_gelu_raw",
    "factor_gelu_bounded",
    "factor_cubic_random",
    "factor_cubic_raw",
    "factor_cubic_bounded",
)


def generator(seed: int, name: str) -> torch.Generator:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))


class FeatureFFN(nn.Module):
    def __init__(self, form: str, seed: int, bases: dict[str, torch.Tensor]) -> None:
        super().__init__()
        if form not in FORMS:
            raise ValueError(form)
        size, self.activation, mode = form.split("_")
        self.form = form
        self.hidden = (
            (1024 if self.activation == "swiglu" else 1536)
            if size == "full"
            else (
                (304 if self.activation == "swiglu" else 456)
                if size == "narrow"
                else (128 if size == "factor" else (57 if self.activation == "swiglu" else 85))
            )
        )
        self.compress = nn.Linear(384, 32, bias=False) if size == "factor" else None
        self.up = nn.Linear(32 if self.compress is not None else 384, self.hidden)
        self.gate = nn.Linear(384, self.hidden) if self.activation == "swiglu" else None
        self.down = nn.Linear(self.hidden, 384)
        with torch.no_grad():
            if self.compress is not None:
                self.compress.weight.copy_(bases[mode].T)
            for name in ("up", "gate"):
                module = getattr(self, name)
                if module is None:
                    continue
                if size == "factor" or mode != "random":
                    coefficients = torch.randn(
                        self.hidden, 32, generator=generator(seed, f"{name}.weight")
                    ) / math.sqrt(32)
                    weight = coefficients if size == "factor" else coefficients @ bases[mode].T
                    module.weight.copy_(weight)
                else:
                    module.weight.normal_(
                        0, 1 / math.sqrt(384), generator=generator(seed, f"{name}.weight")
                    )
                module.bias.uniform_(-1, 1, generator=generator(seed, f"{name}.bias"))
            self.down.weight.zero_()
            self.down.bias.zero_()

    def features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.up(x if self.compress is None else self.compress(x))
        if self.activation == "swiglu":
            features = F.silu(u) * self.gate(x)
        elif self.activation == "gelu":
            features = F.gelu(u)
        elif self.activation == "relu2":
            features = F.relu(u).square()
        else:
            features = u * (u.square() - 3) / math.sqrt(6)
        return u, features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.features(x)[1])

    def optimizer_groups(self, rate: float) -> list[dict]:
        reference = 1024 if self.activation == "swiglu" else 1536
        return [
            {
                "params": [p],
                "lr": rate * (reference / self.hidden if name == "down.weight" else 1),
                "parameter_name": name,
            }
            for name, p in self.named_parameters()
        ]

    def counts(self) -> dict:
        return {
            "parameters": sum(p.numel() for p in self.parameters()),
            "matrix_parameters": sum(p.numel() for p in self.parameters() if p.ndim == 2),
            "fixed_buffer_entries": sum(p.numel() for p in self.buffers()),
        }
