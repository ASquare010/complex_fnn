"""Explicit settings for the retained models and reproducible controls."""

from dataclasses import dataclass

VARIANTS = (
    "gelu",
    "swiglu",
    "gelu_narrow",
    "swiglu_narrow",
    "blockshuffle_swiglu",
)


@dataclass(frozen=True)
class ModelConfig:
    variant: str = "gelu"
    vocab_size: int = 4096
    width: int = 192
    layers: int = 4
    heads: int = 6
    context: int = 128
    hidden: int = 0
    groups: int = 8

    @property
    def ffn_width(self) -> int:
        if self.hidden:
            return self.hidden
        if self.variant == "swiglu_narrow":
            return 2 * self.width // 3
        if "swiglu" in self.variant:
            return 8 * self.width // 3
        return self.width if self.variant == "gelu_narrow" else 4 * self.width

    def validate(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(
                f"Unknown or retired variant: {self.variant}. See research/archive/README.md"
            )
        if min(self.width, self.layers, self.heads, self.context, self.vocab_size) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("RoPE requires an even head dimension dividing model width")
        if self.hidden < 0:
            raise ValueError("Hidden width cannot be negative")
        if self.variant.startswith("blockshuffle") and (
            self.groups <= 0 or self.width % self.groups or self.ffn_width % self.groups
        ):
            raise ValueError("Structured groups must divide model and hidden widths")

    @property
    def ffn_parameters(self) -> int:
        if self.variant.startswith("blockshuffle"):
            base = (
                3 * min(self.width, self.ffn_width) * (self.width + self.ffn_width) // self.groups
            )
            return base
        return (3 if "swiglu" in self.variant else 2) * self.width * self.ffn_width

    @property
    def unique_ffn_parameters(self) -> int:
        return self.layers * self.ffn_parameters

    @property
    def total_parameters(self) -> int:
        d = self.width
        return (
            self.vocab_size * d + self.layers * (4 * d * d + 2 * d) + self.unique_ffn_parameters + d
        )


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 200
    batch_size: int = 16
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    seed: int = 17
    eval_batches: int = 16
    log_every: int = 50
    precision: str = "auto"
    device: str = "cuda"
    ffn_lr_mode: str = "uniform"
    ffn_decay_mode: str = "parameter"
    recompute_gate: bool = False
    gate_recompute_method: str = "checkpoint"
    ffn_width_init_mode: str = "none"
    ffn_width_lr_mode: str = "none"
    activation_backend: str = "eager"

    def validate(self) -> None:
        if min(self.steps, self.batch_size, self.eval_batches, self.log_every) <= 0:
            raise ValueError("Training counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if self.ffn_width_init_mode not in ("none", "fan_in") or self.ffn_width_lr_mode not in (
            "none",
            "fan_in",
        ):
            raise ValueError("Dense FFN width calibration must be none or fan_in")
        if self.activation_backend != "eager":
            raise ValueError(
                "Only the native eager activation backend is retained; compiler repeats failed quality. See research/archive/README.md"
            )
        if self.gate_recompute_method not in ("checkpoint", "native"):
            raise ValueError("Gate recomputation method must be checkpoint or native")
        if self.ffn_decay_mode not in ("parameter", "product"):
            raise ValueError("FFN decay mode must be parameter or product")
        if self.ffn_lr_mode not in ("uniform", "fan_in"):
            raise ValueError("FFN LR mode must be uniform or fan_in")
        if self.precision not in ("auto", "fp32", "bf16"):
            raise ValueError("Precision must be auto, fp32 or bf16")
