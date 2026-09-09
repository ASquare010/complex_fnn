"""Explicit, serializable experimental settings."""

from dataclasses import dataclass

VARIANTS = (
    "gelu",
    "relu",
    "silu",
    "swiglu",
    "gelu_narrow",
    "bezier_shared",
    "bezier_grouped",
    "swiglu_narrow",
    "structured_gelu",
    "structured_swiglu",
    "structured_swiglu_coupled",
    "structured_swiglu_adaptive",
    "shared_gelu",
    "shared_swiglu",
    "shared_swiglu_affine",
    "blockshuffle_swiglu",
    "paired_antipodal_swiglu",
    "paired_reciprocal_swiglu",
    "paired_duplicate_swiglu",
    "quadratic_bezier",
    "quadratic_bezier_mixed",
    "swiglu_quadratic",
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
        if self.variant in (
            "swiglu",
            "swiglu_quadratic",
            "structured_swiglu",
            "structured_swiglu_coupled",
            "structured_swiglu_adaptive",
            "shared_swiglu",
            "shared_swiglu_affine",
            "blockshuffle_swiglu",
        ):
            return 8 * self.width // 3
        if self.variant == "swiglu_narrow":
            return 2 * self.width // 3
        if self.variant in ("gelu", "relu", "silu", "structured_gelu", "shared_gelu"):
            return 4 * self.width
        return self.width

    def validate(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {self.variant}")
        if min(self.width, self.layers, self.heads, self.context, self.vocab_size) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("RoPE requires an even head dimension dividing model width")
        if self.variant == "bezier_grouped" and (self.groups <= 0 or self.ffn_width % self.groups):
            raise ValueError("Groups must divide hidden width")
        if "quadratic" in self.variant and (self.groups <= 0 or self.ffn_width % self.groups):
            raise ValueError("Quadratic curve groups must divide hidden width")
        if self.variant == "quadratic_bezier_mixed" and self.groups != 3:
            raise ValueError("Quadratic product mixing requires three groups")
        if self.variant.startswith(("structured", "blockshuffle")) and (
            self.groups <= 0 or self.width % self.groups or self.ffn_width % self.groups
        ):
            raise ValueError("Structured groups must divide model and hidden widths")
        if self.variant.startswith("paired_") and self.ffn_width % 2:
            raise ValueError("Paired feature width must be even")
        if self.hidden < 0:
            raise ValueError("Hidden width cannot be negative")

    @property
    def ffn_parameters(self) -> int:
        base = (3 if "swiglu" in self.variant else 2) * self.width * self.ffn_width
        if self.variant.startswith("paired_"):
            base = 2 * self.width * self.ffn_width
        if self.variant.startswith("structured"):
            base //= self.groups
        if self.variant.startswith("blockshuffle"):
            base = (
                3 * min(self.width, self.ffn_width) * (self.width + self.ffn_width) // self.groups
            )
        extra = (
            2 * (1 if self.variant == "bezier_shared" else self.groups)
            if self.variant.startswith("bezier")
            else 0
        )
        if "quadratic" in self.variant:
            extra += 2 * self.groups + (3 if self.variant == "quadratic_bezier_mixed" else 0)
        if self.variant == "structured_swiglu_adaptive":
            extra += self.ffn_width
        if self.variant.endswith("_affine"):
            extra += 3 * self.ffn_width
        return base + extra

    @property
    def unique_ffn_parameters(self) -> int:
        count = (self.layers + 3) // 4 if self.variant.startswith("shared_") else self.layers
        if self.variant.endswith("_affine"):
            local = 3 * self.ffn_width
            return count * (self.ffn_parameters - local) + self.layers * local
        return count * self.ffn_parameters

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
        if self.gate_recompute_method not in ("checkpoint", "native"):
            raise ValueError("Gate recomputation method must be checkpoint or native")
        if self.ffn_decay_mode not in ("parameter", "product"):
            raise ValueError("FFN decay mode must be parameter or product")
        if self.ffn_lr_mode not in ("uniform", "fan_in"):
            raise ValueError("FFN LR mode must be uniform or fan_in")
        if self.precision not in ("auto", "fp32", "bf16"):
            raise ValueError("Precision must be auto, fp32 or bf16")
