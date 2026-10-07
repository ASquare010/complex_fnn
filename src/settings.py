"""Selected LM dimensions; historical fields retained for checkpoint metadata."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    name: str = "base_transformer"
    vocab_size: int = 4096
    width: int = 192
    layers: int = 4
    heads: int = 6
    context: int = 256
    hidden: int = 512
    ffn: str = "swiglu"
    # Legacy metadata fields remain readable in historical leaderboard records.
    groups: int = 8
    loops: int = 1
    rope_base: float = 10000.0
    input_rank: int = 128
    output_rank: int = 128
    ffn_output_init_scale: float = 1.0

    def validate(self):
        for name in (
            "vocab_size",
            "width",
            "layers",
            "heads",
            "context",
            "hidden",
            "groups",
            "loops",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("Attention needs an even head dimension and width divisible by heads")
        if self.name not in ('base_transformer', 'branch_sigmoid') or self.loops != 1:
            raise ValueError('Only the selected Branch Sigmoid LM is active')
        if not self.rope_base > 1:
            raise ValueError("rope_base must exceed one")

@dataclass(frozen=True)
class ModelCounts:
    total_parameters: int
    ffn_parameters: int
    ffn_projection_flops_per_token: int
    ffn_passes: int
    attention: str = "full causal multi-head; one pass per block"
