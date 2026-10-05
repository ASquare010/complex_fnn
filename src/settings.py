"""Strict, serializable experiment definitions; unknown fields are errors."""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


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
        if self.name not in ("base_transformer", "channel_curve_transformer"):
            raise ValueError(
                f"Retired or unknown model: {self.name}; see ffn_experiments/README.md"
            )
        if self.loops != 1:
            raise ValueError("Active models use one FFN pass")
        if not math.isfinite(self.ffn_output_init_scale) or self.ffn_output_init_scale <= 0:
            raise ValueError("FFN output initialization scale must be finite and positive")
        if self.name == "channel_curve_transformer":
            if self.ffn != "self_curve_wide" or self.hidden != self.width:
                raise ValueError("The selected model uses self_curve_wide with hidden == width")
            if self.ffn_output_init_scale != 1:
                raise ValueError("Curve-Wide retains its measured initialization")
        elif self.ffn not in ("swiglu", "swiglu_fused", "swiglu_kernel", "gelu"):
            raise ValueError(f"Unknown dense FFN: {self.ffn}")
        if not self.rope_base > 1:
            raise ValueError("rope_base must exceed one")


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 1000
    batch_size: int = 8
    learning_rate: float = 0.0006
    weight_decay: float = 0.1
    warmup_steps: int = 100
    eval_every: int = 100
    eval_batches: int = 32
    seed: int = 17
    device: str = "cpu"
    precision: str = "fp32"
    threads: int = 4

    def validate(self):
        for name in ("steps", "batch_size", "eval_every", "eval_batches", "threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be smaller than steps")
        if not 0 < self.learning_rate < 1 or not 0 <= self.weight_decay < 1:
            raise ValueError("Invalid optimizer settings")
        if self.device not in ("cpu", "cuda") or self.precision not in ("fp32", "bf16"):
            raise ValueError("Use cpu/cuda and fp32/bf16")
        if self.device == "cpu" and self.precision != "fp32":
            raise ValueError("CPU reference runs use fp32")


@dataclass(frozen=True)
class Experiment:
    name: str
    dataset: str
    hypothesis: str
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, raw: dict):
        values = dict(raw)
        values["model"] = ModelConfig(**values.get("model", {}))
        values["training"] = TrainingConfig(**values.get("training", {}))
        result = cls(**values)
        result.model.validate()
        result.training.validate()
        if not result.name or not result.hypothesis:
            raise ValueError("Name and hypothesis are required")
        return result

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ModelCounts:
    total_parameters: int
    ffn_parameters: int
    ffn_projection_flops_per_token: int
    ffn_passes: int
    attention: str = "full causal multi-head; one pass per block"


@dataclass(frozen=True)
class TaskScore:
    correct: int
    examples: int
    accuracy: float


@dataclass(frozen=True)
class Evaluation:
    nll: float
    perplexity: float
    targets: int
    split: str
    complete_split: bool
    exact_match: dict[str, TaskScore] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    run: str
    completed: bool
    step: int
    validation: Evaluation
    counts: ModelCounts


@dataclass(frozen=True)
class SourceSplit:
    source: str
    limit: int
    skip: int = 0


@dataclass(frozen=True)
class DatasetConfig:
    kind: str
    output: str
    vocab_size: int = 4096
    seed: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    repo: str = ""
    revision: str = ""
    subset: str | None = None
    unit: str = "document"
    scope: str = ""
    splits: dict[str, SourceSplit] = field(default_factory=dict)
    max_characters: int = 200_000_000

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        values["splits"] = {
            name: SourceSplit(**value) for name, value in values.get("splits", {}).items()
        }
        result = cls(**values)
        if result.kind not in ("synthetic", "remote") or result.vocab_size < 258:
            raise ValueError("Invalid dataset kind or vocabulary")
        if any(type(n) is not int or n < 1 for n in result.counts.values()):
            raise ValueError("Split counts must be positive integers")
        if any(s.limit < 1 or s.skip < 0 for s in result.splits.values()):
            raise ValueError("Invalid source split bounds")
        return result
