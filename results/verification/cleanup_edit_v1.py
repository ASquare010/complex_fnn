"""Apply the authorized active-model cleanup using the verified H057 snapshot."""

import ast
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path.cwd().resolve()
ARCHIVE = ROOT / "research/archive/h057_source.zip"
manifest = json.loads((ROOT / "research/archive/h057_manifest.json").read_text())
assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == manifest["archive_sha256"]
archive = zipfile.ZipFile(ARCHIVE)


def old(name):
    return archive.read(name).decode("utf-8").replace("\r\n", "\n")


def write(name, value):
    path = ROOT / name
    assert path.resolve().is_relative_to(ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def remove_nodes(source, names):
    lines = source.splitlines(keepends=True)
    for node in sorted(ast.parse(source).body, key=lambda n: n.lineno, reverse=True):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            start = min([node.lineno, *[d.lineno for d in getattr(node, "decorator_list", [])]])
            del lines[start - 1 : node.end_lineno]
    return "".join(lines)


# Save the independent block operator; the rejected grouped FFN is retired.
s = old("src/grouped_ffn/__init__.py")
s = remove_nodes(s, {"StructuredFFN"}).replace("from torch.nn import functional as F\n", "")
s = s.replace(
    '"""Wide nonlinear features with grouped projection matrices."""',
    '"""Reusable block-diagonal projections and channel permutations."""',
)
write("src/core/structured_linear.py", s)

# Retain only the native rational activation. Parameter paths stay identical.
s = old("src/learnable_activation_ffn/__init__.py")
s = remove_nodes(
    s, {"ShiftedBezierActivation", "AffineResidualActivation", "activation", "LearnableDenseFFN"}
)
s = s.replace("from src.dense_ffn import DenseFFN\n", "")
s = s.replace(
    '"""Small learned residual activations on calibrated dense and BlockShuffle FFNs."""',
    '"""Native rational residual activation on the retained BlockShuffle candidate."""',
)
s = s.replace("class LearnableBlockShuffleFFN", "class RationalBlockShuffleFFN")
s = s.replace(
    "def __init__(self, width: int, hidden: int, groups: int, family: str)",
    "def __init__(self, width: int, hidden: int, groups: int)",
)
s = s.replace(
    "self.curve = activation(hidden, groups, family)",
    "self.curve = RationalResidualActivation(hidden, groups)",
)
write("src/rational_blockshuffle_ffn/__init__.py", s)

config_header = '''"""Explicit settings for the retained models and reproducible controls."""
from dataclasses import dataclass

VARIANTS = (
    "gelu", "swiglu", "gelu_narrow", "swiglu_narrow",
    "blockshuffle_swiglu", "blockshuffle_swiglu_rational",
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
            raise ValueError(f"Unknown or retired variant: {self.variant}. See research/archive/README.md")
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
            base = 3 * min(self.width, self.ffn_width) * (self.width + self.ffn_width) // self.groups
            return base + (5 * self.groups if self.variant.endswith("_rational") else 0)
        return (3 if "swiglu" in self.variant else 2) * self.width * self.ffn_width

    @property
    def unique_ffn_parameters(self) -> int:
        return self.layers * self.ffn_parameters

    @property
    def total_parameters(self) -> int:
        d = self.width
        return self.vocab_size * d + self.layers * (4 * d * d + 2 * d) + self.unique_ffn_parameters + d


'''
s = old("src/core/config.py")
s = s[s.index("@dataclass(frozen=True)\nclass TrainConfig:") :]
a = s.index("        if self.activation_backend not in (")
b = s.index("        if self.gate_recompute_method", a)
s = (
    s[:a]
    + """        if self.activation_backend != "eager":
            raise ValueError("Only the native eager activation backend is retained; compiler repeats failed quality. See research/archive/README.md")
"""
    + s[b:]
)
write("src/core/config.py", config_header + s)

s = old("src/core/transformer.py")
a = s.index("from src.adaptive_cross_group_ffn")
b = s.index("\n\nclass RMSNorm", a)
s = (
    s[:a]
    + """from src.blockshuffle_ffn import BlockShuffleFFN, BlockShuffleLinear
from src.core.config import ModelConfig
from src.dense_ffn import DenseFFN
from src.rational_blockshuffle_ffn import RationalBlockShuffleFFN


def make_ffn(config: ModelConfig) -> nn.Module:
    config.validate()
    if config.variant == "blockshuffle_swiglu_rational":
        return RationalBlockShuffleFFN(config.width, config.ffn_width, config.groups)
    if config.variant == "blockshuffle_swiglu":
        return BlockShuffleFFN(config.width, config.ffn_width, config.groups)
    return DenseFFN(config.width, config.ffn_width, config.variant.replace("_narrow", ""))
"""
    + s[b:]
)
a = s.index("        bank: dict[int, nn.Module]")
b = s.index("        self.norm = RMSNorm", a)
s = (
    s[:a]
    + "        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])\n"
    + s[b:]
)
a = s.index("            elif isinstance(module, CalibratedHeadwiseFFN):")
b = s.index("        assert sum(p.numel()", a)
s = s[:a] + s[b:]
write("src/core/transformer.py", s)

s = old("src/core/diagnostics.py")
s = (
    s.replace("from src.bezier_ffn import QuadraticBezierActivation\n", "")
    .replace(
        "from src.learnable_activation_ffn import LearnableActivation",
        "from src.rational_blockshuffle_ffn import LearnableActivation",
    )
    .replace("from src.multihead_ffn import ParallelMultiheadFFN, normalized_sigmoid\n", "")
)
a = s.index("            if isinstance(module, ParallelMultiheadFFN):")
b = s.index("            if isinstance(module, LearnableActivation):", a)
s = s[:a] + s[b:]
a = s.index('            if getattr(module, "mix_theta", None) is not None:')
b = s.index("\n        return collect", a)
s = s[:a] + s[b:]
a = s.index("        if isinstance(curve, QuadraticBezierActivation):")
b = s.index("        if isinstance(curve, LearnableActivation):", a)
s = s[:a] + s[b:]
write("src/core/diagnostics.py", s)

s = old("src/core/trainer.py")
a = s.index('    if variant == "overcomplete_headwise_swiglu":')
b = s.index("    for block in model.blocks:", a)
s = (
    s[:a]
    + """    if not variant.startswith("blockshuffle_swiglu"):
        raise ValueError("Gate recomputation supports BlockShuffle SwiGLU")
    if variant.endswith("_rational") and config.gate_recompute_method != "checkpoint":
        raise ValueError("Learnable activations require checkpoint gate recomputation")
"""
    + s[b:]
)
a = s.index('    if config.activation_backend in ("inductor_rational"')
b = s.index("    groups = parameter_groups(model, config)", a)
s = s[:a] + s[b:]
write("src/core/trainer.py", s)

s = old("src/core/benchmark.py")
a = s.index('    if config.variant.startswith("paired_"):')
b = s.index('    if config.variant.startswith("blockshuffle"):', a)
s = s[:a] + s[b:]
a = s.index('    if (\n        config.variant.startswith(("multihead_", "headwise_"))')
b = s.index("    total = layers", a)
s = s[:a] + s[b:]
write("src/core/benchmark.py", s)

s = old("src/core/cli.py")
s = s.replace(
    'choices=("eager", "inductor_rational", "inductor_rational_correction"),', 'choices=("eager",),'
)
s = s.replace(
    'ModelConfig(variant=variant, groups=3 if variant == "quadratic_bezier_mixed" else 8)',
    "ModelConfig(variant=variant)",
)
write("src/core/cli.py", s)
for name in (
    "src/blockshuffle_ffn/__init__.py",
    "src/core/optimization.py",
    "src/core/model_audit.py",
    "src/core/serving.py",
    "tests/test_blockshuffle.py",
):
    write(
        name,
        old(name).replace("from src.grouped_ffn import", "from src.core.structured_linear import"),
    )
s = old("src/core/optimization.py").replace(
    "from src.grouped_ffn import", "from src.core.structured_linear import"
)
s = s.replace('if model.config.variant.startswith("shared_") or not all(', "if not all(")
write("src/core/optimization.py", s)

keep_core = set(
    "__init__ analysis benchmark cli compile_serving conditioning config data diagnostics frozen_train_worker fused_serving graph_serving model_audit optimization report reproducibility serving trainer transformer wikitext_data structured_linear".split()
)
keep_tests = set(
    "test_baselines test_blockshuffle test_conditioning test_data test_forward_evaluation test_frozen_train_worker test_graph_serving test_learnable_activation test_model_audit test_optimization test_reporting test_serving test_triton_inference test_width_calibration test_wikitext_data".split()
)
keep_configs = {
    f"{prefix}_{recipe}_800.json"
    for prefix in ("scale384", "wikitext2")
    for recipe in ("full_swiglu", "full_gelu", "calibrated_narrow")
}
keep_configs.update(
    (
        "scale384_blockshuffle_800.json",
        "wikitext2_blockshuffle_stronger_800.json",
        "wikitext2_blockshuffle_rational_screen.json",
    )
)
retire_dirs = [
    f"src/{n}"
    for n in (
        "adaptive_cross_group_ffn",
        "affine_shared_ffn",
        "bezier_ffn",
        "cross_group_ffn",
        "grouped_ffn",
        "learnable_activation_ffn",
        "multihead_ffn",
        "paired_feature_ffn",
        "shared_ffn",
    )
]
retire_files = [p.as_posix() for p in Path("src/core").glob("*.py") if p.stem not in keep_core]
retire_files += [p.as_posix() for p in Path("tests").glob("*.py") if p.stem not in keep_tests]
retire_files += [p.as_posix() for p in Path("configs").glob("*.json") if p.name not in keep_configs]
for name in retire_dirs + retire_files:
    p = ROOT / name
    for f in [p] if p.is_file() else p.rglob("*"):
        if f.is_file() and "__pycache__" not in f.parts:
            relative = f.relative_to(ROOT).as_posix()
            assert hashlib.sha256(f.read_bytes()).hexdigest() == manifest["files"][relative], (
                relative
            )
write(
    "results/verification/cleanup_moves_v1.json",
    json.dumps(
        {
            "directories": retire_dirs,
            "files": sorted(retire_files),
            "destination": "research/archive/retired",
            "kept_configs": sorted(keep_configs),
            "kept_core": sorted(keep_core),
        },
        indent=2,
    )
    + "\n",
)
print(
    json.dumps(
        {
            "status": "edits prepared; moves independently hash-verified",
            "directories_to_retire": len(retire_dirs),
            "files_to_retire": len(retire_files),
            "kept_configs": len(keep_configs),
            "kept_core": len(keep_core),
        }
    )
)
