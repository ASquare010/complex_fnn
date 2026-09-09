"""Measure factorized and explicitly cached dense inference separately."""

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.blockshuffle_ffn import BlockShuffleFFN, BlockShuffleLinear, unshuffle_channels
from src.core.benchmark import autocast, evaluate, forward_flops, inference_benchmark, synchronize
from src.core.config import ModelConfig
from src.core.data import TokenData
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.structured_linear import shuffle_channels
from src.core.transformer import Transformer


class CachedProjection(nn.Module):
    """Retain learned factors and account for an additional dense inference buffer."""

    def __init__(self, base: BlockShuffleLinear, cache_dtype: torch.dtype) -> None:
        super().__init__()
        self.base = base
        with torch.no_grad():
            eye = torch.eye(
                base.input_width, device=base.first.weight.device, dtype=base.first.weight.dtype
            )
            weight = base(eye).T.contiguous()
        self.register_buffer("weight", weight.to(cache_dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError(
                "Cached projections support inference only; train the original factors"
            )
        return F.linear(x, self.weight)


class PackedGateFFN(nn.Module):
    """Batch the up/value factors together, retaining factorized matrix work."""

    def __init__(self, base: BlockShuffleFFN, dtype: torch.dtype) -> None:
        super().__init__()
        if type(base) is not BlockShuffleFFN or base.gate is None:
            raise ValueError(
                "Packed serving requires plain BlockShuffle; learned activations are unsupported"
            )
        self.base = base
        self.groups = base.up.groups
        self.middle = base.up.first.output_width
        self.width = base.up.input_width
        self.hidden = base.up.output_width
        self.register_buffer(
            "first_weight",
            torch.cat((base.up.first.weight, base.gate.first.weight)).detach().to(dtype),
        )
        self.register_buffer(
            "second_weight",
            torch.cat((base.up.second.weight, base.gate.second.weight)).detach().to(dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError("Packed projections support inference only")
        grouped = x.reshape(-1, self.groups, self.width // self.groups).transpose(0, 1)
        projected = torch.bmm(torch.cat((grouped, grouped)), self.first_weight.transpose(1, 2))
        first = (
            projected.reshape(2, self.groups, -1, self.middle // self.groups)
            .transpose(1, 2)
            .reshape(2, -1, self.middle)
        )
        mixed = shuffle_channels(first, self.groups)
        regrouped = (
            mixed.reshape(2, -1, self.groups, self.middle // self.groups)
            .transpose(1, 2)
            .reshape(2 * self.groups, -1, self.middle // self.groups)
        )
        second = torch.bmm(regrouped, self.second_weight.transpose(1, 2))
        second = (
            second.reshape(2, self.groups, -1, self.hidden // self.groups)
            .transpose(1, 2)
            .reshape(2, *x.shape[:-1], self.hidden)
        )
        u, v = unshuffle_channels(second, self.groups).unbind(0)
        return self.base.down(F.silu(u) * v)


def pack_gate_factors(model, dtype: torch.dtype) -> int:
    replacements = [PackedGateFFN(block.ffn, dtype) for block in model.blocks]
    total = 0
    for block, replacement in zip(model.blocks, replacements, strict=True):
        block.ffn = replacement
        total += sum(
            t.numel() * t.element_size() for t in (block.ffn.first_weight, block.ffn.second_weight)
        )
    return total


def cache_projections(model, dtype: torch.dtype) -> int:
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, BlockShuffleLinear)
    ]
    for name, module in modules:
        parent_name, attribute = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), attribute, CachedProjection(module, dtype))
    return sum(
        m.weight.numel() * m.weight.element_size()
        for m in model.modules()
        if isinstance(m, CachedProjection)
    )


@torch.no_grad()
def paired_timing(cases, x, rounds: int = 6, repeats: int = 15) -> dict:
    """Rotate mode order; these co-resident models are used for timing only."""
    models = {}
    for name, dtype, source in cases:
        if name == "dense_cache_fp32":
            continue
        model = copy.deepcopy(source)
        if name == "packed_factors_bf16":
            pack_gate_factors(model, dtype)
        elif dtype is not None:
            cache_projections(model, dtype)
        models[name] = model.cuda().eval()
    names = list(models)
    records = []
    for iteration in range(rounds):
        order = names[iteration % len(names) :] + names[: iteration % len(names)]
        times = {}
        for name in order:
            model = models[name]
            for _ in range(4):
                with autocast("cuda", "bf16"):
                    model(x)
            synchronize("cuda")
            start = time.perf_counter()
            for _ in range(repeats):
                with autocast("cuda", "bf16"):
                    model(x)
            synchronize("cuda")
            times[name] = (time.perf_counter() - start) / repeats
        records.append({"round": iteration, "order": order, "seconds_per_forward": times})
    ratios = {}
    if "dense_reference" in names:
        for name in names:
            values = [
                row["seconds_per_forward"]["dense_reference"] / row["seconds_per_forward"][name]
                for row in records
            ]
            ratios[name] = {
                "samples": values,
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
    result = {
        "rounds": records,
        "warmup_forwards_per_mode_per_round": 4,
        "timed_forwards_per_mode_per_round": repeats,
        "throughput_ratio_to_dense_reference": ratios,
        "notes": "Rotating order reduces timing-order bias. All modes are resident together for this timing only; memory figures come from the separate single-model audit. Clock variation can remain. Full-sequence BF16 forward; no KV cache.",
    }
    del models, model
    torch.cuda.empty_cache()
    return result


def audit(
    run: Path, output: Path, cache: Path = Path("data/tinystories_v1"), control: Path | None = None
) -> dict:
    if output.exists():
        raise ValueError("Do not overwrite serving evidence")
    torch.set_num_threads(4)
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["model_config"])
    original = Transformer(config)
    original.load_state_dict(checkpoint["model"])
    torch.backends.cuda.matmul.allow_tf32 = False
    data = TokenData(cache, "cuda", 10017)
    fixed_x, _ = next(data.validation(16, config.context, 1))
    rows = []
    cases = (
        [
            ("factorized", None, original),
            ("dense_cache_fp32", torch.float32, original),
            ("dense_cache_bf16", torch.bfloat16, original),
            ("packed_factors_bf16", torch.bfloat16, original),
        ]
        if config.variant == "blockshuffle_swiglu"
        else [("candidate", None, original)]
    )
    if control is not None:
        candidate_metrics = json.loads((run / "metrics.json").read_text())
        control_metrics = json.loads((control / "metrics.json").read_text())
        assert candidate_metrics["data"]["files"] == control_metrics["data"]["files"], (
            "Data mismatch"
        )
        assert all(
            candidate_metrics["training"][k] == control_metrics["training"][k]
            for k in ("steps", "seed", "batch_size")
        ), "Control training budget mismatch"
        for key in ("width", "layers", "heads", "context", "vocab_size"):
            assert candidate_metrics["model"][key] == control_metrics["model"][key], (
                "Non-FFN shape mismatch"
            )
        state = torch.load(control / "checkpoint.pt", map_location="cpu", weights_only=True)
        reference = Transformer(ModelConfig(**state["model_config"]))
        reference.load_state_dict(state["model"])
        cases.append(("dense_reference", None, reference))
    for name, dtype, source in cases:
        model = copy.deepcopy(source)
        start = time.perf_counter()
        if name == "packed_factors_bf16":
            cache_bytes = pack_gate_factors(model, dtype)
        else:
            cache_bytes = cache_projections(model, dtype) if dtype is not None else 0
        construction_seconds = time.perf_counter() - start
        model = model.cuda().eval()
        synchronize("cuda")
        torch.cuda.reset_peak_memory_stats()
        loss, tokens = evaluate(model, data.validation(16, config.context, 16), "cuda", "bf16")
        timing = inference_benchmark(model, fixed_x, "cuda", "bf16")
        rows.append(
            {
                "mode": name,
                "unique_ffn_parameters": model.config.unique_ffn_parameters,
                "ffn_matrix_flops_per_token": (6 * config.width * config.ffn_width * config.layers)
                if name.startswith("dense_cache")
                else forward_flops(model.config)["ffn_matrix_forward_flops_per_token"],
                "validation_loss": loss,
                "validation_tokens": tokens,
                "cache_bytes": cache_bytes,
                "cache_construction_cpu_seconds": construction_seconds,
                "learned_parameter_bytes": sum(
                    p.numel() * p.element_size() for p in model.parameters()
                ),
                "peak_inference_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
                **timing,
            }
        )
        del model
        torch.cuda.empty_cache()
    paired = paired_timing(cases, fixed_x)
    result = {
        "paired_timing": paired,
        "run": run.name,
        "checkpoint_sha256": sha256(run / "checkpoint.pt"),
        "environment": environment(),
        "provenance": provenance(),
        "rows": rows,
        "source_unique_ffn_parameters": config.unique_ffn_parameters,
        "control": str(control) if control is not None else None,
        "control_checkpoint_sha256": sha256(control / "checkpoint.pt")
        if control is not None
        else None,
        "notes": "Post-training inference audit, batch16, full sequence, no optimizer states. Cached projections retain original factors and allocate extra buffers. BF16 factorized arithmetic and merged arithmetic can differ; each is evaluated separately. Cache construction is excluded from steady-state timing and reported separately. No training-speed claim.",
    }
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    args = parser.parse_args()
    audit(args.run, args.output, control=args.control)
