"""Fixed-shape CUDA graph serving with equal treatment of the dense reference."""

import argparse
import copy
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from src.core.benchmark import autocast, forward_flops, synchronize
from src.core.config import ModelConfig
from src.core.data import TokenData
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.serving import cache_projections, pack_gate_factors
from src.core.transformer import Transformer


class GraphForward:
    """Inference only. Returned output aliases storage reused by the next call."""

    @torch.no_grad()
    def __init__(self, model, example: torch.Tensor, *, warmup_stream=None) -> None:
        if example.device.type != "cuda" or model.training:
            raise ValueError("Graph serving requires a CUDA input and an evaluation model")
        self.model = model
        self.input = example.clone()
        side = warmup_stream if warmup_stream is not None else torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(4):
                with autocast("cuda", "bf16"):
                    model(self.input)
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with autocast("cuda", "bf16"):
                self.output = model(self.input)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.shape != self.input.shape
            or x.dtype != self.input.dtype
            or x.device != self.input.device
        ):
            raise ValueError("Graph input must match captured shape, dtype and device")
        if self.model.training:
            raise RuntimeError("Graph serving is inference only")
        self.input.copy_(x)
        self.graph.replay()
        return self.output


def build(source, mode):
    model = copy.deepcopy(source)
    cache_bytes = 0
    if mode == "packed_factors_bf16":
        cache_bytes = pack_gate_factors(model, torch.bfloat16)
    elif mode == "dense_cache_bf16":
        cache_bytes = cache_projections(model, torch.bfloat16)
    return model.cuda().eval(), cache_bytes


@torch.no_grad()
def audit(
    run: Path, control: Path, output: Path, cache: Path = Path("data/tinystories_v1")
) -> dict:
    if output.exists():
        raise ValueError("Do not overwrite graph audit evidence")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    sources, metrics = [], []
    for path in (run, control):
        saved = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
        model = Transformer(ModelConfig(**saved["model_config"]))
        model.load_state_dict(saved["model"])
        sources.append(model)
        metrics.append(json.loads((path / "metrics.json").read_text()))
    assert metrics[0]["data"]["files"] == metrics[1]["data"]["files"]
    for key in ("seed", "steps", "batch_size"):
        assert metrics[0]["training"][key] == metrics[1]["training"][key]
    for key in ("width", "layers", "heads", "context", "vocab_size"):
        assert metrics[0]["model"][key] == metrics[1]["model"][key]
    cases = (
        [(name, sources[0]) for name in ("factorized", "packed_factors_bf16", "dense_cache_bf16")]
        if sources[0].config.variant == "blockshuffle_swiglu"
        else [("candidate", sources[0])]
    )
    cases.append(("dense_reference", sources[1]))
    data = TokenData(cache, "cuda", 10017)
    context = sources[0].config.context
    fixed_x, _ = next(data.validation(16, context, 1))
    rows = []
    # Explicit shared warmup stream avoids a new persistent cuBLAS workspace per mode.
    warmup_stream = torch.cuda.Stream()
    for name, source in cases:
        model, cache_bytes = build(source, name)
        # Prime the ordinary stream before every memory measurement as well.
        with autocast("cuda", "bf16"):
            model(fixed_x)
        synchronize("cuda")
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        graph = GraphForward(model, fixed_x, warmup_stream=warmup_stream)
        synchronize("cuda")
        construction_seconds = time.perf_counter() - start
        capture_peak = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        total, count, errors = 0.0, 0, []
        for x, y in data.validation(16, context, 16):
            logits = graph(x)
            total += (
                F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1)).item()
                * y.numel()
            )
            count += y.numel()
        inference_peak = torch.cuda.max_memory_allocated()
        inference_reserved_peak = torch.cuda.max_memory_reserved()
        # Check independent fresh inputs after recording inference memory.
        for x, _ in data.validation(16, context, 3):
            with autocast("cuda", "bf16"):
                expected = model(x)
            actual = graph(x)
            errors.append((expected - actual).abs().max().item())
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        config = model.config
        rows.append(
            {
                "mode": name,
                "validation_loss": total / count,
                "validation_tokens": count,
                "fresh_input_eager_max_abs_errors": errors,
                "graph_construction_seconds": construction_seconds,
                "capture_peak_allocated_vram_bytes": capture_peak,
                "peak_inference_allocated_vram_bytes": inference_peak,
                "peak_inference_reserved_vram_bytes": inference_reserved_peak,
                "cache_bytes": cache_bytes,
                "ffn_matrix_flops_per_token": 6 * config.width * config.ffn_width * config.layers
                if name == "dense_cache_bf16"
                else forward_flops(config)["ffn_matrix_forward_flops_per_token"],
            }
        )
        print(json.dumps(rows[-1]), flush=True)
        del model, graph, logits, expected, actual
        gc.collect()
        torch.cuda.empty_cache()
    graphs = {}
    for name, source in cases:
        model, _ = build(source, name)
        graphs[name] = GraphForward(model, fixed_x, warmup_stream=warmup_stream)
    rounds, names = [], list(graphs)
    for iteration in range(8):
        order = names[iteration % len(names) :] + names[: iteration % len(names)]
        latencies = {}
        for name in order:
            for _ in range(4):
                graphs[name](fixed_x)
            synchronize("cuda")
            start = time.perf_counter()
            for _ in range(30):
                graphs[name](fixed_x)
            synchronize("cuda")
            latencies[name] = (time.perf_counter() - start) / 30
        rounds.append({"round": iteration, "order": order, "seconds_per_forward": latencies})
    ratios = {}
    for name in names:
        samples = [
            r["seconds_per_forward"]["dense_reference"] / r["seconds_per_forward"][name]
            for r in rounds
        ]
        ratios[name] = {
            "samples": samples,
            "median": statistics.median(samples),
            "min": min(samples),
            "max": max(samples),
        }
    result = {
        "memory_protocol": "shared_warmup_stream_v2; ordinary stream primed before capture; no per-case workspace accumulation",
        "run": run.name,
        "control": control.name,
        "checkpoint_sha256": sha256(run / "checkpoint.pt"),
        "control_checkpoint_sha256": sha256(control / "checkpoint.pt"),
        "environment": environment(),
        "provenance": provenance(),
        "data_hashes": data.manifest["files"],
        "rows": rows,
        "paired_timing": {
            "rounds": rounds,
            "throughput_ratio_to_dense_reference": ratios,
            "warmup_forwards_per_round": 4,
            "timed_forwards_per_round": 30,
        },
        "limitations": "Fixed-shape full-sequence CUDA graph replay, batch 16, context 128, BF16. Includes device-to-device input copy; excludes graph construction and host data loading from replay timing. All modes resident during rotating timing; memory measured separately with one model including graph storage. No training, arbitrary-shape or KV-cache generation speed claim.",
    }
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.control, args.output)
