"""Operator-level diagnostics; instrumented times are not serving benchmarks."""

import argparse
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from src.core.benchmark import autocast, synchronize
from src.core.compile_serving import compile_model, load_model, setup
from src.core.data import TokenData
from src.core.reproducibility import environment, provenance, sha256, write_json


@torch.no_grad()
def audit(run: Path, mode: str, output: Path, fused: bool = False) -> dict:
    if output.exists():
        raise ValueError("Do not overwrite operator profile evidence")
    setup()
    model = load_model(run)
    if fused:
        from src.blockshuffle_ffn.triton_inference import replace_ffns

        replace_ffns(model)
    data = TokenData(Path("data/tinystories_v1"), "cuda", 10017)
    x, _ = next(data.validation(16, model.config.context, 1))
    forward = compile_model(model) if mode == "compiled" else model
    for _ in range(10):
        with autocast("cuda", "bf16"):
            forward(x)
    synchronize("cuda")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        for _ in range(10):
            with autocast("cuda", "bf16"):
                forward(x)
        synchronize("cuda")
    rows = [
        {
            "operator": e.key,
            "calls": e.count,
            "self_cpu_us": e.self_cpu_time_total,
            "self_device_us": e.self_device_time_total,
            "device_us_inclusive": e.device_time_total,
        }
        for e in prof.key_averages()
    ]
    rows.sort(key=lambda r: r["self_device_us"], reverse=True)
    result = {
        "run": run.name,
        "mode": mode,
        "ffn_execution": "four_kernel" if fused else "native",
        "checkpoint_sha256": sha256(run / "checkpoint.pt"),
        "environment": environment(),
        "provenance": provenance(),
        "warmup_forwards": 10,
        "profiled_forwards": 10,
        "rows": rows,
        "device_times_available": any(r["self_device_us"] > 0 for r in rows),
        "interpretation": "Instrumented operator diagnostics on fixed batch16/context128 BF16. Events may include both host operators and individual device kernels; do not sum overlapping categories as a model wall time. Timing overhead, warmup and compilation invalidate comparison with uninstrumented serving throughput.",
    }
    write_json(output, result)
    print(json.dumps({"run": run.name, "mode": mode, "top_events": rows[:12]}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--mode", choices=("eager", "compiled"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fused", action="store_true")
    args = parser.parse_args()
    audit(args.run, args.mode, args.output, args.fused)
