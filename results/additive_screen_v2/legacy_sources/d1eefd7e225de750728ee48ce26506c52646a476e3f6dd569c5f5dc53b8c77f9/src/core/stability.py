"""Depth-at-initialization gradient probes, explicitly not trained scaling evidence."""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from src.core.benchmark import autocast
from src.core.config import ModelConfig
from src.core.diagnostics import gradient_stats, inspect_layers
from src.core.reproducibility import environment, provenance, write_json
from src.core.transformer import Transformer


def run(output: Path, device: str = "cuda") -> dict:
    if output.exists():
        raise ValueError("Do not overwrite a stability experiment")
    torch.set_num_threads(4)
    precision = "bf16" if device.startswith("cuda") and torch.cuda.is_bf16_supported() else "fp32"
    rows = []
    for depth in (4, 12, 24):
        for variant in ("gelu", "swiglu", "gelu_narrow", "bezier_shared", "bezier_grouped"):
            config = ModelConfig(
                variant=variant, vocab_size=256, width=96, heads=3, layers=depth, context=32
            )
            model = Transformer(config, 17).to(device)
            generator = torch.Generator(device=device).manual_seed(71)
            x = torch.randint(256, (2, 32), device=device, generator=generator)
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            with autocast(device, precision):
                loss = model.loss(x, torch.roll(x, -1, 1))
            loss.backward()
            gradients = gradient_stats(model)
            norm = (
                torch.stack([p.grad.float().square().sum() for p in model.parameters()])
                .sum()
                .sqrt()
            )
            diagnostics = inspect_layers(model, x, device, precision)
            rows.append(
                {
                    "config": asdict(config),
                    "loss": loss.item(),
                    "global_gradient_norm": norm.item(),
                    "layer_gradient_norms": gradients,
                    "layer_statistics": diagnostics,
                    "all_gradients_finite": all(
                        bool(torch.isfinite(p.grad).all()) for p in model.parameters()
                    ),
                    "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated()
                    if device.startswith("cuda")
                    else 0,
                }
            )
            del model
    result = {
        "benchmark": "initialization_depth_probe_v1",
        "seed": 17,
        "precision": precision,
        "environment": environment(),
        "provenance": provenance(),
        "rows": rows,
        "limitation": "Random token inputs and one backward pass at initialization; no claim about trained deep-network stability or reasoning.",
    }
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/stability_initialization.json")
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.output, args.device)
