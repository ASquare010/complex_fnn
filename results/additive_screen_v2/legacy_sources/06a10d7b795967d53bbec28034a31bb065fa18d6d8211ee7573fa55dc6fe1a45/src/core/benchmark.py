"""Synchronized eager benchmarks; inference is full-sequence throughput."""

import time
from contextlib import nullcontext

import torch


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def autocast(device: str, precision: str):
    return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


@torch.no_grad()
def evaluate(model, batches, device: str, precision: str) -> tuple[float, int]:
    model.eval()
    total, count = 0.0, 0
    for x, y in batches:
        with autocast(device, precision):
            loss = model.loss(x, y)
        total += loss.item() * y.numel()
        count += y.numel()
    if not count:
        raise ValueError("No evaluation targets")
    model.train()
    return total / count, count


@torch.no_grad()
def evaluate_forward(forward, batches, device: str, precision: str) -> tuple[float, int]:
    """Evaluate the supplied callable; its caller manages evaluation mode.

    Compiled wrappers may delegate .loss to the original module, bypassing
    compiled forward. Computing loss here avoids that silent validation error.
    """
    total, count = 0.0, 0
    for x, y in batches:
        with autocast(device, precision):
            logits = forward(x)
            loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        total += loss.item() * y.numel()
        count += y.numel()
    if not count:
        raise ValueError("No evaluation targets")
    return total / count, count


@torch.no_grad()
def inference_benchmark(
    model, x: torch.Tensor, device: str, precision: str, warmup: int = 10, repeats: int = 30
) -> dict:
    model.eval()
    for _ in range(warmup):
        with autocast(device, precision):
            model(x)
    synchronize(device)
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(repeats):
            with autocast(device, precision):
                model(x)
        synchronize(device)
        samples.append((time.perf_counter() - start) / repeats)
    samples.sort()
    model.train()
    return {
        "inference_tokens_per_second": x.numel() / samples[1],
        "inference_latency_ms": samples[1] * 1000,
        "inference_latency_samples_ms": [s * 1000 for s in samples],
        "inference_mode": "full_sequence_no_kv_cache",
        "warmup_forwards": warmup,
        "timed_forwards_per_repeat": repeats,
    }


def forward_flops(config) -> dict:
    d, h, t, layers, v = (
        config.width,
        config.ffn_width,
        config.context,
        config.layers,
        config.vocab_size,
    )
    ffn = (6 if "swiglu" in config.variant else 4) * d * h
    if config.variant.startswith("paired_"):
        ffn = 4 * d * h
    if config.variant.startswith("structured"):
        ffn //= config.groups
    if config.variant.startswith("blockshuffle"):
        ffn = 2 * config.ffn_parameters
    total = layers * (8 * d * d + 4 * t * d + ffn) + 2 * d * v
    return {
        "ffn_matrix_forward_flops_per_token": ffn * layers,
        "model_matrix_forward_flops_per_token": total,
        "training_matrix_flops_per_token_estimate": 3 * total,
        "flop_limitations": "Dense attention upper estimate; excludes norms, softmax, nonlinearities, loss and optimizer",
    }
