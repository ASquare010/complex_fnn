"""Small, architecture-independent CUDA regression loop with CPU-resident data.

Models provide ``optimizer_groups(rate)`` and ``features(x)`` for shared
initialization-independent optimization and diagnostics. No target knowledge is
passed into the model. Sampling streams and split boundaries belong to callers.
"""

import math
import statistics
import time

import torch
from torch.nn import functional as F


@torch.no_grad()
def regression_score(model, x, y, batch_size: int = 1024) -> float:
    model.eval()
    total = 0.0
    for start in range(0, len(x), batch_size):
        inputs = x[start : start + batch_size].cuda()
        targets = y[start : start + batch_size].cuda()
        total += F.mse_loss(model(inputs), targets, reduction="sum").item()
    model.train()
    return total / y.numel()


@torch.no_grad()
def regression_diagnostics(model, probe, step: int) -> dict:
    before, after = model.features(probe.cuda())
    return {
        "step": step,
        "activations": {
            name: {
                "mean": value.mean().item(),
                "std": value.std(correction=0).item(),
                "max_abs": value.abs().max().item(),
                "finite": bool(torch.isfinite(value).all()),
            }
            for name, value in (("pre", before), ("post", after))
        },
        "gradients": {
            name: None if p.grad is None else p.grad.norm().item()
            for name, p in model.named_parameters()
        },
    }


def fit_regression(model, x, y, stream, rate: float, warmup: int = 50) -> dict:
    if x.device.type != "cpu" or y.device.type != "cpu" or stream.device.type != "cpu":
        raise ValueError("Data/indices must stay on CPU for memory accounting")
    optimizer = torch.optim.AdamW(
        model.optimizer_groups(rate), betas=(0.9, 0.95), eps=1e-8, weight_decay=0
    )
    history = []
    probe = x[: len(stream[0])]
    diagnostics = [regression_diagnostics(model, probe, 0)]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for step, ids in enumerate(stream, 1):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        inputs, targets = x[ids].cuda(), y[ids].cuda()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        loss = F.mse_loss(model(inputs), targets)
        torch.cuda.synchronize()
        forward_end = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward_end = time.perf_counter()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        end = time.perf_counter()
        value, gradient = loss.item(), norm.item()
        if not math.isfinite(value) or not math.isfinite(gradient):
            raise RuntimeError("Nonfinite loss/gradient")
        history.append(
            {
                "step": step,
                "loss": value,
                "preclip_norm": gradient,
                "update_ms": 1000 * (end - begin),
                "forward_ms": 1000 * (forward_end - forward_start),
                "backward_ms": 1000 * (backward_end - forward_end),
            }
        )
        if step in (50, 150, len(stream)):
            diagnostics.append(regression_diagnostics(model, probe, step))
    peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    optimizer_bytes = sum(
        v.numel() * v.element_size()
        for state in optimizer.state.values()
        for v in state.values()
        if isinstance(v, torch.Tensor)
    )
    finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    finite = finite and all(
        bool(torch.isfinite(state[k]).all())
        for state in optimizer.state.values()
        for k in ("exp_avg", "exp_avg_sq")
    )
    if not finite:
        raise RuntimeError("Nonfinite weights or moments")
    model.eval()
    fixed = probe.cuda()
    with torch.no_grad():
        for _ in range(10):
            model(fixed)
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(20):
                model(fixed)
            torch.cuda.synchronize()
            samples.append(1000 * (time.perf_counter() - start) / 20)
    model.train()
    return {
        "history": history,
        "diagnostics": diagnostics,
        "peak_allocated_bytes": peak,
        "peak_reserved_bytes": reserved,
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "optimizer_bytes": optimizer_bytes,
        "all_finite": finite,
        "clip_fraction": sum(r["preclip_norm"] > 1 for r in history) / len(history),
        "inference_ms": statistics.median(samples),
        "inference_samples_ms": samples,
        **{
            f"median_{k}": statistics.median(r[k] for r in history[warmup:])
            for k in ("update_ms", "forward_ms", "backward_ms")
        },
    }
