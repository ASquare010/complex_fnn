"""Cheap held-out function screens. These are not language-model evidence."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.bezier_ffn import BezierActivation
from src.core.benchmark import synchronize
from src.core.reproducibility import environment, provenance, write_json

TASKS = (
    "smooth",
    "oscillatory",
    "piecewise",
    "multiplicative",
    "compositional",
    "parity",
    "random_smooth",
)
VARIANTS = ("gelu", "swiglu", "gelu_narrow", "gelu_budget", "bezier_shared", "bezier_grouped")


def target(name: str, x: torch.Tensor) -> torch.Tensor:
    a, b, c, d = x.unbind(-1)
    if name == "smooth":
        return torch.sin(2 * a) + b.square() + torch.exp(0.5 * c) - d
    if name == "oscillatory":
        return torch.sin(9 * a) * torch.cos(7 * b) + 0.5 * torch.sin(11 * c + 3 * d)
    if name == "piecewise":
        return F.relu(a + b) - 0.7 * (c - d).abs() + torch.where(a > 0, b, c)
    if name == "multiplicative":
        return a * b + 2 * b * c * d + a.square() * d
    if name == "compositional":
        return torch.sin(3 * (a * b + torch.sin(c))) + torch.tanh(2 * (c * d - a))
    if name == "parity":
        return torch.sign(x).prod(-1)
    if name == "random_smooth":
        g = torch.Generator(device=x.device).manual_seed(9281)
        w = torch.randn(4, 12, generator=g, device=x.device) * 2
        phase = torch.randn(12, generator=g, device=x.device)
        amplitudes = torch.randn(12, generator=g, device=x.device)
        return (torch.sin(x @ w + phase) * amplitudes).sum(-1) / math.sqrt(12)
    raise ValueError(name)


class Regressor(nn.Module):
    def __init__(self, variant: str, seed: int) -> None:
        super().__init__()
        hidden = (
            64
            if variant == "gelu"
            else 35
            if variant == "swiglu"
            else 17
            if variant == "gelu_budget"
            else 16
        )
        self.up = nn.Linear(4, hidden)
        self.down = nn.Linear(hidden, 1)
        self.gate = nn.Linear(4, hidden) if variant == "swiglu" else None
        self.activation = (
            BezierActivation(hidden, 1 if variant == "bezier_shared" else 4)
            if variant.startswith("bezier")
            else nn.GELU()
        )
        with torch.no_grad():
            for name, p in self.named_parameters():
                if "theta" in name:
                    continue
                if p.ndim == 1:
                    p.zero_()
                else:
                    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
                    g = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
                    p.normal_(0, 1 / math.sqrt(p.shape[1]), generator=g)

    def forward(self, x):
        z = self.up(x)
        z = F.silu(z) * self.gate(x) if self.gate is not None else self.activation(z)
        return self.down(z).squeeze(-1)


def run(output: Path, steps: int = 300, seed: int = 17, device: str = "cuda") -> dict:
    if output.exists():
        raise ValueError("Do not overwrite function evidence")
    torch.set_num_threads(4)
    rows = []
    for task in TASKS:
        g = torch.Generator(device=device).manual_seed(812)
        train_x = 2 * torch.rand(2048, 4, generator=g, device=device) - 1
        test_x = 2 * torch.rand(4096, 4, generator=g, device=device) - 1
        train_y, test_y = target(task, train_x), target(task, test_x)
        mean, std = train_y.mean(), train_y.std()
        train_y, test_y = (train_y - mean) / std, (test_y - mean) / std
        for variant in VARIANTS:
            model = Regressor(variant, seed).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0)
            sampling = torch.Generator(device=device).manual_seed(seed + 20000)
            synchronize(device)
            start = time.perf_counter()
            max_norm = 0.0
            for _ in range(steps):
                idx = torch.randint(len(train_x), (256,), device=device, generator=sampling)
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(train_x[idx]), train_y[idx])
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float("inf"), error_if_nonfinite=True
                )
                max_norm = max(max_norm, norm.item())
                optimizer.step()
            synchronize(device)
            runtime = time.perf_counter() - start
            with torch.no_grad():
                mse = F.mse_loss(model(test_x), test_y).item()
                train_mse = F.mse_loss(model(train_x), train_y).item()
            row = {
                "task": task,
                "variant": variant,
                "parameters": sum(p.numel() for p in model.parameters()),
                "heldout_normalized_mse": mse,
                "train_normalized_mse": train_mse,
                "max_gradient_norm": max_norm,
                "runtime_seconds": runtime,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = {
        "benchmark": "functions_v1",
        "seed": seed,
        "steps": steps,
        "batch": 256,
        "train_samples": 2048,
        "test_samples": 4096,
        "precision": "fp32",
        "precision_reason": "All candidates use FP32 to screen small approximation errors",
        "environment": environment(),
        "provenance": provenance(),
        "rows": rows,
        "limitations": "Single seed, finite optimization; no LM or novelty claim. Full SwiGLU 386 vs GELU 385; grouped cubic 105 vs GELU budget 103; shared cubic 99 vs narrow GELU 97 parameters.",
    }
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/functions_s17.json"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.output, args.steps, args.seed, args.device)
