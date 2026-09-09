"""Controlled function experiment for the cross-group separation theorem."""

import argparse
import hashlib
import json
import math
import time
import traceback
import zipfile
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.adaptive_cross_group_ffn import AdaptiveCoupledFFN
from src.bezier_ffn import QuadraticBezierFFN
from src.core.benchmark import synchronize
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.cross_group_ffn import CoupledStructuredFFN
from src.dense_ffn import DenseFFN
from src.grouped_ffn import StructuredFFN
from src.paired_feature_ffn import PairedFeatureFFN

TASKS = ("constructed_polynomial", "additive_control", "pure_product")
VARIANTS = ("swiglu", "swiglu_narrow", "structured_swiglu", "structured_swiglu_coupled")


def target(task: str, x: torch.Tensor) -> torch.Tensor:
    a, d = x[:, 0], x[:, 3]
    if task == "constructed_polynomial":
        return a * (a + 0.5 * d)
    if task == "additive_control":
        return a.square() + 0.5 * d.square()
    if task == "pure_product":
        return a * d
    raise ValueError(task)


class InteractionRegressor(nn.Module):
    """Architecture from the LM; fixed sum readout and one common scalar bias."""

    def __init__(self, variant: str, seed: int) -> None:
        super().__init__()
        if "quadratic" in variant:
            gated = variant == "swiglu_quadratic"
            self.ffn = QuadraticBezierFFN(
                4,
                4 if gated else 6,
                1 if gated else 3,
                gated=gated,
                mixed=variant == "quadratic_bezier_mixed",
            )
        elif variant == "gelu":
            self.ffn = DenseFFN(4, 6, "gelu")
        elif variant.startswith("paired_"):
            self.ffn = PairedFeatureFFN(4, 6, variant.split("_")[1])
        elif variant == "structured_swiglu_adaptive":
            self.ffn = AdaptiveCoupledFFN(4, 16, 4)
        elif variant == "structured_swiglu_coupled":
            self.ffn = CoupledStructuredFFN(4, 16, 4)
        elif variant == "structured_swiglu":
            self.ffn = StructuredFFN(4, 16, 4, gated=True)
        elif variant in ("swiglu", "swiglu_narrow", "swiglu_budget"):
            self.ffn = DenseFFN(
                4, 16 if variant == "swiglu" else 5 if variant == "swiglu_budget" else 4, "swiglu"
            )
        else:
            raise ValueError(variant)
        self.bias = nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            for name, parameter in self.ffn.named_parameters():
                if "theta" in name:
                    parameter.zero_()
                    continue
                digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
                generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
                # Common unit fan-in variance, using each actual sparse block's fan-in.
                parameter.normal_(0, 1 / math.sqrt(parameter.shape[-1]), generator=generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x).sum(-1) + self.bias


def run(
    root: Path,
    steps: int = 1000,
    seeds: tuple[int, ...] = (17, 29, 43),
    device="cuda",
    variants: tuple[str, ...] = VARIANTS,
) -> dict:
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    env, prov = environment(), provenance()
    write_json(
        root / "protocol.json",
        {
            "name": "cross_group_functions_v1",
            "variants": variants,
            "steps": steps,
            "seeds": seeds,
            "device": device,
            "precision": "fp32",
            "batch_size": 256,
            "learning_rate": 0.003,
            "weight_decay": 0,
            "adam_betas": [0.9, 0.999],
            "adam_epsilon": 1e-8,
            "gradient_clipping": "none; nonfinite gradients stop the trial",
            "environment": env,
            "provenance": prov,
            "timing": "Serial GPU trials; training excludes first 10 steps and evaluations; eager FP32.",
            "interpretation": "Chosen mechanism diagnostics, not broad function or LM superiority.",
        },
    )
    with zipfile.ZipFile(root / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in prov["source_files"]:
            archive.write(name, name)
    generator = torch.Generator().manual_seed(812)
    train_x = 2 * torch.rand(2048, 4, generator=generator) - 1
    test_x = 2 * torch.rand(4096, 4, generator=generator) - 1
    torch.save({"train_x": train_x, "test_x": test_x}, root / "inputs.pt")
    inputs_hash = sha256(root / "inputs.pt")
    train_x, test_x = train_x.to(device), test_x.to(device)
    rows = []
    for task in TASKS:
        train_y, test_y = target(task, train_x), target(task, test_x)
        mean, std = train_y.mean(), train_y.std()
        train_y, test_y = (train_y - mean) / std, (test_y - mean) / std
        for seed in seeds:
            for variant in variants:
                output = root / f"{task}_{variant}_s{seed}"
                output.mkdir()
                config = {
                    "task": task,
                    "seed": seed,
                    "variant": variant,
                    "train_target_mean": mean.item(),
                    "train_target_std": std.item(),
                    "inputs_sha256": inputs_hash,
                    "protocol": "../protocol.json",
                    "population_additive_unnormalized_mse_floor": 1 / 36
                    if task == "constructed_polynomial"
                    else 1 / 9
                    if task == "pure_product"
                    else 0,
                }
                write_json(output / "config.json", config)
                try:
                    model = InteractionRegressor(variant, seed).to(device)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0)
                    sampling = torch.Generator(device=device).manual_seed(seed + 20000)
                    synchronize(device)
                    if device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats()
                    elapsed = 0.0
                    max_norm = 0.0
                    history = []
                    for step in range(steps):
                        synchronize(device)
                        start = time.perf_counter()
                        idx = torch.randint(2048, (256,), generator=sampling, device=device)
                        optimizer.zero_grad(set_to_none=True)
                        loss = F.mse_loss(model(train_x[idx]), train_y[idx])
                        loss.backward()
                        norm = nn.utils.clip_grad_norm_(
                            model.parameters(), float("inf"), error_if_nonfinite=True
                        )
                        max_norm = max(max_norm, norm.item())
                        optimizer.step()
                        synchronize(device)
                        if step >= 10:
                            elapsed += time.perf_counter() - start
                        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
                            with torch.no_grad():
                                history.append(
                                    {
                                        "step": step + 1,
                                        "training_batch_mse": loss.item(),
                                        "heldout_normalized_mse": F.mse_loss(
                                            model(test_x), test_y
                                        ).item(),
                                        "gradient_norm": norm.item(),
                                    }
                                )
                    peak = torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
                    with torch.no_grad():
                        train_mse = F.mse_loss(model(train_x), train_y).item()
                    row = {
                        **config,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "heldout_normalized_mse": history[-1]["heldout_normalized_mse"],
                        "training_normalized_mse": train_mse,
                        "max_gradient_norm": max_norm,
                        "timed_training_seconds": elapsed,
                        "training_examples_per_second": max(0, steps - 10) * 256 / elapsed
                        if elapsed
                        else None,
                        "peak_allocated_vram_bytes": peak,
                    }
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "sampling_rng": sampling.get_state(),
                            "step": steps,
                            "config": config,
                        },
                        output / "checkpoint.pt",
                    )
                    write_json(output / "history.json", history)
                    write_json(output / "metrics.json", row)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    del optimizer, model
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
                except Exception as exc:
                    write_json(
                        output / "failure.json",
                        {**config, "error": repr(exc), "traceback": traceback.format_exc()},
                    )
                    raise
    result = {"benchmark": "cross_group_functions_v1", "rows": rows}
    write_json(root / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/interactions_v1"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.root, steps=args.steps, device=args.device)
