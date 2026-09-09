"""Compare retained model behavior across cleanup; no corpus training."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.core.benchmark import autocast, forward_flops
from src.core.config import ModelConfig, TrainConfig
from src.core.diagnostics import gradient_stats, inspect_layers
from src.core.optimization import group_summary, initialize_dense_width, parameter_groups
from src.core.trainer import configure_gate_recomputation
from src.core.transformer import Transformer


def digest(tensor):
    value = tensor.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def signatures():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = {}
    for device in ("cpu", "cuda"):
        assert device != "cuda" or torch.cuda.is_available()
        precision = "bf16" if device == "cuda" else "fp32"
        for variant in (
            "gelu",
            "swiglu",
            "gelu_narrow",
            "swiglu_narrow",
            "blockshuffle_swiglu",
            "blockshuffle_swiglu_rational",
        ):
            cfg = ModelConfig(
                variant=variant,
                width=24,
                layers=2,
                heads=3,
                hidden=48 if variant.startswith("blockshuffle") else 0,
                groups=8,
                vocab_size=32,
                context=8,
            )
            tc = TrainConfig(
                device=device,
                precision=precision,
                ffn_lr_mode="fan_in" if variant.startswith("blockshuffle") else "uniform",
                ffn_width_init_mode="fan_in" if variant.endswith("narrow") else "none",
                ffn_width_lr_mode="fan_in" if variant.endswith("narrow") else "none",
                ffn_decay_mode="product" if variant.endswith("narrow") else "parameter",
                recompute_gate=variant.startswith("blockshuffle"),
                gate_recompute_method="native"
                if variant == "blockshuffle_swiglu"
                else "checkpoint",
            )
            model = Transformer(cfg, 29).to(device)
            initialize_dense_width(model, tc)
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if "theta" in name:
                        p.copy_(torch.linspace(-0.2, 0.3, p.numel(), device=device).reshape_as(p))
            configure_gate_recomputation(model, tc)
            groups = parameter_groups(model, tc)
            record = {
                "initial": {n: digest(p) for n, p in model.named_parameters()},
                "optimizer_groups": group_summary(groups),
                "total_parameters": cfg.total_parameters,
                "ffn_parameters": cfg.unique_ffn_parameters,
                "flops": forward_flops(cfg),
                "steps": [],
            }
            optimizer = torch.optim.AdamW(groups, lr=tc.learning_rate, betas=(0.9, 0.95), eps=1e-8)
            x = torch.arange(16, device=device).reshape(2, 8)
            y = (x + 1) % 32
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                with autocast(device, precision):
                    logits = model(x)
                    loss = model.loss(x, y)
                loss.backward()
                step = {
                    "logits": digest(logits),
                    "loss": loss.item(),
                    "gradients": {n: digest(p.grad) for n, p in model.named_parameters()},
                    "gradient_stats": gradient_stats(model),
                }
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                step["updated_parameters"] = {n: digest(p) for n, p in model.named_parameters()}
                record["steps"].append(step)
            record["diagnostics"] = inspect_layers(model, x, device, precision)
            records[f"{device}/{variant}"] = record
            del optimizer, groups, model
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("before", "after"))
    args = parser.parse_args()
    root = Path("results/verification")
    path = root / f"cleanup_{args.phase}_signatures_v1.json"
    assert not path.exists()
    result = signatures()
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.phase == "after":
        before = json.loads((root / "cleanup_before_signatures_v1.json").read_text())
        assert result == before, "Retained behavior changed; inspect signatures before proceeding"
    print(
        json.dumps(
            {
                "phase": args.phase,
                "cases": len(result),
                "status": "PASS",
                "exact_comparison": args.phase == "after",
            }
        ),
        flush=True,
    )
