"""Bounded initialization and GPU pipeline qualification for the multi-head comparator."""

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
import traceback
import zipfile
from dataclasses import asdict
from pathlib import Path

import torch

from src.core.benchmark import autocast, forward_flops
from src.core.config import VARIANTS, ModelConfig, TrainConfig
from src.core.data import TokenData
from src.core.diagnostics import gradient_stats, inspect_layers
from src.core.optimization import group_summary, parameter_groups
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.transformer import Transformer

PLAN = Path("research/multihead_integration_plan.md")
VARIANTS_NEW = ("multihead_swiglu", "multihead_swiglu_calibrated")


def configuration(variant):
    return ModelConfig(
        variant=variant,
        width=384,
        layers=8,
        heads=6,
        context=128,
        vocab_size=4096,
        hidden=24,
        groups=48,
    )


def cpu_preflight():
    torch.set_num_threads(4)
    snapshot = Path("results/verification/multihead_before_v1.pt")
    meta = json.loads(snapshot.with_suffix(".json").read_text())
    assert sha256(snapshot) == meta["sha256"]
    before = torch.load(snapshot, map_location="cpu", weights_only=True)
    for variant, row in before.items():
        config = ModelConfig(**row["config"])
        model = Transformer(config, 17)
        x = torch.arange(16).reshape(2, 8)
        loss = model.loss(x, x + 1)
        loss.backward()
        assert torch.equal(model(x), row["logits"]) and torch.equal(loss, row["loss"])
        assert all(torch.equal(t, row["state"][n]) for n, t in model.state_dict().items())
        assert all(torch.equal(p.grad, row["gradients"][n]) for n, p in model.named_parameters())
        assert forward_flops(config) == row["flops"]
    assert set(VARIANTS) - set(before) == set(VARIANTS_NEW)
    x = torch.randn((4096, 384), generator=torch.Generator().manual_seed(73))
    moments = {}
    for variant in ("swiglu", *VARIANTS_NEW):
        config = (
            configuration(variant)
            if variant in VARIANTS_NEW
            else ModelConfig(
                variant="swiglu", width=384, layers=8, heads=6, vocab_size=4096, context=128
            )
        )
        model = Transformer(config, 17)
        ffn = model.blocks[0].ffn
        with torch.no_grad():
            y = ffn(x)
        moments[variant] = {
            "rms": y.square().mean().sqrt().item(),
            "std": y.std(unbiased=False).item(),
            "mean": y.mean().item(),
            "finite": bool(torch.isfinite(y).all()),
            "ffn_parameters": config.unique_ffn_parameters,
            "total_parameters": config.total_parameters,
        }
        del model, ffn, y
    ratio = moments["multihead_swiglu_calibrated"]["rms"] / moments["swiglu"]["rms"]
    assert all(r["finite"] for r in moments.values())
    return {
        "old_variants_exact": len(before),
        "old_snapshot_sha256": sha256(snapshot),
        "new_variants": VARIANTS_NEW,
        "moments": moments,
        "calibrated_full_rms_ratio": ratio,
        "calibration_ratio_passes": 0.5 <= ratio <= 2.0,
        "input": "4096 independent standard Gaussian rows, width384, CPU FP32 generator seed73; first FFN of layer0/seed17; no validation data",
        "environment": environment(),
        "provenance": provenance(),
    }


def gpu_worker(variant, output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = configuration(variant)
    training = TrainConfig(
        steps=10,
        batch_size=16,
        learning_rate=0.0006,
        weight_decay=0.1,
        seed=17,
        eval_batches=1,
        log_every=10,
        precision="bf16",
        device="cuda",
        ffn_lr_mode="uniform",
    )
    model = Transformer(config, 17).cuda()
    data = TokenData(Path("data/wikitext2_v1"), "cuda", 10017)
    x, y = data.batch(16, 128)
    batch_hash = hashlib.sha256(x.cpu().numpy().tobytes() + y.cpu().numpy().tobytes()).hexdigest()
    with torch.no_grad():
        expected = model(x).float().cpu()
        with autocast("cuda", "bf16"):
            actual = model(x).float().cpu()
    difference = actual - expected
    error = {
        "relative_l2": difference.norm().item() / max(expected.norm().item(), 1e-30),
        "max_abs": difference.abs().max().item(),
        "rms": difference.square().mean().sqrt().item(),
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
    }
    del actual, expected, difference
    assert error["finite"] and error["relative_l2"] <= 0.05
    initial = inspect_layers(model, x, "cuda", "bf16")
    write_json(output / "initial_diagnostics.json", initial)
    assert all(r["finite"] and r.get("routing_finite", True) for r in initial.values())
    groups = parameter_groups(model, training)
    optimizer = torch.optim.AdamW(groups, lr=training.learning_rate, betas=(0.9, 0.95), eps=1e-8)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    history = []
    for step in range(10):
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", "bf16"):
            loss = model.loss(x, y)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        routers = {
            f"layer_{i}": b.ffn.router.grad.float().norm().item()
            for i, b in enumerate(model.blocks)
        }
        row = {
            "step": step + 1,
            "fixed_training_batch_loss": loss.item(),
            "gradient_norm_pre_clip": norm.item(),
            "layer_gradient_norms_post_clip": gradient_stats(model),
            "router_gradient_norms_post_clip": routers,
        }
        assert math.isfinite(row["fixed_training_batch_loss"]) and all(
            math.isfinite(g) and g > 0 for g in routers.values()
        )
        optimizer.step()
        history.append(row)
        with (output / "history.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(row, allow_nan=False) + "\n")
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    final = inspect_layers(model, x, "cuda", "bf16")
    assert all(r["finite"] and r.get("routing_finite", True) for r in final.values())
    write_json(output / "final_diagnostics.json", final)
    torch.save(
        {
            "model": {n: t.detach().cpu() for n, t in model.state_dict().items()},
            "model_config": asdict(config),
            "training_config": asdict(training),
            "step": 10,
            "scope": "Ten constant-rate pipeline updates on one repeated training batch; not an LM validation result",
        },
        output / "checkpoint.pt",
    )
    result = {
        "status": "PASS",
        "variant": variant,
        "model": asdict(config),
        "training": asdict(training),
        "learning_rate_schedule": "constant for this pipeline test only",
        "precision_error": error,
        "initial_fixed_batch_loss": history[0]["fixed_training_batch_loss"],
        "last_preupdate_fixed_batch_loss": history[-1]["fixed_training_batch_loss"],
        "peak_training_allocated_vram_bytes": peak,
        "clipped_step_fraction": sum(r["gradient_norm_pre_clip"] > 1 for r in history) / 10,
        "every_router_has_finite_nonzero_gradients": True,
        "all_recorded_diagnostics_finite": True,
        "optimizer_parameter_groups": group_summary(groups),
        "unique_batch_tokens": y.numel(),
        "training_token_exposures": 10 * y.numel(),
        "batch_sha256": batch_hash,
        "data_hashes": data.manifest["files"],
        "checkpoint_sha256": sha256(output / "checkpoint.pt"),
        "plan_sha256": sha256(PLAN),
        "environment": environment(),
        "provenance": provenance(),
        "scope": "GPU pipeline qualification only; no validation/test scoring, convergence or performance promotion",
    }
    write_json(output / "result.json", result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in (
                    "provenance",
                    "data_hashes",
                    "model",
                    "training",
                    "optimizer_parameter_groups",
                )
            }
        ),
        flush=True,
    )


def audit(output):
    output.mkdir(parents=True, exist_ok=False)
    completed = []
    try:
        preflight = cpu_preflight()
        write_json(output / "preflight.json", preflight)
        protocol = {
            "plan_sha256": sha256(PLAN),
            "provenance": provenance(),
            "environment": environment(),
            "variants": VARIANTS_NEW,
            "worker_timeout_seconds": 600,
            "optimizer_updates_per_worker": 10,
            "validation_tokens": 0,
            "test_tokens": 0,
        }
        write_json(output / "protocol.json", protocol)
        with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for n in protocol["provenance"]["source_files"]:
                archive.write(n, n)
            archive.write(PLAN, PLAN.as_posix())
        if not preflight["calibration_ratio_passes"]:
            write_json(
                output / "result.json",
                {
                    "status": "CALIBRATION_REJECTION",
                    "preflight": preflight,
                    "earns_training_screen": False,
                },
            )
            return
        batch_hash = None
        for variant in VARIANTS_NEW:
            print(f"Starting {variant} pipeline qualification", flush=True)
            write_json(
                output / "progress.json",
                {"status": "running", "active": variant, "completed": completed},
            )
            with (output / f"{variant}.log").open("x", encoding="utf-8") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-X",
                        "faulthandler",
                        "-m",
                        "src.core.multihead_integration",
                        "worker",
                        "--variant",
                        variant,
                        "--output",
                        str(output / variant),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=600,
                )
            r = json.loads((output / variant / "result.json").read_text())
            assert (
                r["status"] == "PASS"
                and r["provenance"]["source_files"] == protocol["provenance"]["source_files"]
            )
            if batch_hash is None:
                batch_hash = r["batch_sha256"]
            assert r["batch_sha256"] == batch_hash
            completed.append(
                {
                    "variant": variant,
                    "status": r["status"],
                    "peak_mib": r["peak_training_allocated_vram_bytes"] / 2**20,
                    "result_sha256": sha256(output / variant / "result.json"),
                }
            )
            print(json.dumps(completed[-1]), flush=True)
        result = {
            "status": "complete",
            "cases": completed,
            "earns_training_screen": True,
            "old_variants_exact": preflight["old_variants_exact"],
            "calibrated_full_rms_ratio": preflight["calibrated_full_rms_ratio"],
            "same_fixed_training_batch": True,
            "plan_sha256": sha256(PLAN),
        }
        write_json(output / "result.json", result)
        write_json(output / "progress.json", {"status": "complete", "completed": completed})
        print(json.dumps(result), flush=True)
    except Exception as exc:
        failure = {
            "status": "failed",
            "completed": completed,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output / "failure.json", failure)
        write_json(output / "progress.json", failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cpu", "worker", "audit"))
    parser.add_argument("--variant", choices=VARIANTS_NEW)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "cpu":
        assert not args.output.exists()
        result = cpu_preflight()
        write_json(args.output, result)
        print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, indent=2))
    elif args.command == "worker":
        if args.variant is None:
            parser.error("worker requires variant")
        gpu_worker(args.variant, args.output)
    else:
        audit(args.output)
