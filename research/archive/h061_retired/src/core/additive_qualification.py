"""Frozen synthetic full-model qualification for additive block/low-rank SwiGLU."""

import argparse
import hashlib
import json
import statistics
import time
import traceback
import zipfile
from dataclasses import asdict
from pathlib import Path

import torch

from src.core.benchmark import autocast, forward_flops, synchronize
from src.core.config import ModelConfig, TrainConfig
from src.core.optimization import group_summary, initialize_dense_width, parameter_groups
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.trainer import configure_gate_recomputation, learning_rate
from src.core.transformer import Transformer

ROOT = Path("results/additive_qualification_v1")
PLAN = Path("research/additive_block_lowrank_qualification_plan.md")
PLAN_SHA = "288dc7e764677eea77faa3930aacda2e3af55814ff5d95faab4d52fda896a77e"
RECIPES = ("full_swiglu", "full_gelu", "calibrated_narrow", "blockshuffle", "additive")
SEEDS = (17, 29, 43)


def configurations(recipe, seed):
    variants = {
        "full_swiglu": "swiglu",
        "full_gelu": "gelu",
        "calibrated_narrow": "swiglu_narrow",
        "blockshuffle": "blockshuffle_swiglu",
        "additive": "additive_block_lowrank_swiglu",
    }
    config = ModelConfig(
        variant=variants[recipe],
        width=384,
        layers=8,
        hidden={"calibrated_narrow": 304, "blockshuffle": 2048}.get(recipe, 0),
    )
    tc = TrainConfig(
        steps=20,
        batch_size=16,
        learning_rate=0.0006,
        seed=seed,
        eval_batches=1,
        log_every=20,
        precision="bf16",
        device="cuda",
        ffn_lr_mode="perturbation"
        if recipe == "additive"
        else "fan_in"
        if recipe == "blockshuffle"
        else "uniform",
        ffn_decay_mode="product" if recipe == "calibrated_narrow" else "parameter",
        ffn_width_init_mode="fan_in" if recipe == "calibrated_narrow" else "none",
        ffn_width_lr_mode="fan_in" if recipe == "calibrated_narrow" else "none",
        recompute_gate=recipe in ("blockshuffle", "additive"),
        gate_recompute_method="native" if recipe in ("blockshuffle", "additive") else "checkpoint",
    )
    config.validate()
    tc.validate()
    return config, tc


def synthetic_tokens(seed):
    generator = torch.Generator().manual_seed(seed + 60000)
    return torch.randint(4096, (20, 16, 129), generator=generator)


def tensor_sha(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def preflight():
    assert sha256(PLAN) == PLAN_SHA
    assert not (ROOT / "protocol.json").exists()
    local = json.loads((ROOT / "local_tests.json").read_text())
    assert local["status"] == "PASS" and local["source_unchanged"]
    from results.verification.cleanup_signatures_v1 import signatures

    retained = signatures()
    assert retained == json.loads(
        Path("results/verification/cleanup_before_signatures_v1.json").read_text()
    )
    write_json(ROOT / "retained_signatures.json", retained)
    cells = {}
    for seed in SEEDS:
        for recipe in RECIPES:
            config, tc = configurations(recipe, seed)
            cells[f"{recipe}_s{seed}"] = {"model": asdict(config), "training": asdict(tc)}
    source = provenance()
    with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in source["source_files"]:
            archive.write(name, name)
        archive.write(PLAN, PLAN.as_posix())
    protocol = {
        "plan_sha256": PLAN_SHA,
        "configurations": cells,
        "source": source,
        "source_archive_sha256": sha256(ROOT / "source.zip"),
        "environment": environment(),
        "synthetic_tokens_sha256": {
            str(seed): tensor_sha(synthetic_tokens(seed)) for seed in SEEDS
        },
        "retained_signature_cases_exact": len(retained),
        "corpus_data_used": False,
    }
    write_json(ROOT / "protocol.json", protocol)
    print(
        json.dumps(
            {"status": "PASS", "cells": len(cells), "retained_signatures_exact": len(retained)}
        ),
        flush=True,
    )


def frozen_protocol():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert sha256(PLAN) == protocol["plan_sha256"] == PLAN_SHA
    for name, expected in protocol["source"]["source_files"].items():
        assert sha256(Path(name)) == expected, name
    return protocol


def worker(recipe, seed):
    protocol = frozen_protocol()
    key = f"{recipe}_s{seed}"
    output = ROOT / "workers" / key
    output.mkdir(parents=True, exist_ok=False)
    config, tc = configurations(recipe, seed)
    assert protocol["configurations"][key] == {"model": asdict(config), "training": asdict(tc)}
    write_json(
        output / "config.json",
        {
            "model": asdict(config),
            "training": asdict(tc),
            "protocol_sha256": sha256(ROOT / "protocol.json"),
        },
    )
    history = []
    try:
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.manual_seed_all(seed)
        tokens = synthetic_tokens(seed)
        assert tensor_sha(tokens) == protocol["synthetic_tokens_sha256"][str(seed)]
        tokens = tokens.cuda()
        model = Transformer(config, seed)
        initialize_dense_width(model, tc)
        initial_common = {n: tensor_sha(p) for n, p in model.named_parameters() if ".ffn." not in n}
        model = model.cuda().train()
        configure_gate_recomputation(model, tc)
        groups = parameter_groups(model, tc)
        optimizer = torch.optim.AdamW(groups, lr=tc.learning_rate, betas=(0.9, 0.95), eps=1e-8)
        assert sum(p.numel() for p in model.parameters()) == config.total_parameters
        synchronize("cuda")
        torch.cuda.reset_peak_memory_stats()
        times = []
        for step in range(20):
            x, y = tokens[step, :, :-1], tokens[step, :, 1:]
            optimizer.zero_grad(set_to_none=True)
            rate = learning_rate(step, tc)
            for group in optimizer.param_groups:
                group["lr"] = rate * group["lr_scale"]
            synchronize("cuda")
            start = time.perf_counter()
            with autocast("cuda", "bf16"):
                loss = model.loss(x, y)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            synchronize("cuda")
            seconds = time.perf_counter() - start
            if step >= 10:
                times.append(seconds)
            assert all(torch.isfinite(p).all() for p in model.parameters())
            record = {
                "step": step + 1,
                "synthetic_loss": loss.item(),
                "preclip_gradient_norm": norm.item(),
                "learning_rate": rate,
                "synchronized_seconds": seconds,
            }
            history.append(record)
            with (output / "history.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        peak = torch.cuda.max_memory_allocated()
        finite_moments = all(
            torch.isfinite(value).all()
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        )
        assert finite_moments
        result = {
            "status": "PASS",
            "recipe": recipe,
            "seed": seed,
            "model": asdict(config),
            "training": asdict(tc),
            "environment": environment(),
            "initial_common_hashes": initial_common,
            "synthetic_tokens_sha256": protocol["synthetic_tokens_sha256"][str(seed)],
            "optimizer_groups": group_summary(groups),
            "ffn_parameters": config.unique_ffn_parameters,
            "total_parameters": config.total_parameters,
            "peak_allocated_bytes": peak,
            "median_training_step_seconds_after_warmup": statistics.median(times),
            "training_tokens_per_second_after_warmup": 16 * 128 * len(times) / sum(times),
            "timed_updates": len(times),
            "clipped_step_fraction": sum(row["preclip_gradient_norm"] > 1 for row in history) / 20,
            "all_weights_and_moments_finite": True,
            "synthetic_training_targets": 20 * 16 * 128,
            "corpus_validation_or_test_targets": 0,
            "matrix_work": forward_flops(config),
            "limitation": "Twenty synthetic next-token updates on one GPU; resource and finite-update qualification only, no language-quality ranking or convergence claim.",
        }
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": asdict(config),
                "training_config": asdict(tc),
                "step": 20,
                "synthetic_tokens_sha256": result["synthetic_tokens_sha256"],
            },
            output / "checkpoint.pt",
        )
        result["checkpoint_sha256"] = sha256(output / "checkpoint.pt")
        write_json(output / "result.json", result)
        print(
            json.dumps(
                {
                    "cell": key,
                    "status": "PASS",
                    "peak_mib": peak / 2**20,
                    "training_step_ms": 1000 * statistics.median(times),
                }
            ),
            flush=True,
        )
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "completed_updates": len(history),
            },
        )
        raise


def finish():
    protocol = frozen_protocol()
    assert not (ROOT / "result.json").exists()
    rows = {}
    for key in protocol["configurations"]:
        path = ROOT / "workers" / key
        result = json.loads((path / "result.json").read_text())
        assert result["status"] == "PASS" and not (path / "failure.json").exists()
        assert sha256(path / "checkpoint.pt") == result["checkpoint_sha256"]
        assert len((path / "history.jsonl").read_text().splitlines()) == 20
        rows[key] = result
    seeds = {}
    for seed in SEEDS:
        controls = {name: rows[f"{name}_s{seed}"] for name in RECIPES}
        common = controls["full_swiglu"]["initial_common_hashes"]
        assert all(row["initial_common_hashes"] == common for row in controls.values())
        assert len({row["synthetic_tokens_sha256"] for row in controls.values()}) == 1
        candidate = controls["additive"]
        ratios = {
            name: candidate["peak_allocated_bytes"] / row["peak_allocated_bytes"]
            for name, row in controls.items()
            if name != "additive"
        }
        seeds[str(seed)] = {
            "memory_ratios": ratios,
            "full_control_memory_pass": all(
                ratios[name] <= 1.1 for name in ("full_gelu", "full_swiglu")
            ),
        }
    memory_pass = all(row["full_control_memory_pass"] for row in seeds.values())
    result = {
        "status": "PASS" if memory_pass else "FAIL",
        "plan_sha256": PLAN_SHA,
        "source_archive_sha256": protocol["source_archive_sha256"],
        "workers": rows,
        "seed_gates": seeds,
        "finite_updates": True,
        "retained_signatures_exact": protocol["retained_signature_cases_exact"],
        "synthetic_updates": 300,
        "synthetic_training_targets": 614400,
        "corpus_training_or_validation_targets": 0,
        "quality_screen_earned": memory_pass,
        "research_target_achieved": False,
        "interpretation": "Local numerical and synthetic resource qualification; independent language-quality evidence is still required.",
    }
    write_json(ROOT / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "workers"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "worker", "finish"))
    parser.add_argument("--recipe", choices=RECIPES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.phase == "worker":
        if args.recipe is None or args.seed is None:
            parser.error("worker requires --recipe and --seed")
        worker(args.recipe, args.seed)
    elif args.phase == "preflight":
        preflight()
    else:
        finish()
