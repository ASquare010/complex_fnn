"""Isolated training-memory audit for native and compiled rational gate products."""

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import torch

from src.core.activation_screen import CACHE, controls, set_execution
from src.core.benchmark import autocast, evaluate, synchronize
from src.core.compile_serving import setup
from src.core.config import ModelConfig, TrainConfig
from src.core.data import TokenData
from src.core.optimization import parameter_groups
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.transformer import Transformer
from src.learnable_activation_ffn.compiled_product import install_compiled_products

CASES = ("full_gelu", "full_swiglu", "rational_native", "rational_compiled")
PLAN = Path("research/activation_execution_plan.md")


def selected_runs():
    refs = controls()
    selected = json.loads(Path("results/activation_screen_v1/result.json").read_text())
    return {
        "full_gelu": refs["full_gelu"]["run"],
        "full_swiglu": refs["full_swiglu"]["run"],
        "rational_native": selected["selected"]["blockshuffle_rational"],
        "rational_compiled": selected["selected"]["blockshuffle_rational"],
    }


def grad_snapshot(model, x, y):
    model.zero_grad(set_to_none=True)
    with autocast("cuda", "bf16"):
        loss = model.loss(x, y)
    loss.backward()
    gradients = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()}
    assert all(torch.isfinite(p).all() for p in gradients.values())
    model.zero_grad(set_to_none=True)
    return gradients


def gradient_error(actual, expected, shape_only=False):
    names = [n for n in expected if not shape_only or ".curve." in n]
    delta = sum((actual[n].double() - expected[n].double()).square().sum().item() for n in names)
    scale = sum(expected[n].double().square().sum().item() for n in names)
    return math.sqrt(delta / max(scale, 1e-30))


@torch.no_grad()
def logits(model, x):
    with autocast("cuda", "bf16"):
        return model(x).detach().cpu().float()


def worker(case, output):
    setup()
    torch._dynamo.config.recompile_limit = 64
    output.mkdir(parents=True, exist_ok=False)
    correction_only = case == "rational_correction_compiled"
    plan = Path("research/activation_correction_execution_plan.md") if correction_only else PLAN
    source_case = "rational_compiled" if correction_only else case
    path = Path("results/runs") / selected_runs()[source_case]
    checkpoint_hash = sha256(path / "checkpoint.pt")
    m = json.loads((path / "metrics.json").read_text())
    record = {
        "case": case,
        "run": path.name,
        "environment": environment(),
        "provenance": provenance(),
        "checkpoint_sha256": checkpoint_hash,
        "plan_sha256": sha256(plan),
    }
    write_json(output / "protocol.json", record)
    try:
        checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
        mc, tc = ModelConfig(**m["model"]), TrainConfig(**m["training"])
        model = Transformer(mc, tc.seed).cuda()
        model.load_state_dict(checkpoint["model"], strict=True)
        set_execution(model, tc)
        groups = parameter_groups(model, tc)
        optimizer = torch.optim.AdamW(groups, lr=tc.learning_rate, betas=(0.9, 0.95), eps=1e-8)
        optimizer.load_state_dict(checkpoint["optimizer"])
        data = TokenData(CACHE, "cuda", 9017)
        native_nll, targets = evaluate(model, data.validation(16, 128, 158), "cuda", "bf16")
        assert targets == 322688 and abs(native_nll - m["validation_loss"]) < 1e-7
        record["native_nll"] = native_nll
        record["validation_targets"] = targets
        record["data_hashes"] = data.manifest["files"]
        x, y = data.batch(16, 128)
        identities = {n: id(p) for n, p in model.named_parameters()}
        if case in ("rational_compiled", "rational_correction_compiled"):
            expected_logits = logits(model, x)
            expected_gradients = grad_snapshot(model, x, y)
            start = time.perf_counter()
            if correction_only:
                from src.learnable_activation_ffn.compiled_correction import (
                    install_compiled_corrections,
                )

                install_compiled_corrections(model)
            else:
                install_compiled_products(model)
            actual_logits = logits(model, x)
            actual_gradients = grad_snapshot(model, x, y)
            synchronize("cuda")
            record["first_compile_and_gradient_check_seconds"] = time.perf_counter() - start
            delta = actual_logits - expected_logits
            fidelity = {
                "logits_max_abs": delta.abs().max().item(),
                "logits_relative_l2": (
                    delta.double().norm() / expected_logits.double().norm()
                ).item(),
                "all_gradients_relative_l2": gradient_error(actual_gradients, expected_gradients),
                "activation_gradients_relative_l2": gradient_error(
                    actual_gradients, expected_gradients, True
                ),
                "finite": bool(torch.isfinite(actual_logits).all()),
            }
            record["fidelity"] = fidelity
            write_json(output / "numerical_preflight.json", record)
            assert fidelity["finite"] and fidelity["logits_max_abs"] <= 0.05
            assert fidelity["logits_relative_l2"] <= 0.003
            assert fidelity["all_gradients_relative_l2"] <= (0.0001 if correction_only else 0.01)
            assert fidelity["activation_gradients_relative_l2"] <= 0.02
            compiled_nll, count = evaluate(model, data.validation(16, 128, 158), "cuda", "bf16")
            assert count == targets and abs(compiled_nll / native_nll - 1) <= 0.0001
            record["compiled_nll"] = compiled_nll
            record["compiled_relative_nll_percent"] = 100 * (compiled_nll / native_nll - 1)
            del expected_logits, expected_gradients, actual_logits, actual_gradients, delta
        assert {n: id(p) for n, p in model.named_parameters()} == identities
        record["original_parameter_objects_preserved"] = True
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        del checkpoint, x, y
        model.train()
        data.generator.manual_seed(19017)
        digests = []
        elapsed = []
        losses = []
        norms = []
        for step in range(30):
            if step == 10:
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                synchronize("cuda")
                torch.cuda.reset_peak_memory_stats()
                record["resident_before_measured_steps"] = torch.cuda.memory_allocated()
            synchronize("cuda")
            start = time.perf_counter()
            x, y = data.batch(16, 128)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", "bf16"):
                loss = model.loss(x, y)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            synchronize("cuda")
            duration = time.perf_counter() - start
            assert torch.isfinite(loss)
            losses.append(loss.item())
            norms.append(norm.item())
            digests.append(
                hashlib.sha256(x.cpu().numpy().tobytes() + y.cpu().numpy().tobytes()).hexdigest()
            )
            if step >= 10:
                elapsed.append(duration)
        record.update(
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(),
            resident_after_measured_steps=torch.cuda.memory_allocated(),
            warmup_steps=10,
            measured_steps=20,
            training_tokens_per_second=20 * 16 * 128 / sum(elapsed),
            step_seconds=elapsed,
            transient_losses=losses,
            gradient_norms_pre_clip=norms,
            minibatch_hashes=digests,
            checkpoint_unchanged=sha256(path / "checkpoint.pt") == checkpoint_hash,
            total_parameters=sum(p.numel() for p in model.parameters()),
            status="complete",
            note="Transient profiling updates are not saved or scored as new training evidence.",
        )
        assert record["checkpoint_unchanged"]
        write_json(output / "result.json", record)
        print(
            json.dumps(
                {
                    k: record[k]
                    for k in (
                        "case",
                        "status",
                        "peak_allocated_vram_bytes",
                        "training_tokens_per_second",
                    )
                }
            ),
            flush=True,
        )
    except Exception as exc:
        write_json(
            output / "failure.json",
            {**record, "error": repr(exc), "traceback": traceback.format_exc()},
        )
        raise


def audit(output):
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "cases": CASES,
        "selected_runs": selected_runs(),
        "provenance": provenance(),
        "plan_sha256": sha256(PLAN),
        "worker_timeout_seconds": 900,
        "fidelity_sampling_seed": 9017,
        "profiling_sampling_seed": 19017,
    }
    write_json(output / "protocol.json", protocol)
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in protocol["provenance"]["source_files"]:
            archive.write(name, name)
        archive.write(PLAN, PLAN.as_posix())
    records = {}
    try:
        for case in CASES:
            with (output / f"{case}.log").open("x", encoding="utf-8") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-X",
                        "faulthandler",
                        "-m",
                        "src.core.activation_execution",
                        "worker",
                        "--case",
                        case,
                        "--output",
                        str(output / case),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=900,
                )
            r = json.loads((output / case / "result.json").read_text())
            records[case] = r
            write_json(output / "progress.json", {"status": "running", "completed": list(records)})
            print(
                json.dumps({"case": case, "peak_mib": r["peak_allocated_vram_bytes"] / 2**20}),
                flush=True,
            )
        assert all(
            r["minibatch_hashes"] == records[CASES[0]]["minibatch_hashes"] for r in records.values()
        )
        candidate = records["rational_compiled"]
        gates = {
            f"memory_within_ten_percent_{ref}": candidate["peak_allocated_vram_bytes"]
            <= 1.1 * records[ref]["peak_allocated_vram_bytes"]
            for ref in ("full_gelu", "full_swiglu")
        }
        result = {
            "status": "complete",
            "same_minibatches_all_cases": True,
            "gates": gates,
            "execution_passes": all(gates.values()),
            "compiled_vs_native_memory_ratio": candidate["peak_allocated_vram_bytes"]
            / records["rational_native"]["peak_allocated_vram_bytes"],
            "result_hashes": {case: sha256(output / case / "result.json") for case in CASES},
        }
        write_json(output / "result.json", result)
        write_json(output / "progress.json", {"status": "complete", "completed": list(records)})
        print(json.dumps(result), flush=True)
    except Exception as exc:
        write_json(
            output / "failure.json",
            {"completed": list(records), "error": repr(exc), "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "audit"))
    parser.add_argument("--case", choices=CASES + ("rational_correction_compiled",))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "worker":
        if not args.case:
            parser.error("worker requires case")
        worker(args.case, args.output)
    else:
        audit(args.output)
