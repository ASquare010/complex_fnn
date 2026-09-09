"""Bounded post-training compiler audit with isolated memory workers."""

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from src.core.benchmark import autocast, evaluate_forward, forward_flops, synchronize
from src.core.config import ModelConfig
from src.core.data import TokenData
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.transformer import Transformer


def setup() -> None:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(Path(".cache/torchinductor").resolve())
    os.environ["TRITON_CACHE_DIR"] = str(Path(".cache/triton").resolve())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.config.suppress_errors = False


def load_model(run: Path):
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["model_config"])
    model = Transformer(config, checkpoint["training_config"]["seed"])
    model.load_state_dict(checkpoint["model"])
    return model.cuda().eval()


def compile_model(model):
    return torch.compile(model, backend="inductor", mode="default", fullgraph=True, dynamic=False)


@torch.no_grad()
def errors(eager, compiled, inputs) -> list[dict]:
    rows = []
    for x in inputs:
        with autocast("cuda", "bf16"):
            reference = eager(x).float()
            actual = compiled(x).float()
        delta = actual - reference
        rows.append(
            {
                "max_abs": delta.abs().max().item(),
                "rms": delta.square().mean().sqrt().item(),
                "finite": bool(torch.isfinite(actual).all()),
            }
        )
    return rows


@torch.no_grad()
def latency(forward, x, warmup=4, repeats=15) -> float:
    for _ in range(warmup):
        with autocast("cuda", "bf16"):
            forward(x)
    synchronize("cuda")
    start = time.perf_counter()
    for _ in range(repeats):
        with autocast("cuda", "bf16"):
            forward(x)
    synchronize("cuda")
    return (time.perf_counter() - start) / repeats


@torch.no_grad()
def worker(run: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Do not overwrite compiler worker evidence")
    print("Worker imports complete; starting setup", flush=True)
    setup()
    print("Worker setup complete", flush=True)
    import triton

    record = {
        "run": run.name,
        "checkpoint_sha256": sha256(run / "checkpoint.pt"),
        "environment": environment(),
        "triton_version": triton.__version__,
        "provenance": provenance(),
        "compiler": {"backend": "inductor", "mode": "default", "fullgraph": True, "dynamic": False},
        "memory_protocol": "Fresh process for this model; eager and compiled callables share the same learned tensors. Reset peak after compilation and numerical checks; measure compiled fixed validation and forwards, with no optimizer.",
    }
    try:
        print("Loading checkpoint", flush=True)
        model = load_model(run)
        data = TokenData(Path("data/tinystories_v1"), "cuda", 10017)
        batches = list(data.validation(16, model.config.context, 16))
        inputs = [x for x, y in batches[:3]]
        print("Starting eager validation", flush=True)
        eager_nll, tokens = evaluate_forward(model, batches, "cuda", "bf16")
        synchronize("cuda")
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        print("Starting model compile", flush=True)
        compiled = compile_model(model)
        with autocast("cuda", "bf16"):
            compiled(inputs[0])
        synchronize("cuda")
        record["first_compile_and_forward_seconds"] = time.perf_counter() - start
        record["compile_peak_allocated_vram_bytes"] = torch.cuda.max_memory_allocated()
        print("Compilation finished; checking fresh inputs", flush=True)
        record["fresh_input_errors_before_warmup"] = errors(model, compiled, inputs)
        latency(compiled, inputs[0])
        record["fresh_input_errors_after_warmup"] = errors(model, compiled, inputs)
        compiled_nll, count = evaluate_forward(compiled, batches, "cuda", "bf16")
        assert count == tokens == 32768
        all_errors = (
            record["fresh_input_errors_before_warmup"] + record["fresh_input_errors_after_warmup"]
        )
        passed = abs(compiled_nll - eager_nll) <= 0.001 and all(
            r["finite"] and r["rms"] <= 0.01 and r["max_abs"] <= 0.15 for r in all_errors
        )
        record.update(
            eager_nll=eager_nll,
            compiled_nll=compiled_nll,
            validation_tokens=count,
            numerical_checks_pass=passed,
            **forward_flops(model.config),
        )
        gc.collect()
        torch.cuda.empty_cache()
        synchronize("cuda")
        torch.cuda.reset_peak_memory_stats()
        evaluate_forward(compiled, batches, "cuda", "bf16")
        record["single_process_compiled_latency_seconds"] = latency(compiled, inputs[0])
        record["peak_inference_allocated_vram_bytes"] = torch.cuda.max_memory_allocated()
        record["learned_parameter_bytes"] = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        record["status"] = "PASS" if passed else "NUMERICAL_REJECTION"
        write_json(output, record)
        return record
    except Exception as exc:
        record.update(status="FAILED", error=repr(exc), traceback=traceback.format_exc())
        write_json(output, record)
        raise


@torch.no_grad()
def audit(run: Path, control: Path, output: Path) -> dict:
    setup()
    output.mkdir(parents=True, exist_ok=False)
    sources = {"full_reference": control, "candidate": run}
    metrics = {k: json.loads((p / "metrics.json").read_text()) for k, p in sources.items()}
    assert metrics["candidate"]["model"]["variant"] == "blockshuffle_swiglu"
    assert metrics["full_reference"]["model"]["variant"] == "swiglu"
    for field in ("width", "layers", "heads", "context", "vocab_size"):
        assert metrics["candidate"]["model"][field] == metrics["full_reference"]["model"][field]
    assert metrics["candidate"]["data"]["files"] == metrics["full_reference"]["data"]["files"]
    assert all(
        m["training"]["steps"] == 800 and m["training"]["seed"] == 17 for m in metrics.values()
    )
    protocol = {
        "sources": {k: str(p) for k, p in sources.items()},
        "environment": environment(),
        "provenance": provenance(),
        "rounds": 6,
        "warmup_forwards_per_case_per_round": 4,
        "timed_forwards_per_case_per_round": 15,
        "worker_timeout_seconds": 300,
        "mode": "default; fullgraph; static shape; no eager fallback",
        "cache_directories": {
            "inductor": os.environ["TORCHINDUCTOR_CACHE_DIR"],
            "triton": os.environ["TRITON_CACHE_DIR"],
        },
        "memory_scope": "Isolated fresh worker per model. Co-resident parent process is used only for paired timing, not memory comparisons.",
    }
    write_json(output / "protocol.json", protocol)
    import zipfile

    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in protocol["provenance"]["source_files"]:
            archive.write(name, name)
    workers = {}
    try:
        for label, path in sources.items():
            print(f"Starting isolated compiler worker: {label}", flush=True)
            with (output / f"{label}_worker.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-X",
                        "faulthandler",
                        "-m",
                        "src.core.compile_serving",
                        "worker",
                        str(path),
                        "--output",
                        str(output / f"{label}.json"),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=300,
                )
            workers[label] = json.loads((output / f"{label}.json").read_text())
            if not workers[label]["numerical_checks_pass"]:
                raise RuntimeError(f"Numerical checks failed for {label}")
            print(
                json.dumps(
                    {
                        "worker": label,
                        "status": workers[label]["status"],
                        "compile_seconds": workers[label]["first_compile_and_forward_seconds"],
                        "nll": workers[label]["compiled_nll"],
                    }
                ),
                flush=True,
            )
        data = TokenData(Path("data/tinystories_v1"), "cuda", 10017)
        context = metrics["candidate"]["model"]["context"]
        batches = list(data.validation(16, context, 16))
        inputs = [x for x, y in batches[:3]]
        cases = {}
        paired_nll = {}
        for label, path in sources.items():
            model = load_model(path)
            compiled = compile_model(model)
            check = errors(model, compiled, inputs)
            if not all(r["finite"] and r["rms"] <= 0.01 and r["max_abs"] <= 0.15 for r in check):
                raise RuntimeError(f"Fresh paired-process inputs failed for {label}: {check}")
            cases[f"{label}_eager"] = model
            cases[f"{label}_compiled"] = compiled
            paired_nll[label], tokens = evaluate_forward(compiled, batches, "cuda", "bf16")
            assert tokens == 32768 and abs(paired_nll[label] - workers[label]["eager_nll"]) <= 0.001
        order = list(cases)
        rounds = []
        for index in range(6):
            rotation = order[index % 4 :] + order[: index % 4]
            times = {label: latency(cases[label], inputs[0]) for label in rotation}
            rounds.append({"round": index, "order": rotation, "seconds_per_forward": times})
            print(json.dumps(rounds[-1]), flush=True)
        samples = [
            r["seconds_per_forward"]["full_reference_compiled"]
            / r["seconds_per_forward"]["candidate_compiled"]
            for r in rounds
        ]
        speedups = {
            label: [
                r["seconds_per_forward"][f"{label}_eager"]
                / r["seconds_per_forward"][f"{label}_compiled"]
                for r in rounds
            ]
            for label in sources
        }
        after = {
            label: errors(cases[f"{label}_eager"], cases[f"{label}_compiled"], inputs)
            for label in sources
        }
        assert all(
            r["finite"] and r["rms"] <= 0.01 and r["max_abs"] <= 0.15
            for checks in after.values()
            for r in checks
        )
        result = {
            "protocol": protocol,
            "workers": workers,
            "paired_compiled_nll": paired_nll,
            "paired_rounds": rounds,
            "compiled_candidate_throughput_ratio": {
                "samples": samples,
                "median": statistics.median(samples),
                "min": min(samples),
                "max": max(samples),
            },
            "compiled_over_eager_speedup": {
                k: {"samples": v, "median": statistics.median(v)} for k, v in speedups.items()
            },
            "fresh_input_errors_after_timing": after,
            "retention_gates": {
                "numerical_checks": True,
                "within_1_percent_compiled_full_nll": paired_nll["candidate"]
                <= 1.01 * paired_nll["full_reference"],
                "median_relative_speed_at_least_point_eight": statistics.median(samples) >= 0.8,
            },
            "interpretation": "One fixed-shape post-training compiler mode on one GPU. Actual compiled NLL is reported. No architecture novelty, training-speed, autoregressive or bitwise trajectory claim.",
        }
        write_json(output / "result.json", result)
        return result
    except Exception as exc:
        write_json(
            output / "failure.json",
            {
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "completed_workers": list(workers),
                "protocol": protocol,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "audit"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "worker":
        result = worker(args.run, args.output)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ("status", "eager_nll", "compiled_nll", "numerical_checks_pass")
                }
            ),
            flush=True,
        )
    else:
        if args.control is None:
            parser.error("audit requires --control")
        result = audit(args.run, args.control, args.output)
        print(json.dumps(result["retention_gates"], indent=2))
