"""Full-model fused serving with isolated memory and three equal compiler controls."""

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import torch

from src.blockshuffle_ffn.triton_inference import matrix_work, replace_ffns
from src.core.benchmark import autocast, evaluate_forward, forward_flops, synchronize
from src.core.compile_serving import compile_model, latency, load_model, setup
from src.core.data import TokenData
from src.core.reproducibility import environment, provenance, sha256, write_json


@torch.no_grad()
def reference_outputs(model, inputs):
    rows = []
    for x in inputs:
        with autocast("cuda", "bf16"):
            rows.append(model(x).detach().cpu())
    return rows


@torch.no_grad()
def cached_errors(forward, inputs, references):
    rows = []
    for x, expected in zip(inputs, references, strict=True):
        with autocast("cuda", "bf16"):
            actual = forward(x).float()
        delta = actual - expected.to(actual.device).float()
        rows.append(
            {
                "max_abs": delta.abs().max().item(),
                "rms": delta.square().mean().sqrt().item(),
                "finite": bool(torch.isfinite(actual).all()),
            }
        )
    return rows


def errors_pass(rows):
    return all(r["finite"] and r["max_abs"] <= 0.15 and r["rms"] <= 0.01 for r in rows)


@torch.no_grad()
def worker(run: Path, output: Path, fused: bool = False):
    if output.exists():
        raise ValueError("Do not overwrite serving worker evidence")
    setup()
    record = {
        "run": run.name,
        "execution": "four_kernel" if fused else "native",
        "environment": environment(),
        "provenance": provenance(),
        "checkpoint_sha256": sha256(run / "checkpoint.pt"),
        "compiler": {"backend": "inductor", "mode": "default", "fullgraph": True, "dynamic": False},
        "memory_protocol": "One set of learned GPU tensors, no optimizer. Native eager reference logits are kept on CPU. Peak reset after compilation/correctness; compiled validation and forwards measured. Parent co-resident timing does not supply memory values.",
    }
    try:
        print("Loading native checkpoint", flush=True)
        model = load_model(run)
        data = TokenData(Path("data/tinystories_v1"), "cuda", 10017)
        batches = list(data.validation(16, model.config.context, 16))
        inputs = [x for x, y in batches[:3]]
        record["data_hashes"] = data.manifest["files"]
        record["native_eager_nll"], tokens = evaluate_forward(model, batches, "cuda", "bf16")
        references = reference_outputs(model, inputs)
        record["cpu_reference_bytes"] = sum(r.numel() * r.element_size() for r in references)
        original_ids = [id(p) for p in model.parameters()]
        if fused:
            replace_ffns(model)
        record["parameter_objects_unchanged"] = [id(p) for p in model.parameters()] == original_ids
        assert record["parameter_objects_unchanged"]
        record["target_eager_nll"], count = evaluate_forward(model, batches, "cuda", "bf16")
        record["fresh_eager_errors"] = cached_errors(model, inputs, references)
        assert count == tokens == 32768
        if abs(record["target_eager_nll"] - record["native_eager_nll"]) > 0.001 or not errors_pass(
            record["fresh_eager_errors"]
        ):
            raise RuntimeError("Target eager accuracy failed before compilation")
        synchronize("cuda")
        start = time.perf_counter()
        print("Compiling full model", flush=True)
        compiled = compile_model(model)
        with autocast("cuda", "bf16"):
            compiled(inputs[0])
        synchronize("cuda")
        record["first_compile_and_forward_seconds"] = time.perf_counter() - start
        record["fresh_compiled_errors_before_warmup"] = cached_errors(compiled, inputs, references)
        latency(compiled, inputs[0])
        record["fresh_compiled_errors_after_warmup"] = cached_errors(compiled, inputs, references)
        record["compiled_nll"], count = evaluate_forward(compiled, batches, "cuda", "bf16")
        record["validation_tokens"] = count
        passed = abs(record["compiled_nll"] - record["native_eager_nll"]) <= 0.001 and errors_pass(
            record["fresh_compiled_errors_before_warmup"]
            + record["fresh_compiled_errors_after_warmup"]
        )
        record["numerical_checks_pass"] = passed
        record.update(forward_flops(model.config))
        if fused:
            record["fused_matrix_work"] = matrix_work(
                model.config.width, model.config.ffn_width, model.config.groups
            )
        gc.collect()
        torch.cuda.empty_cache()
        synchronize("cuda")
        torch.cuda.reset_peak_memory_stats()
        evaluate_forward(compiled, batches, "cuda", "bf16")
        record["isolated_compiled_latency_seconds"] = latency(compiled, inputs[0])
        record["peak_inference_allocated_vram_bytes"] = torch.cuda.max_memory_allocated()
        record["learned_parameter_bytes"] = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        record["persistent_extra_cache_bytes"] = 0
        record["status"] = "PASS" if passed else "NUMERICAL_REJECTION"
        write_json(output, record)
        print(
            json.dumps(
                {
                    k: record[k]
                    for k in ("status", "compiled_nll", "peak_inference_allocated_vram_bytes")
                }
            ),
            flush=True,
        )
        return record
    except Exception as exc:
        record.update(status="FAILED", error=repr(exc), traceback=traceback.format_exc())
        write_json(output, record)
        raise


@torch.no_grad()
def audit(run: Path, control: Path, output: Path):
    setup()
    output.mkdir(parents=True, exist_ok=False)
    sources = {"full_reference": control, "native_candidate": run, "fused_candidate": run}
    metrics = {k: json.loads((p / "metrics.json").read_text()) for k, p in sources.items()}
    for k, m in metrics.items():
        assert m["data"]["files"] == metrics["full_reference"]["data"]["files"]
        for field in ("width", "layers", "heads", "context", "vocab_size"):
            assert m["model"][field] == metrics["full_reference"]["model"][field]
        assert m["training"]["steps"] == 800 and m["training"]["seed"] == 17
    protocol = {
        "sources": {k: str(p) for k, p in sources.items()},
        "environment": environment(),
        "provenance": provenance(),
        "plan_sha256": sha256(Path("research/fused_execution_plan.md")),
        "rounds": 6,
        "warmup_forwards_per_case_per_round": 4,
        "timed_forwards_per_case_per_round": 15,
        "worker_timeout_seconds": 300,
        "mode": "default; fullgraph; static shape; no fallback",
        "cache_directories": {
            "inductor": os.environ["TORCHINDUCTOR_CACHE_DIR"],
            "triton": os.environ["TRITON_CACHE_DIR"],
        },
        "memory_scope": "Fresh worker per model; parent holds three models only for paired timing.",
    }
    write_json(output / "protocol.json", protocol)
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in protocol["provenance"]["source_files"]:
            archive.write(name, name)
        archive.write("research/fused_execution_plan.md", "research/fused_execution_plan.md")
    workers = {}
    try:
        for label, path in sources.items():
            print(f"Starting isolated worker: {label}", flush=True)
            command = [
                sys.executable,
                "-X",
                "faulthandler",
                "-m",
                "src.core.fused_serving",
                "worker",
                str(path),
                "--output",
                str(output / f"{label}.json"),
            ]
            if label == "fused_candidate":
                command.append("--fused")
            with (output / f"{label}_worker.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300
                )
            workers[label] = json.loads((output / f"{label}.json").read_text())
            if not workers[label]["numerical_checks_pass"]:
                raise RuntimeError(f"Numerical rejection for {label}")
            print(
                json.dumps(
                    {
                        "worker": label,
                        "nll": workers[label]["compiled_nll"],
                        "compile_seconds": workers[label]["first_compile_and_forward_seconds"],
                    }
                ),
                flush=True,
            )
        data = TokenData(Path("data/tinystories_v1"), "cuda", 10017)
        batches = list(data.validation(16, metrics["full_reference"]["model"]["context"], 16))
        inputs = [x for x, y in batches[:3]]
        cases, references, nll = {}, {}, {}
        for label, path in sources.items():
            model = load_model(path)
            references[label] = reference_outputs(model, inputs)
            if label == "fused_candidate":
                replace_ffns(model)
            cases[label] = compile_model(model)
            check = cached_errors(cases[label], inputs, references[label])
            if not errors_pass(check):
                raise RuntimeError(f"Paired-process correctness failed: {label}")
            nll[label], tokens = evaluate_forward(cases[label], batches, "cuda", "bf16")
            assert tokens == 32768 and abs(nll[label] - workers[label]["native_eager_nll"]) <= 0.001
        names = list(cases)
        rounds = []
        for i in range(6):
            order = names[i % 3 :] + names[: i % 3]
            times = {k: latency(cases[k], inputs[0]) for k in order}
            rounds.append({"round": i, "order": order, "seconds_per_forward": times})
            print(json.dumps(rounds[-1]), flush=True)
        ratios = {}
        for denominator in ("full_reference", "native_candidate"):
            samples = [
                r["seconds_per_forward"][denominator] / r["seconds_per_forward"]["fused_candidate"]
                for r in rounds
            ]
            ratios[denominator] = {
                "samples": samples,
                "median": statistics.median(samples),
                "min": min(samples),
                "max": max(samples),
            }
        after = {k: cached_errors(v, inputs, references[k]) for k, v in cases.items()}
        assert all(errors_pass(rows) for rows in after.values())
        result = {
            "protocol": protocol,
            "workers": workers,
            "paired_compiled_nll": nll,
            "paired_rounds": rounds,
            "fused_throughput_ratio": ratios,
            "fresh_input_errors_after_timing": after,
            "retention_gates": {
                "numerical_checks": True,
                "within_1_percent_compiled_full_nll": nll["fused_candidate"]
                <= 1.01 * nll["full_reference"],
                "median_relative_speed_at_least_point_eight": ratios["full_reference"]["median"]
                >= 0.8,
            },
            "interpretation": "One fixed-shape, post-training inference implementation on one GPU/checkpoint. No training speed, autoregressive, bitwise trajectory, architectural novelty or broader-corpus claim.",
        }
        write_json(output / "result.json", result)
        print(json.dumps(result["retention_gates"]), flush=True)
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
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.run, args.output, args.fused)
    else:
        if args.control is None:
            parser.error("audit requires --control")
        audit(args.run, args.control, args.output)
