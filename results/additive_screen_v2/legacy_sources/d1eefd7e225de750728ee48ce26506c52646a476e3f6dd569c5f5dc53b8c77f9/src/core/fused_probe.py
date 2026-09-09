"""Cheap trained-checkpoint correctness and isolated-FFN fusion timing."""

import argparse
import copy
import json
import statistics
import time
import traceback
import zipfile
from pathlib import Path

import torch

from src.blockshuffle_ffn.triton_inference import matrix_work, replace_ffns
from src.core.benchmark import autocast, evaluate_forward, synchronize
from src.core.compile_serving import compile_model, errors, latency, load_model, setup
from src.core.data import TokenData
from src.core.reproducibility import environment, provenance, sha256, write_json


@torch.no_grad()
def output_error(expected, actual):
    delta = actual.float() - expected.float()
    rms = delta.square().mean().sqrt().item()
    reference_rms = expected.float().square().mean().sqrt().item()
    return {
        "rms": rms,
        "reference_rms": reference_rms,
        "relative_rms": rms / max(reference_rms, 1e-30),
        "max_abs": delta.abs().max().item(),
        "finite": bool(torch.isfinite(actual).all()),
    }


@torch.no_grad()
def probe(run: Path, control: Path, output: Path, token_tile: int = 32) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    setup()
    prov = provenance()
    protocol = {
        "run": run.name,
        "control": control.name,
        "checkpoint_sha256": sha256(run / "checkpoint.pt"),
        "control_checkpoint_sha256": sha256(control / "checkpoint.pt"),
        "environment": environment(),
        "provenance": prov,
        "token_tile": token_tile,
        "plan_sha256": sha256(Path("research/fused_execution_plan.md")),
        "interpretation": "Numerical validation and isolated-FFN timing, not full-model serving performance.",
    }
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in prov["source_files"]:
            archive.write(name, name)
        archive.write("research/fused_execution_plan.md", "research/fused_execution_plan.md")
    write_json(output / "protocol.json", protocol)
    results = {"protocol": protocol}
    try:
        native = load_model(run)
        fused = copy.deepcopy(native)
        replace_ffns(fused, token_tile)
        assert sum(p.numel() for p in fused.parameters()) == sum(
            p.numel() for p in native.parameters()
        )
        assert all(
            torch.equal(a, b) for a, b in zip(native.parameters(), fused.parameters(), strict=True)
        )
        data = TokenData(Path("data/tinystories_v1"), "cuda", 10017)
        batches = list(data.validation(16, native.config.context, 16))
        results["data_hashes"] = data.manifest["files"]
        results["matrix_work"] = matrix_work(
            native.config.width, native.config.ffn_width, native.config.groups, 2048, token_tile
        )
        captured = {i: [] for i in range(native.config.layers)}
        handles = []

        def hook(index):
            def capture(module, inputs):
                captured[index].append(inputs[0].detach())

            return capture

        for i, block in enumerate(native.blocks):
            handles.append(block.ffn.register_forward_pre_hook(hook(i)))
        for x, y in batches[:3]:
            with autocast("cuda", "bf16"):
                native(x)
        for h in handles:
            h.remove()
        checks = []
        for i, block in enumerate(native.blocks):
            for x in captured[i]:
                with autocast("cuda", "bf16"):
                    row = output_error(block.ffn(x), fused.blocks[i].ffn(x))
                row["layer"] = i
                checks.append(row)
        results["trained_ffn_errors"] = checks
        results["trained_ffn_numerical_pass"] = all(
            r["finite"] and r["relative_rms"] <= 0.005 and r["max_abs"] <= 0.05 for r in checks
        )
        write_json(output / "trained_ffn_checks.json", results)
        if not results["trained_ffn_numerical_pass"]:
            raise RuntimeError("Trained eager FFN numerical checks failed")
        print("Trained FFN checks passed; evaluating whole-model eager NLL", flush=True)
        results["native_eager_nll"], tokens = evaluate_forward(native, batches, "cuda", "bf16")
        results["fused_eager_nll"], count = evaluate_forward(fused, batches, "cuda", "bf16")
        assert count == tokens == 32768
        results["validation_tokens"] = tokens
        results["fresh_eager_logit_errors"] = errors(native, fused, [x for x, y in batches[:3]])
        results["eager_numerical_pass"] = abs(
            results["native_eager_nll"] - results["fused_eager_nll"]
        ) <= 0.001 and all(
            r["finite"] and r["rms"] <= 0.01 and r["max_abs"] <= 0.15
            for r in results["fresh_eager_logit_errors"]
        )
        write_json(output / "eager_checks.json", results)
        if not results["eager_numerical_pass"]:
            raise RuntimeError("Whole-model eager numerical checks failed")
        print(
            f"Eager NLL native={results['native_eager_nll']:.9f}, fused={results['fused_eager_nll']:.9f}; starting isolated FFN compilation",
            flush=True,
        )
        full = load_model(control)
        eager = {
            "full_swiglu": full.blocks[0].ffn,
            "native_blockshuffle": native.blocks[0].ffn,
            "fused_blockshuffle": fused.blocks[0].ffn,
        }
        forwards = {}
        x = captured[0][0]
        results["micro_compile_seconds"] = {}
        results["micro_compiled_errors"] = {}
        for name, model in eager.items():
            start = time.perf_counter()
            forwards[name] = compile_model(model)
            with autocast("cuda", "bf16"):
                forwards[name](x)
            synchronize("cuda")
            results["micro_compile_seconds"][name] = time.perf_counter() - start
            rows = []
            for xx in captured[0]:
                with autocast("cuda", "bf16"):
                    rows.append(output_error(model(xx), forwards[name](xx)))
            results["micro_compiled_errors"][name] = rows
            print(f"Compiled {name}", flush=True)
        results["micro_compiled_numerical_pass"] = all(
            r["finite"] and r["relative_rms"] <= 0.005 and r["max_abs"] <= 0.05
            for rows in results["micro_compiled_errors"].values()
            for r in rows
        )
        if not results["micro_compiled_numerical_pass"]:
            raise RuntimeError("Compiled FFN numerical checks failed")
        names = list(forwards)
        rounds = []
        for iteration in range(6):
            order = names[iteration % 3 :] + names[: iteration % 3]
            times = {name: latency(forwards[name], x, warmup=4, repeats=30) for name in order}
            rounds.append({"order": order, "seconds_per_forward": times})
        samples = [
            r["seconds_per_forward"]["native_blockshuffle"]
            / r["seconds_per_forward"]["fused_blockshuffle"]
            for r in rounds
        ]
        results["micro_rounds"] = rounds
        results["micro_fused_over_native_speedup"] = {
            "samples": samples,
            "median": statistics.median(samples),
            "min": min(samples),
            "max": max(samples),
        }
        results["earned_full_model_audit"] = statistics.median(samples) >= 1.1
        write_json(output / "result.json", results)
        print(
            json.dumps(
                {
                    k: results[k]
                    for k in (
                        "native_eager_nll",
                        "fused_eager_nll",
                        "micro_fused_over_native_speedup",
                        "earned_full_model_audit",
                    )
                },
                indent=2,
            ),
            flush=True,
        )
        return results
    except Exception as exc:
        write_json(
            output / "failure.json",
            {"error": repr(exc), "traceback": traceback.format_exc(), "partial_results": results},
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-tile", type=int, choices=(16, 32), default=32)
    args = parser.parse_args()
    probe(args.run, args.control, args.output, args.token_tile)
