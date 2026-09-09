"""Fixed post-training condition floor with native and fused checkpoint validation."""

import argparse
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

from src.core.benchmark import autocast, evaluate_forward, synchronize
from src.core.compile_serving import compile_model, setup
from src.core.conditioning import down_spectra, floor_square_blocks
from src.core.config import ModelConfig
from src.core.data import TokenData
from src.core.model_audit import gradient_flow
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.transformer import Transformer
from src.core.validation_tail import per_window_losses, window_batches

SEEDS = (17, 29, 43)
PLAN = Path("research/joint_conditioning_plan.md")
TAIL = Path("results/validation_tail_scale384_v1/result.json")


def compare_seed(original, floored, controls):
    """Keep weight-intervention, implementation-accuracy and baseline gates distinct."""
    gates = {
        "condition_improves_at_least_tenfold": original["worst_down_condition"]
        >= 10 * floored["worst_down_condition"],
        "original_numerical_checks": original["numerical_checks_pass"],
        "floored_numerical_checks": floored["numerical_checks_pass"],
        "original_finite_gradients": original["finite_gradients"],
        "floored_finite_gradients": floored["finite_gradients"],
    }
    for region in ("prefix", "tail"):
        base = original["native"][region + "_nll"]
        for execution in ("native", "compiled"):
            value = floored[execution][region + "_nll"]
            gates[f"{region}_{execution}_floor_cost_at_most_point_one_percent"] = (
                value <= 1.001 * base
            )
        value = floored["compiled"][region + "_nll"]
        for recipe in ("full_swiglu", "full_gelu"):
            gates[f"{region}_within_one_percent_{recipe}"] = (
                value <= 1.01 * controls[recipe][region + "_nll"]
            )
        gates[f"{region}_beats_calibrated_narrow"] = (
            value < controls["calibrated_narrow"][region + "_nll"]
        )
    return gates


@torch.no_grad()
def quality(forward, data, prefix, output, label):
    nll, count = evaluate_forward(forward, prefix, "cuda", "bf16")
    tail, targets, rows = per_window_losses(
        forward, window_batches(data.valid, 128, 16, 256, 1568), "cuda", "bf16"
    )
    assert count == 32768 and targets == 167936
    write_json(output / f"{label}_tail_windows.json", rows)
    return {
        "prefix_nll": nll,
        "prefix_targets": count,
        "tail_nll": tail,
        "tail_targets": targets,
        "aggregate_nll": (nll * count + tail * targets) / (count + targets),
    }


def worker(seed: int, floored: bool, output: Path):
    from src.blockshuffle_ffn.triton_inference import replace_ffns
    from src.core.fused_serving import cached_errors, errors_pass, reference_outputs

    setup()
    output.mkdir(parents=True, exist_ok=False)
    run = Path(f"results/runs/scale384_blockshuffle_s{seed}_800")
    source_hash = sha256(run / "checkpoint.pt")
    record = {
        "seed": seed,
        "floor_strength": 0.1 if floored else None,
        "run": run.name,
        "checkpoint_sha256": source_hash,
        "environment": environment(),
        "provenance": provenance(),
        "plan_sha256": sha256(PLAN),
        "timing_claim": False,
    }
    try:
        checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
        config = ModelConfig(**checkpoint["model_config"])
        assert (config.width, config.layers, config.hidden, config.groups) == (384, 8, 2048, 8)
        assert config.variant == "blockshuffle_swiglu" and config.context == 128
        assert (
            checkpoint["training_config"]["seed"] == seed
            and checkpoint["training_config"]["steps"] == 800
        )
        model = Transformer(config, seed).eval()
        model.load_state_dict(checkpoint["model"])
        transformed, changes = {}, {}
        if floored:
            with torch.no_grad():
                for i, block in enumerate(model.blocks):
                    key = f"blocks.{i}.ffn.down.second.weight"
                    weight = block.ffn.down.second.weight
                    before = weight.detach().clone()
                    after = floor_square_blocks(before, 0.1)
                    weight.copy_(after)
                    transformed[key] = after
                    changes[key] = {
                        "sha256": hashlib.sha256(after.numpy().tobytes()).hexdigest(),
                        "relative_frobenius_edit": (
                            (after - before).double().norm() / before.double().norm()
                        ).item(),
                        "frobenius_norm_ratio": (
                            after.double().norm() / before.double().norm()
                        ).item(),
                    }
        record["other_weights_bitwise_unchanged"] = all(
            torch.equal(v, checkpoint["model"][k])
            for k, v in model.state_dict().items()
            if k not in transformed
        )
        assert record["other_weights_bitwise_unchanged"]
        record["unique_parameters"] = sum(p.numel() for p in model.parameters())
        record["ffn_parameters"] = sum(p.numel() for b in model.blocks for p in b.ffn.parameters())
        assert record["unique_parameters"] == 9099648 and record["ffn_parameters"] == 2801664
        if transformed:
            torch.save(transformed, output / "transformed_factors.pt")
            record["transform_file_sha256"] = sha256(output / "transformed_factors.pt")
        record["changes"] = changes
        print("Computing CPU spectra and fixed loss-direction gradients", flush=True)
        record["down_spectra"] = down_spectra(model)
        record["worst_down_condition"] = max(
            r["projection"]["condition_nonzero"] for r in record["down_spectra"].values()
        )
        if floored:
            assert all(
                r["square_factor"]["condition_nonzero"] <= 10.0001
                for r in record["down_spectra"].values()
            )
        cpu_data = TokenData(Path("data/tinystories_v1"), "cpu", 10017)
        x, y = next(cpu_data.validation(2, 128, 1))
        record["gradient_flow"] = gradient_flow(model, x, y)
        record["finite_gradients"] = all(
            r["finite"] for r in record["gradient_flow"]["records"].values()
        )
        model.cuda()
        data = TokenData(Path("data/tinystories_v1"), "cuda", 10017)
        assert data.valid.numel() == 200803
        record["data_hashes"] = data.manifest["files"]
        reference_audit = json.loads(TAIL.read_text())
        assert record["data_hashes"] == reference_audit["protocol"]["data_hashes"]
        prefix = list(data.validation(16, 128, 16))
        inputs = [x for x, y in prefix[:3]]
        print("Evaluating native prefix and tail", flush=True)
        record["native"] = quality(model, data, prefix, output, "native")
        if not floored:
            old = reference_audit["records"][run.name]
            record["baseline_reproduced"] = all(
                abs(record["native"][region + "_nll"] - old[region + "_nll"]) <= 0.0002
                for region in ("prefix", "tail")
            )
            assert record["baseline_reproduced"]
        with torch.no_grad():
            references = reference_outputs(model, inputs)
            ids = [id(p) for p in model.parameters()]
            replace_ffns(model)
            record["parameter_objects_unchanged"] = ids == [id(p) for p in model.parameters()]
            assert record["parameter_objects_unchanged"]
            eager_nll, count = evaluate_forward(model, prefix, "cuda", "bf16")
            record["fused_eager_prefix_nll"] = eager_nll
            record["fused_eager_errors"] = cached_errors(model, inputs, references)
            assert count == 32768 and abs(eager_nll - record["native"]["prefix_nll"]) <= 0.001
            assert errors_pass(record["fused_eager_errors"])
            print("Compiling fused full model", flush=True)
            start = time.perf_counter()
            compiled = compile_model(model)
            with autocast("cuda", "bf16"):
                compiled(inputs[0])
            synchronize("cuda")
            record["compile_and_first_forward_seconds"] = time.perf_counter() - start
            record["compiled_errors_before_warmup"] = cached_errors(compiled, inputs, references)
            for _ in range(4):
                with autocast("cuda", "bf16"):
                    compiled(inputs[0])
            record["compiled_errors_after_warmup"] = cached_errors(compiled, inputs, references)
            print("Evaluating compiled prefix and tail", flush=True)
            record["compiled"] = quality(compiled, data, prefix, output, "compiled")
            record["numerical_checks_pass"] = errors_pass(
                record["compiled_errors_before_warmup"] + record["compiled_errors_after_warmup"]
            ) and all(
                abs(record["native"][region + "_nll"] - record["compiled"][region + "_nll"])
                <= 0.001
                for region in ("prefix", "tail")
            )
        record["source_checkpoint_unchanged"] = sha256(run / "checkpoint.pt") == source_hash
        assert record["source_checkpoint_unchanged"] and record["finite_gradients"]
        assert math.isfinite(record["worst_down_condition"]) and record["numerical_checks_pass"]
        record["status"] = "complete"
        write_json(output / "result.json", record)
        print(
            json.dumps(
                {
                    k: record[k]
                    for k in (
                        "seed",
                        "floor_strength",
                        "native",
                        "compiled",
                        "worst_down_condition",
                        "numerical_checks_pass",
                    )
                }
            ),
            flush=True,
        )
        return record
    except Exception as exc:
        record.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        write_json(output / "failure.json", record)
        raise


def audit(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    prov = provenance()
    protocol = {
        "provenance": prov,
        "plan_sha256": sha256(PLAN),
        "reference_audit_sha256": sha256(TAIL),
        "seeds": list(SEEDS),
        "floor_strength": 0.1,
        "worker_timeout_seconds": 420,
        "compiler": {"backend": "inductor", "mode": "default", "fullgraph": True, "dynamic": False},
        "interpretation": "Six isolated workers, no training or paired timing; tail already scored before this intervention.",
    }
    write_json(output / "protocol.json", protocol)
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in prov["source_files"]:
            archive.write(name, name)
        archive.write(PLAN, PLAN.as_posix())
    records = {}
    try:
        for seed in SEEDS:
            for floored in (False, True):
                name = f"s{seed}_" + ("floor" if floored else "original")
                command = [
                    sys.executable,
                    "-X",
                    "faulthandler",
                    "-m",
                    "src.core.joint_conditioning",
                    "worker",
                    "--seed",
                    str(seed),
                    "--output",
                    str(output / name),
                ]
                if floored:
                    command.append("--floored")
                print(f"Starting {name}", flush=True)
                with (output / f"{name}.log").open("w", encoding="utf-8") as log:
                    subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=420
                    )
                records[name] = json.loads((output / name / "result.json").read_text())
                write_json(
                    output / "progress.json", {"status": "running", "completed": list(records)}
                )
                print(f"Completed {name}", flush=True)
        references = json.loads(TAIL.read_text())["records"]
        gates = {}
        for seed in SEEDS:
            controls = {
                k: references[f"scale384_{k}_s{seed}_800"]
                for k in ("full_swiglu", "full_gelu", "calibrated_narrow")
            }
            gates[seed] = compare_seed(
                records[f"s{seed}_original"], records[f"s{seed}_floor"], controls
            )
        result = {
            "status": "complete",
            "protocol": protocol,
            "records": records,
            "gates": gates,
            "all_gates_pass": all(all(g.values()) for g in gates.values()),
        }
        write_json(output / "result.json", result)
        write_json(output / "progress.json", {"status": "complete", "completed": list(records)})
        print(json.dumps({"all_gates_pass": result["all_gates_pass"], "gates": gates}), flush=True)
        return result
    except Exception as exc:
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "completed": list(records),
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "audit"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--floored", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "worker":
        if args.seed is None:
            parser.error("worker requires --seed")
        worker(args.seed, args.floored, args.output)
    else:
        audit(args.output)
