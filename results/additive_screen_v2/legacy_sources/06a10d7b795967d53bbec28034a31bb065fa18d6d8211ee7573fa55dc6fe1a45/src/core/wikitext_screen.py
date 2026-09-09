"""Equal new tuning budget for four locked recipes on the pinned WikiText cache."""

import argparse
import ast
import gc
import json
import math
import subprocess
import sys
import traceback
import zipfile
from dataclasses import asdict
from pathlib import Path

import torch

from src.core.config import ModelConfig, TrainConfig
from src.core.data import TokenData, load_manifest
from src.core.optimization import group_summary, initialize_dense_width, parameter_groups
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.trainer import train
from src.core.transformer import Transformer

RECIPES = ("full_swiglu", "full_gelu", "calibrated_narrow", "blockshuffle")
RATES = (0.0003, 0.0006, 0.0012)
CACHE = Path("data/wikitext2_v1")
PLAN = Path("research/wikitext_screen_plan.md")


def configuration(recipe, rate):
    raw = json.loads(Path(f"configs/wikitext2_{recipe}_screen.json").read_text())
    raw["training"]["learning_rate"] = rate
    return ModelConfig(**raw["model"]), TrainConfig(**raw["training"])


def output_path(recipe, rate):
    return Path(f"results/runs/wikitext2_{recipe}_lr{round(rate * 1e6)}_s17_200")


def select_best(rows):
    """Final NLL wins; exact ties use the lower peak rate."""
    valid = [r for r in rows if r.get("status") != "FAILED" and math.isfinite(r["validation_loss"])]
    if not valid:
        raise ValueError("No finite complete trial for this recipe")
    return min(valid, key=lambda r: (r["validation_loss"], r["training"]["learning_rate"]))


def promotion(selected):
    candidate = selected["blockshuffle"]
    loss = candidate["validation_loss"]
    gates = {
        "at_least_70_percent_fewer_ffn_weights": candidate["ffn_reduction_percent"] >= 70,
        "beats_calibrated_narrow": loss < selected["calibrated_narrow"]["validation_loss"],
    }
    for recipe in ("full_swiglu", "full_gelu"):
        gates[f"within_one_percent_{recipe}"] = loss <= 1.01 * selected[recipe]["validation_loss"]
        gates[f"memory_within_ten_percent_{recipe}"] = (
            candidate["peak_allocated_vram_bytes"]
            <= 1.1 * selected[recipe]["peak_allocated_vram_bytes"]
        )
    return gates


def preflight():
    torch.set_num_threads(4)
    manifest = load_manifest(CACHE)
    reconstructed = json.loads(
        Path("results/verification/wikitext_reconstruction_v1.json").read_text()
    )
    assert reconstructed["all_file_hashes_match"] and manifest["files"] == reconstructed["files"]
    assert (
        manifest["splits"]["train"]["tokens"] == 3083650
        and manifest["splits"]["valid"]["tokens"] == 322802
    )
    data = TokenData(CACHE, "cpu", 10017)
    batches = list(data.validation(16, 128, 158))
    assert len(batches) == 158 and batches[-1][0].shape == (9, 128)
    assert sum(y.numel() for _, y in batches) == 322688
    # Target stream must be exact; the final partial batch cannot be dropped.
    assert torch.equal(torch.cat([y.flatten() for _, y in batches]), data.valid[1:322689])
    source = Path("results/runs/scale384_blockshuffle_s17_800/source.zip")
    critical = (
        "src/core/trainer.py",
        "src/core/transformer.py",
        "src/core/config.py",
        "src/core/optimization.py",
        "src/core/data.py",
        "src/blockshuffle_ffn/__init__.py",
        "src/grouped_ffn/__init__.py",
        "src/dense_ffn/__init__.py",
    )
    with zipfile.ZipFile(source) as archive:
        assert all(archive.read(name) == Path(name).read_bytes() for name in critical)
        old = ast.parse(archive.read("src/core/benchmark.py").decode())
        new = ast.parse(Path("src/core/benchmark.py").read_text())
        old_functions = {n.name: ast.dump(n) for n in old.body if isinstance(n, ast.FunctionDef)}
        new_functions = {n.name: ast.dump(n) for n in new.body if isinstance(n, ast.FunctionDef)}
        assert all(new_functions[k] == v for k, v in old_functions.items())
    common, records = None, {}
    x, y = batches[0]
    for recipe in RECIPES:
        mc, tc = configuration(recipe, RATES[1])
        mc.validate()
        tc.validate()
        model = Transformer(mc, 17)
        initialize_dense_width(model, tc)
        if tc.recompute_gate:
            for block in model.blocks:
                block.ffn.recompute_gate = True
                block.ffn.gate_recompute_method = tc.gate_recompute_method
        values = {
            name: p.detach().clone() for name, p in model.named_parameters() if ".ffn." not in name
        }
        if common is None:
            common = values
        assert all(torch.equal(value, common[name]) for name, value in values.items())
        count = sum(p.numel() for p in model.parameters())
        ffn = sum(p.numel() for b in model.blocks for p in b.ffn.parameters())
        assert count == mc.total_parameters and ffn == mc.unique_ffn_parameters
        loss = model.loss(x[:2, :16], y[:2, :16])
        loss.backward()
        assert torch.isfinite(loss) and all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
        )
        records[recipe] = {
            "model": asdict(mc),
            "training_base": asdict(tc),
            "actual_total_parameters": count,
            "actual_ffn_parameters": ffn,
            "common_initial_tensors_equal": True,
            "real_token_cpu_backward_finite": True,
            "optimizer_groups": group_summary(parameter_groups(model, tc)),
        }
        del model, values, loss
    del common
    gc.collect()
    return {
        "environment": environment(),
        "provenance": provenance(),
        "cache_hashes": manifest["files"],
        "critical_sources_unchanged": list(critical),
        "original_benchmark_functions_unchanged": True,
        "validation_target_stream_exact": True,
        "final_batch_shape": [9, 128],
        "validation_targets": 322688,
        "recipes": records,
    }


def audit(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    initial = preflight()
    write_json(output / "preflight.json", initial)
    protocol = {
        "plan_sha256": sha256(PLAN),
        "provenance": provenance(),
        "environment": initial["environment"],
        "cache_manifest_sha256": sha256(CACHE / "manifest.json"),
        "rates": list(RATES),
        "recipes": list(RECIPES),
        "seed": 17,
        "steps": 200,
        "worker_timeout_seconds": 600,
        "train_tokens_per_trial": 409600,
        "validation_targets": 322688,
    }
    write_json(output / "protocol.json", protocol)
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in protocol["provenance"]["source_files"]:
            archive.write(name, name)
        archive.write(PLAN, PLAN.as_posix())
    rows = {k: [] for k in RECIPES}
    completed = []
    try:
        for index, rate in enumerate(RATES):
            order = RECIPES[index:] + RECIPES[:index]
            for recipe in order:
                path = output_path(recipe, rate)
                if path.exists():
                    raise ValueError(f"Existing run requires investigation, not overwrite: {path}")
                print(f"Starting {recipe}, peak LR {rate:g}", flush=True)
                with (output / f"{path.name}.log").open("x", encoding="utf-8") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "-X",
                            "faulthandler",
                            "-m",
                            "src.core.wikitext_screen",
                            "worker",
                            "--recipe",
                            recipe,
                            "--rate",
                            str(rate),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=600,
                    )
                metrics = json.loads((path / "metrics.json").read_text())
                assert (
                    metrics["validation_tokens"] == 322688 and metrics["training_tokens"] == 409600
                )
                assert metrics["data"]["files"] == initial["cache_hashes"]
                assert math.isfinite(metrics["validation_loss"]) and math.isfinite(
                    metrics["final_gradient_norm"]
                )
                for stage in ("initial", "final"):
                    diagnostics = json.loads((path / f"{stage}_diagnostics.json").read_text())
                    assert all(row["finite"] for row in diagnostics.values())
                rows[recipe].append(metrics)
                completed.append(
                    {
                        "recipe": recipe,
                        "rate": rate,
                        "run": path.name,
                        "validation_nll": metrics["validation_loss"],
                        "metrics_sha256": sha256(path / "metrics.json"),
                    }
                )
                write_json(output / "progress.json", {"status": "running", "completed": completed})
                print(json.dumps(completed[-1]), flush=True)
        selected = {k: select_best(v) for k, v in rows.items()}
        gates = promotion(selected)
        gates["all_trial_activation_diagnostics_finite"] = True
        result = {
            "status": "complete",
            "protocol": protocol,
            "trials": completed,
            "selected": {k: v["run"] for k, v in selected.items()},
            "promotion_gates": gates,
            "earns_800_steps": all(gates.values()),
            "grid_boundary_winners": [
                k
                for k, v in selected.items()
                if v["training"]["learning_rate"] in (RATES[0], RATES[-1])
            ],
        }
        write_json(output / "result.json", result)
        write_json(output / "progress.json", {"status": "complete", "completed": completed})
        print(json.dumps(result), flush=True)
        return result
    except Exception as exc:
        write_json(
            output / "failure.json",
            {"completed": completed, "error": repr(exc), "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "audit", "preflight"))
    parser.add_argument("--recipe", choices=RECIPES)
    parser.add_argument("--rate", type=float, choices=RATES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        if args.recipe is None or args.rate is None:
            parser.error("worker requires --recipe and --rate")
        mc, tc = configuration(args.recipe, args.rate)
        train(mc, tc, CACHE, output_path(args.recipe, args.rate))
    elif args.command == "preflight":
        result = preflight()
        if args.output:
            write_json(args.output, result)
        print(json.dumps({k: v for k, v in result.items() if k != "provenance"}), flush=True)
    else:
        if args.output is None:
            parser.error("audit requires --output")
        audit(args.output)
