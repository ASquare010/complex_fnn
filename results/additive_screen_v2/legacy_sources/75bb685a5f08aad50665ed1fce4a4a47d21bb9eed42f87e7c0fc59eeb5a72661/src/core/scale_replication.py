"""Frozen larger-model replication; one fresh training worker at a time."""

import argparse
import ast
import gc
import json
import subprocess
import sys
import traceback
import zipfile
from dataclasses import asdict
from pathlib import Path

import torch

from src.core.config import ModelConfig, TrainConfig
from src.core.optimization import initialize_dense_width
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.trainer import train
from src.core.transformer import Transformer

RECIPES = ("full_swiglu", "full_gelu", "calibrated_narrow", "blockshuffle")


def configuration(recipe, seed):
    raw = json.loads(Path(f"configs/scale384_{recipe}_800.json").read_text())
    raw["training"]["seed"] = seed
    return ModelConfig(**raw["model"]), TrainConfig(**raw["training"])


def output_path(recipe, seed):
    return Path(f"results/runs/scale384_{recipe}_s{seed}_800")


def preflight():
    torch.set_num_threads(4)
    source = Path("results/runs/scale384_blockshuffle_s17_800/source.zip")
    files = [
        "src/core/trainer.py",
        "src/core/transformer.py",
        "src/core/config.py",
        "src/core/optimization.py",
        "src/core/data.py",
        "src/blockshuffle_ffn/__init__.py",
        "src/grouped_ffn/__init__.py",
        "src/dense_ffn/__init__.py",
    ]
    checks = {}
    with zipfile.ZipFile(source) as archive:
        for name in files:
            same = archive.read(name) == Path(name).read_bytes()
            assert same, name
            checks[name] = {"exact_source_match_to_seed17": same, "sha256": sha256(Path(name))}
        old = ast.parse(archive.read("src/core/benchmark.py").decode())
        new = ast.parse(Path("src/core/benchmark.py").read_text())
        old_functions = {
            n.name: ast.dump(n, include_attributes=False)
            for n in old.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        new_functions = {
            n.name: ast.dump(n, include_attributes=False)
            for n in new.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert all(new_functions[k] == v for k, v in old_functions.items())
        checks["src/core/benchmark.py"] = {
            "all_original_function_asts_unchanged": True,
            "new_functions_only": sorted(set(new_functions) - set(old_functions)),
        }
    seeds = {}
    for seed in (29, 43):
        common = None
        rows = {}
        for recipe in RECIPES:
            mc, tc = configuration(recipe, seed)
            model = Transformer(mc, seed)
            initialize_dense_width(model, tc)
            values = {
                name: p.detach().clone()
                for name, p in model.named_parameters()
                if ".ffn." not in name
            }
            if common is None:
                common = values
            assert all(torch.equal(value, common[name]) for name, value in values.items())
            x = torch.arange(16).reshape(2, 8)
            loss = model.loss(x, (x + 1) % mc.vocab_size)
            loss.backward()
            assert torch.isfinite(loss) and all(
                torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
            )
            rows[recipe] = {
                "common_initial_tensors_equal": True,
                "cpu_tiny_backward_finite": True,
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "ffn_parameters": mc.ffn_parameters * mc.layers,
                "model_config": asdict(mc),
                "training_config": asdict(tc),
            }
            del model, values, loss
        seeds[str(seed)] = rows
        del common
    gc.collect()
    return {
        "training_source_checks": checks,
        "seeds": seeds,
        "environment": environment(),
        "provenance": provenance(),
    }


def worker(recipe, seed):
    mc, tc = configuration(recipe, seed)
    return train(mc, tc, Path("data/tinystories_v1"), output_path(recipe, seed))


def audit(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    initial_path = Path("results/verification/scale_replication_initial.json")
    initial = json.loads(initial_path.read_text()) if initial_path.exists() else preflight()
    assert initial["environment"] == environment()
    for name in initial["training_source_checks"]:
        assert sha256(Path(name)) == initial["provenance"]["source_files"][name]
    for seed in (29, 43):
        for recipe in RECIPES:
            mc, tc = configuration(recipe, seed)
            row = initial["seeds"][str(seed)][recipe]
            assert row["model_config"] == asdict(mc) and row["training_config"] == asdict(tc)
            assert row["common_initial_tensors_equal"] and row["cpu_tiny_backward_finite"]
    protocol = {
        "plan_sha256": sha256(Path("research/scale_replication_plan.md")),
        "provenance": provenance(),
        "environment": environment(),
        "seeds": [29, 43],
        "recipes": list(RECIPES),
        "worker_timeout_seconds": 900,
        "preflight": initial,
    }
    write_json(output / "protocol.json", protocol)
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in protocol["provenance"]["source_files"]:
            archive.write(name, name)
        archive.write("research/scale_replication_plan.md", "research/scale_replication_plan.md")
    completed = []
    try:
        for seed in (29, 43):
            for recipe in RECIPES:
                path = output_path(recipe, seed)
                if path.exists():
                    raise ValueError(
                        f"Run already exists: {path}; inspect evidence before any retry"
                    )
                print(f"Starting {recipe}, seed{seed}, 800 steps", flush=True)
                with (output / f"{recipe}_s{seed}.log").open("w", encoding="utf-8") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "-X",
                            "faulthandler",
                            "-m",
                            "src.core.scale_replication",
                            "worker",
                            "--recipe",
                            recipe,
                            "--seed",
                            str(seed),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=900,
                    )
                metrics = json.loads((path / "metrics.json").read_text())
                completed.append(
                    {
                        "run": path.name,
                        "validation_nll": metrics["validation_loss"],
                        "metrics_sha256": sha256(path / "metrics.json"),
                    }
                )
                write_json(output / "progress.json", {"completed": completed, "status": "running"})
                print(json.dumps(completed[-1]), flush=True)
        write_json(output / "progress.json", {"completed": completed, "status": "complete"})
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
    parser.add_argument("--seed", type=int, choices=(29, 43))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        if args.recipe is None or args.seed is None:
            parser.error("worker needs recipe and seed")
        worker(args.recipe, args.seed)
    elif args.command == "preflight":
        result = preflight()
        if args.output:
            write_json(args.output, result)
        print(
            json.dumps(
                {
                    "source_checks": len(result["training_source_checks"]),
                    "seeds": list(result["seeds"]),
                },
                indent=2,
            )
        )
    else:
        if args.output is None:
            parser.error("audit needs output")
        audit(args.output)
