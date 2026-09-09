"""Frozen five-recipe, 800-step WikiText comparison after affine promotion."""

import argparse
import ast
import hashlib
import json
import math
import subprocess
import sys
import traceback
import zipfile
from dataclasses import replace
from pathlib import Path

import torch

from src.core.activation_screen import CACHE
from src.core.benchmark import forward_flops
from src.core.config import ModelConfig, TrainConfig
from src.core.data import TokenData
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.trainer import train
from src.core.wikitext_screen import promotion as reference_gates

RECIPES = ("full_swiglu", "calibrated_narrow", "blockshuffle", "full_gelu", "blockshuffle_affine")
PLAN = Path("research/affine_activation_longer_plan.md")
CRITICAL = (
    "src/core/trainer.py",
    "src/core/transformer.py",
    "src/core/config.py",
    "src/core/data.py",
    "src/core/optimization.py",
    "src/core/benchmark.py",
    "src/core/diagnostics.py",
    "src/blockshuffle_ffn/__init__.py",
    "src/dense_ffn/__init__.py",
    "src/learnable_activation_ffn/__init__.py",
)


def configuration(recipe, seed=17):
    raw = json.loads(Path(f"configs/wikitext2_{recipe}_800.json").read_text())
    return ModelConfig(**raw["model"]), replace(TrainConfig(**raw["training"]), seed=seed)


def output_path(recipe, seed=17):
    _, tc = configuration(recipe, seed)
    return Path(f"results/runs/wikitext2_{recipe}_lr{round(tc.learning_rate * 1e6)}_s{seed}_800")


def selected_sources():
    old = json.loads(Path("results/wikitext_screen_v1/result.json").read_text())
    affine = json.loads(Path("results/affine_activation_screen_v1/result.json").read_text())
    assert old["status"] == affine["status"] == "complete"
    assert affine["earns_longer_training"]["blockshuffle_affine"]
    return {**old["selected"], "blockshuffle_affine": affine["selected"]["blockshuffle_affine"]}


def promotion(rows):
    candidate = rows["blockshuffle_affine"]
    gates = reference_gates({**rows, "blockshuffle": candidate})
    gates["at_least_point_two_percent_better_than_unmodified_blockshuffle"] = (
        candidate["validation_loss"] <= 0.998 * rows["blockshuffle"]["validation_loss"]
    )
    return gates


def training_loop(source):
    loops = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "step"
    ]
    assert len(loops) == 1
    return ast.dump(loops[0], include_attributes=False)


def preflight():
    torch.set_num_threads(4)
    tested = json.loads(Path("results/verification/affine_tests_v3.json").read_text())
    assert tested["returncode"] == 0 and tested["source_unchanged_through_tests"]
    assert all(
        sha256(Path(name)) == tested["provenance"]["source_files"][name] for name in CRITICAL
    )
    data = TokenData(CACHE, "cpu", 10017)
    batches = list(data.validation(16, 128, 158))
    assert len(batches) == 158 and batches[-1][0].shape == (9, 128)
    assert torch.equal(torch.cat([y.flatten() for _, y in batches]), data.valid[1:322689])
    sources = {}
    for recipe, run in selected_sources().items():
        path = Path("results/runs") / run
        m = json.loads((path / "metrics.json").read_text())
        mc, tc = configuration(recipe)
        mc.validate()
        tc.validate()
        assert mc == ModelConfig(**m["model"])
        assert tc == replace(TrainConfig(**m["training"]), steps=800, log_every=200)
        assert tc.seed == 17 and tc.activation_backend == "eager"
        assert m["data"]["files"] == data.manifest["files"]
        with zipfile.ZipFile(path / "source.zip") as archive:
            assert all(
                hashlib.sha256(archive.read(n)).hexdigest() == h
                for n, h in m["provenance"]["source_files"].items()
            )
            assert training_loop(archive.read("src/core/trainer.py")) == training_loop(
                Path("src/core/trainer.py").read_bytes()
            )
            for name in (
                "src/core/data.py",
                "src/core/optimization.py",
                "src/dense_ffn/__init__.py",
                "src/blockshuffle_ffn/__init__.py",
            ):
                assert archive.read(name) == Path(name).read_bytes()
        with zipfile.ZipFile(path / "source.zip") as archive:
            old_tree = ast.parse(archive.read("src/core/benchmark.py"))
            new_tree = ast.parse(Path("src/core/benchmark.py").read_bytes())

            def without_counting(tree):
                return [
                    ast.dump(n, include_attributes=False)
                    for n in tree.body
                    if not (isinstance(n, ast.FunctionDef) and n.name == "forward_flops")
                ]

            assert without_counting(old_tree) == without_counting(new_tree)
        assert all(m[key] == value for key, value in forward_flops(mc).items())
        checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert checkpoint["step"] == 200
        assert sum(t.numel() for t in checkpoint["model"].values()) == mc.total_parameters
        del checkpoint
        sources[recipe] = {
            "run": run,
            "metrics_sha256": sha256(path / "metrics.json"),
            "checkpoint_sha256": sha256(path / "checkpoint.pt"),
            "initial_nll": m["initial_validation_loss"],
            "all_archive_hashes_match": True,
        }
    return {
        "sources": sources,
        "cache_hashes": data.manifest["files"],
        "validation_targets": 322688,
        "validation_stream_exact": True,
        "critical_sources_match_198_test_snapshot": True,
        "benchmark_execution_ast_unchanged": True,
        "historical_matrix_work_counts_reproduced": True,
        "environment": environment(),
    }


def verify_trial(recipe, initial, seed=17):
    path = output_path(recipe, seed)
    m = json.loads((path / "metrics.json").read_text())
    mc, tc = configuration(recipe, seed)
    assert mc == ModelConfig(**m["model"]) and tc == TrainConfig(**m["training"])
    assert m["training_tokens"] == 1638400 and m["validation_tokens"] == 322688
    assert (
        m["ffn_parameters"] == mc.unique_ffn_parameters
        and m["total_parameters"] == mc.total_parameters
    )
    assert (
        m["data"]["files"] == initial["cache_hashes"] and m["environment"] == initial["environment"]
    )
    source = Path("results/runs") / initial["sources"][recipe]["run"]
    prior = json.loads((source / "metrics.json").read_text())
    assert sha256(source / "metrics.json") == initial["sources"][recipe]["metrics_sha256"]
    if seed == 17:
        assert abs(m["initial_validation_loss"] - prior["initial_validation_loss"]) < 1e-7
    assert m["optimizer_parameter_groups"] == prior["optimizer_parameter_groups"]
    assert m["ffn_width_initialization"] == prior["ffn_width_initialization"]
    assert m["activation_compilation"] is None
    history = [json.loads(line) for line in (path / "history.jsonl").read_text().splitlines()]
    assert [r["step"] for r in history] == [1, 200, 400, 600, 800]
    assert history[-1]["validation_loss"] == m["validation_loss"]
    for row in history:
        assert all(
            math.isfinite(row[k])
            for k in ("training_loss", "validation_loss", "gradient_norm_pre_clip")
        )
        assert all(math.isfinite(v) for v in row["layer_gradient_norms_post_clip"].values())
    for stage in ("initial", "final"):
        diagnostics = json.loads((path / f"{stage}_diagnostics.json").read_text())
        assert all(r["finite"] and r.get("slope_finite", True) for r in diagnostics.values())
    with zipfile.ZipFile(path / "source.zip") as archive:
        assert all(
            hashlib.sha256(archive.read(n)).hexdigest() == h
            for n, h in m["provenance"]["source_files"].items()
        )
    checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["step"] == 800 and checkpoint["model_config"] == m["model"]
    assert checkpoint["training_config"] == m["training"]
    assert sum(t.numel() for t in checkpoint["model"].values()) == m["total_parameters"]
    sampling = checkpoint["sampling_rng"].clone()
    del checkpoint
    return m, sampling


def audit(output):
    output.mkdir(parents=True, exist_ok=False)
    completed = []
    try:
        initial = preflight()
        write_json(output / "preflight.json", initial)
        protocol = {
            "plan_sha256": sha256(PLAN),
            "provenance": provenance(),
            "environment": initial["environment"],
            "recipes": RECIPES,
            "steps": 800,
            "seed": 17,
            "worker_timeout_seconds": 2400,
            "training_tokens_per_trial": 1638400,
            "validation_targets": 322688,
            "config_hashes": {r: sha256(Path(f"configs/wikitext2_{r}_800.json")) for r in RECIPES},
        }
        write_json(output / "protocol.json", protocol)
        with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in protocol["provenance"]["source_files"]:
                archive.write(name, name)
            archive.write(PLAN, PLAN.as_posix())
        rows = {}
        sampling = None
        for recipe in RECIPES:
            path = output_path(recipe)
            assert not path.exists(), f"Do not overwrite {path}"
            print(f"Starting {recipe}: 800 steps", flush=True)
            with (output / f"{recipe}.log").open("x", encoding="utf-8") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-X",
                        "faulthandler",
                        "-m",
                        "src.core.affine_longer",
                        "worker",
                        "--recipe",
                        recipe,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=2400,
                    check=True,
                )
            m, rng = verify_trial(recipe, initial)
            assert all(
                m["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
                for n in CRITICAL
            )
            if sampling is None:
                sampling = rng
            assert torch.equal(sampling, rng)
            rows[recipe] = m
            completed.append(
                {
                    "recipe": recipe,
                    "run": path.name,
                    "validation_nll": m["validation_loss"],
                    "peak_allocated_vram_bytes": m["peak_allocated_vram_bytes"],
                    "metrics_sha256": sha256(path / "metrics.json"),
                    "checkpoint_sha256": sha256(path / "checkpoint.pt"),
                }
            )
            write_json(output / "progress.json", {"status": "running", "completed": completed})
            print(json.dumps(completed[-1]), flush=True)
        gates = promotion(rows)
        result = {
            "status": "complete",
            "trials": completed,
            "gates": gates,
            "earns_independent_seeds": all(gates.values()),
            "all_diagnostics_finite": True,
            "initial_nll_matches_all_sources": True,
            "sampling_rng_matches_all_recipes": True,
            "final_sampling_rng_sha256": hashlib.sha256(sampling.numpy().tobytes()).hexdigest(),
            "plan_sha256": protocol["plan_sha256"],
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
    parser.add_argument("command", choices=("worker", "audit"))
    parser.add_argument("--recipe", choices=RECIPES)
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        if not args.recipe:
            parser.error("worker requires recipe")
        mc, tc = configuration(args.recipe, args.seed)
        train(mc, tc, CACHE, output_path(args.recipe, args.seed))
    else:
        if args.output is None:
            parser.error("audit requires output")
        audit(args.output)
