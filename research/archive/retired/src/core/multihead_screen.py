"""Frozen six-trial validation screen for the qualified multi-head comparator."""

import argparse
import ast
import gc
import hashlib
import json
import math
import subprocess
import sys
import traceback
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import torch

from src.core.affine_longer import training_loop
from src.core.benchmark import evaluate, forward_flops
from src.core.config import ModelConfig, TrainConfig
from src.core.data import TokenData, load_manifest
from src.core.optimization import group_summary, initialize_dense_width, parameter_groups
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.trainer import train
from src.core.transformer import Transformer
from src.core.wikitext_screen import RECIPES as CONTROL_RECIPES
from src.core.wikitext_screen import configuration as control_configuration
from src.core.wikitext_screen import output_path as control_path
from src.core.wikitext_screen import promotion, select_best

RECIPES = ("multihead_reference", "multihead_calibrated")
RATES = (0.0003, 0.0006, 0.0012)
CACHE = Path("data/wikitext2_v1")
PLAN = Path("research/multihead_screen_plan.md")
COMPUTATION = (
    "src/core/config.py",
    "src/core/transformer.py",
    "src/core/benchmark.py",
    "src/core/diagnostics.py",
    "src/core/trainer.py",
    "src/core/optimization.py",
    "src/core/data.py",
    "src/dense_ffn/__init__.py",
    "src/blockshuffle_ffn/__init__.py",
    "src/grouped_ffn/__init__.py",
    "src/multihead_ffn/__init__.py",
)


def configuration(recipe, rate):
    raw = json.loads(Path(f"configs/wikitext2_{recipe}_screen.json").read_text())
    return ModelConfig(**raw["model"]), replace(TrainConfig(**raw["training"]), learning_rate=rate)


def output_path(recipe, rate):
    return Path(f"results/runs/wikitext2_{recipe}_lr{round(rate * 1e6)}_s17_200")


def candidate_gates(candidate, references):
    gates = promotion({**references, "blockshuffle": candidate})
    gates["at_least_point_two_percent_better_than_calibrated_narrow"] = (
        candidate["validation_loss"] <= 0.998 * references["calibrated_narrow"]["validation_loss"]
    )
    quality = all(v for k, v in gates.items() if not k.startswith("memory_"))
    return {
        "gates": gates,
        "quality_investigation_passes": quality,
        "full_promotion_passes": all(gates.values()),
    }


def verify_archive(path, metrics):
    with zipfile.ZipFile(path / "source.zip") as archive:
        assert all(
            hashlib.sha256(archive.read(n)).hexdigest() == h
            for n, h in metrics["provenance"]["source_files"].items()
        )


def preflight():
    torch.set_num_threads(4)
    qualified = json.loads(Path("results/multihead_integration_v1/result.json").read_text())
    assert (
        qualified["status"] == "complete"
        and qualified["earns_training_screen"]
        and qualified["old_variants_exact"] == 28
    )
    tested = json.loads(Path("results/verification/multihead_tests_v2.json").read_text())
    assert tested["returncode"] == 0 and tested["source_unchanged_through_tests"]
    assert all(sha256(Path(n)) == tested["source_files"][n] for n in COMPUTATION)
    manifest = load_manifest(CACHE)
    assert all(sha256(CACHE / n) == h for n, h in manifest["files"].items())
    data = TokenData(CACHE, "cpu", 10017)
    batches = list(data.validation(16, 128, 158))
    assert len(batches) == 158 and batches[-1][0].shape == (9, 128)
    assert torch.equal(torch.cat([y.flatten() for _, y in batches]), data.valid[1:322689])
    old = json.loads(Path("results/wikitext_screen_v1/result.json").read_text())
    assert old["status"] == "complete" and len(old["trials"]) == 12
    controls = {r: [] for r in CONTROL_RECIPES}
    records = []
    sampling = None
    literal_files = (
        "src/core/data.py",
        "src/core/optimization.py",
        "src/dense_ffn/__init__.py",
        "src/blockshuffle_ffn/__init__.py",
        "src/grouped_ffn/__init__.py",
    )
    for trial in old["trials"]:
        path = Path("results/runs") / trial["run"]
        assert sha256(path / "metrics.json") == trial["metrics_sha256"]
        m = json.loads((path / "metrics.json").read_text())
        mc, tc = control_configuration(trial["recipe"], trial["rate"])
        assert ModelConfig(**m["model"]) == mc and TrainConfig(**m["training"]) == tc
        assert m["data"]["files"] == manifest["files"] and m["environment"] == environment()
        assert m["training_tokens"] == 409600 and m["validation_tokens"] == 322688
        verify_archive(path, m)
        with zipfile.ZipFile(path / "source.zip") as archive:
            assert all(archive.read(n) == Path(n).read_bytes() for n in literal_files)
            assert training_loop(archive.read("src/core/trainer.py")) == training_loop(
                Path("src/core/trainer.py").read_bytes()
            )
            old_functions = {
                n.name: ast.dump(n, include_attributes=False)
                for n in ast.parse(archive.read("src/core/benchmark.py")).body
                if isinstance(n, ast.FunctionDef)
            }
            new_functions = {
                n.name: ast.dump(n, include_attributes=False)
                for n in ast.parse(Path("src/core/benchmark.py").read_bytes()).body
                if isinstance(n, ast.FunctionDef)
            }
            assert all(
                new_functions[n] == v for n, v in old_functions.items() if n != "forward_flops"
            )
            old_classes = {
                n.name: n
                for n in ast.parse(archive.read("src/core/transformer.py")).body
                if isinstance(n, ast.ClassDef)
            }
            new_classes = {
                n.name: n
                for n in ast.parse(Path("src/core/transformer.py").read_bytes()).body
                if isinstance(n, ast.ClassDef)
            }
            for name in ("RMSNorm", "Attention"):
                assert ast.dump(old_classes[name], include_attributes=False) == ast.dump(
                    new_classes[name], include_attributes=False
                )
            for name in ("Block", "Transformer"):
                for method in ("forward", "loss"):
                    old_methods = [
                        n
                        for n in old_classes[name].body
                        if isinstance(n, ast.FunctionDef) and n.name == method
                    ]
                    new_methods = [
                        n
                        for n in new_classes[name].body
                        if isinstance(n, ast.FunctionDef) and n.name == method
                    ]
                    assert [ast.dump(n, include_attributes=False) for n in old_methods] == [
                        ast.dump(n, include_attributes=False) for n in new_methods
                    ]
        assert all(m[k] == v for k, v in forward_flops(mc).items())
        ckpt = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert (
            ckpt["step"] == 200
            and ckpt["model_config"] == m["model"]
            and ckpt["training_config"] == m["training"]
        )
        if sampling is None:
            sampling = ckpt["sampling_rng"].clone()
        assert torch.equal(sampling, ckpt["sampling_rng"])
        assert sum(t.numel() for t in ckpt["model"].values()) == mc.total_parameters
        del ckpt
        controls[trial["recipe"]].append(m)
        records.append(
            {
                **trial,
                "checkpoint_sha256": sha256(path / "checkpoint.pt"),
                "source_archive_sha256": sha256(path / "source.zip"),
            }
        )
    common = None
    cpu = []
    x, y = data.batch(2, 16)
    for recipe in ("full_swiglu", *RECIPES):
        mc, tc = (
            control_configuration(recipe, RATES[1])
            if recipe == "full_swiglu"
            else configuration(recipe, RATES[1])
        )
        if recipe in RECIPES:
            _, full_tc = control_configuration("full_swiglu", RATES[1])
            assert tc == full_tc
        model = Transformer(mc, 17)
        initialize_dense_width(model, tc)
        shared = {n: p.detach().clone() for n, p in model.named_parameters() if ".ffn." not in n}
        if common is None:
            common = shared
        assert all(torch.equal(t, common[n]) for n, t in shared.items())
        loss = model.loss(x, y)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        cpu.append(
            {
                "recipe": recipe,
                "common_non_ffn_parameters_exact": True,
                "real_training_token_backward_finite": True,
                "optimizer_groups": group_summary(parameter_groups(model, tc)),
                "model": asdict(mc),
                "training": asdict(tc),
            }
        )
        del model, shared, loss
    del common, data, batches
    gc.collect()
    return {
        "controls": records,
        "new_cpu_checks": cpu,
        "data_hashes": manifest["files"],
        "sampling_rng_sha256": hashlib.sha256(sampling.numpy().tobytes()).hexdigest(),
        "validation_stream_exact": True,
        "control_base_sources_and_training_loop_unchanged": True,
        "control_attention_norm_and_forward_asts_unchanged": True,
        "old_28_model_snapshot_record": "results/multihead_integration_v1/preflight.json",
        "current_computation_matches_213_test_snapshot": True,
        "environment": environment(),
    }


def baseline_initial_worker(output):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = TokenData(CACHE, "cuda", 10017)
    rows = []
    for recipe in CONTROL_RECIPES:
        mc, tc = control_configuration(recipe, RATES[1])
        model = Transformer(mc, 17).cuda()
        initialize_dense_width(model, tc)
        if tc.recompute_gate:
            for block in model.blocks:
                block.ffn.recompute_gate = True
                block.ffn.gate_recompute_method = tc.gate_recompute_method
        loss, count = evaluate(model, data.validation(16, 128, 158), "cuda", "bf16")
        assert count == 322688
        references = [
            json.loads((control_path(recipe, rate) / "metrics.json").read_text())[
                "initial_validation_loss"
            ]
            for rate in RATES
        ]
        assert all(abs(loss - v) < 1e-7 for v in references)
        rows.append(
            {
                "recipe": recipe,
                "current_initial_nll": loss,
                "original_initial_nlls": references,
                "targets": count,
                "matches_all_rates": True,
            }
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(
        output,
        {
            "status": "PASS",
            "rows": rows,
            "data_hashes": data.manifest["files"],
            "provenance": provenance(),
            "environment": environment(),
        },
    )
    print(json.dumps(rows), flush=True)


def verify_trial(recipe, rate, initial):
    path = output_path(recipe, rate)
    m = json.loads((path / "metrics.json").read_text())
    mc, tc = configuration(recipe, rate)
    assert ModelConfig(**m["model"]) == mc and TrainConfig(**m["training"]) == tc
    assert m["training_tokens"] == 409600 and m["validation_tokens"] == 322688
    assert (
        m["data"]["files"] == initial["data_hashes"] and m["environment"] == initial["environment"]
    )
    assert (
        m["ffn_parameters"] == mc.unique_ffn_parameters
        and m["total_parameters"] == mc.total_parameters
    )
    expected = next(r for r in initial["new_cpu_checks"] if r["recipe"] == recipe)
    assert (
        m["optimizer_parameter_groups"] == expected["optimizer_groups"]
        and m["activation_compilation"] is None
    )
    history = [json.loads(line) for line in (path / "history.jsonl").read_text().splitlines()]
    assert [r["step"] for r in history] == [1, 50, 100, 150, 200]
    assert history[-1]["validation_loss"] == m["validation_loss"]
    for row in history:
        assert all(
            math.isfinite(row[k])
            for k in ("training_loss", "validation_loss", "gradient_norm_pre_clip")
        )
        assert all(math.isfinite(v) for v in row["layer_gradient_norms_post_clip"].values())
    for stage in ("initial", "final"):
        d = json.loads((path / f"{stage}_diagnostics.json").read_text())
        assert all(r["finite"] and r.get("routing_finite", True) for r in d.values())
    verify_archive(path, m)
    ckpt = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert (
        ckpt["step"] == 200
        and ckpt["model_config"] == m["model"]
        and ckpt["training_config"] == m["training"]
    )
    assert sum(t.numel() for t in ckpt["model"].values()) == mc.total_parameters
    assert (
        hashlib.sha256(ckpt["sampling_rng"].numpy().tobytes()).hexdigest()
        == initial["sampling_rng_sha256"]
    )
    return m


def audit(output):
    output.mkdir(parents=True, exist_ok=False)
    completed = []
    try:
        initial = preflight()
        write_json(output / "preflight.json", initial)
        protocol = {
            "plan_sha256": sha256(PLAN),
            "provenance": provenance(),
            "environment": environment(),
            "rates": RATES,
            "recipes": RECIPES,
            "new_training_tokens": 2457600,
            "training_worker_timeout_seconds": 1200,
            "initial_worker_timeout_seconds": 600,
            "config_hashes": {
                r: sha256(Path(f"configs/wikitext2_{r}_screen.json")) for r in RECIPES
            },
        }
        write_json(output / "protocol.json", protocol)
        with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for n in protocol["provenance"]["source_files"]:
                archive.write(n, n)
            archive.write(PLAN, PLAN.as_posix())
        print("Reproducing the four original initial GPU validation losses", flush=True)
        with (output / "baseline_initial.log").open("x", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "faulthandler",
                    "-m",
                    "src.core.multihead_screen",
                    "baseline-initial",
                    "--output",
                    str(output / "baseline_initial.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
        baseline = json.loads((output / "baseline_initial.json").read_text())
        assert baseline["status"] == "PASS" and all(
            baseline["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
            for n in COMPUTATION
        )
        rows = {r: [] for r in RECIPES}
        for index, rate in enumerate(RATES):
            order = RECIPES if index % 2 == 0 else RECIPES[::-1]
            for recipe in order:
                path = output_path(recipe, rate)
                assert not path.exists(), f"Do not overwrite {path}"
                print(f"Starting {recipe}, LR{rate}, 200 steps", flush=True)
                write_json(
                    output / "progress.json",
                    {"status": "running", "active": path.name, "completed": completed},
                )
                with (output / f"{path.name}.log").open("x", encoding="utf-8") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "-X",
                            "faulthandler",
                            "-m",
                            "src.core.multihead_screen",
                            "worker",
                            "--recipe",
                            recipe,
                            "--rate",
                            str(rate),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=1200,
                    )
                m = verify_trial(recipe, rate, initial)
                assert all(
                    m["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
                    for n in (*COMPUTATION, "src/core/multihead_screen.py")
                )
                rows[recipe].append(m)
                completed.append(
                    {
                        "recipe": recipe,
                        "rate": rate,
                        "run": path.name,
                        "nll": m["validation_loss"],
                        "peak_mib": m["peak_allocated_vram_bytes"] / 2**20,
                        "metrics_sha256": sha256(path / "metrics.json"),
                        "checkpoint_sha256": sha256(path / "checkpoint.pt"),
                    }
                )
                print(json.dumps(completed[-1]), flush=True)
        controls = {
            r: [json.loads((control_path(r, rate) / "metrics.json").read_text()) for rate in RATES]
            for r in CONTROL_RECIPES
        }
        references = {r: select_best(v) for r, v in controls.items()}
        selected = {r: select_best(v) for r, v in rows.items()}
        decisions = {r: candidate_gates(m, references) for r, m in selected.items()}
        same_rate = {
            r: {
                str(round(rate * 1e6)): {
                    c: 100 * (rows[r][i]["validation_loss"] / controls[c][i]["validation_loss"] - 1)
                    for c in CONTROL_RECIPES
                }
                for i, rate in enumerate(RATES)
            }
            for r in RECIPES
        }
        for r, values in rows.items():
            assert (
                max(m["initial_validation_loss"] for m in values)
                - min(m["initial_validation_loss"] for m in values)
                < 1e-7
            )
        result = {
            "status": "complete",
            "trials": completed,
            "selected": {r: m["run"] for r, m in selected.items()},
            "selected_controls": {r: m["run"] for r, m in references.items()},
            "decisions": decisions,
            "same_rate_relative_nll_percent": same_rate,
            "grid_boundary_winners": [
                r
                for r, m in selected.items()
                if m["training"]["learning_rate"] in (RATES[0], RATES[-1])
            ],
            "final_sampling_rng_matches_all_eighteen_runs": True,
            "all_diagnostics_finite": True,
            "initial_nll_consistent_across_rates": True,
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
    parser.add_argument("command", choices=("preflight", "baseline-initial", "worker", "audit"))
    parser.add_argument("--recipe", choices=RECIPES)
    parser.add_argument("--rate", type=float, choices=RATES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        if args.recipe is None or args.rate is None:
            parser.error("worker requires recipe and rate")
        mc, tc = configuration(args.recipe, args.rate)
        train(mc, tc, CACHE, output_path(args.recipe, args.rate))
    elif args.output is None:
        parser.error("this command requires output")
    elif args.command == "preflight":
        assert not args.output.exists()
        result = preflight()
        write_json(args.output, result)
        print(
            json.dumps(
                {
                    "controls_verified": len(result["controls"]),
                    "sampling_rng_sha256": result["sampling_rng_sha256"],
                }
            )
        )
    elif args.command == "baseline-initial":
        assert not args.output.exists()
        baseline_initial_worker(args.output)
    else:
        audit(args.output)
