"""Frozen learnable-activation experiments on both compressed WikiText bases."""

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

from src.core.benchmark import autocast, forward_flops
from src.core.config import ModelConfig, TrainConfig
from src.core.data import TokenData, load_manifest
from src.core.optimization import group_summary, initialize_dense_width, parameter_groups
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.trainer import train
from src.core.transformer import Transformer
from src.core.wikitext_screen import promotion as reference_gates
from src.core.wikitext_screen import select_best

RECIPES = (
    "calibrated_narrow_shifted",
    "blockshuffle_shifted",
    "calibrated_narrow_rational",
    "blockshuffle_rational",
)
AFFINE_RECIPES = ("calibrated_narrow_affine", "blockshuffle_affine")
RATES = (0.0003, 0.0006, 0.0012)
CACHE = Path("data/wikitext2_v1")
PLAN = Path("research/learnable_activation_plan.md")


def configuration(recipe, rate):
    raw = json.loads(Path(f"configs/wikitext2_{recipe}_screen.json").read_text())
    raw["training"]["learning_rate"] = rate
    return ModelConfig(**raw["model"]), TrainConfig(**raw["training"])


def output_path(recipe, rate):
    return Path(f"results/runs/wikitext2_{recipe}_lr{round(rate * 1e6)}_s17_200")


def controls():
    result = json.loads(Path("results/wikitext_screen_v1/result.json").read_text())
    assert result["status"] == "complete" and len(result["trials"]) == 12
    return {
        k: json.loads(Path(f"results/runs/{run}/metrics.json").read_text())
        for k, run in result["selected"].items()
    }


def promotion(candidate, base, references):
    gates = reference_gates({**references, "blockshuffle": candidate})
    gates["at_least_point_two_percent_better_than_own_base"] = (
        candidate["validation_loss"] <= 0.998 * references[base]["validation_loss"]
    )
    if candidate.get("model", {}).get("variant", "").endswith("_affine_activation"):
        prior = json.loads(Path("results/activation_screen_v1/result.json").read_text())
        rational_run = prior["selected"][f"{base}_rational"]
        rational = json.loads(Path(f"results/runs/{rational_run}/metrics.json").read_text())
        gates["within_point_two_percent_of_selected_rational"] = (
            candidate["validation_loss"] <= 1.002 * rational["validation_loss"]
        )
    return gates


def set_execution(model, tc):
    if tc.recompute_gate:
        for block in model.blocks:
            block.ffn.recompute_gate = True
            block.ffn.gate_recompute_method = tc.gate_recompute_method


def preflight(
    snapshot=Path("results/verification/activation_before_v1.pt"), families=("shifted", "rational")
):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    before = torch.load(snapshot, map_location="cpu", weights_only=True)
    original_variant_count = len(before)
    for variant, row in before.items():
        mc = ModelConfig(**row["config"])
        model = Transformer(mc, 17)
        x = torch.arange(16).reshape(2, 8) % 32
        loss = model.loss(x, (x + 1) % 32)
        loss.backward()
        assert forward_flops(mc) == row["flops"]
        assert torch.equal(model(x), row["logits"]) and torch.equal(loss, row["loss"])
        assert all(
            torch.equal(value, row["state"][name]) for name, value in model.state_dict().items()
        )
        assert all(
            torch.equal(p.grad, row["gradients"][name]) for name, p in model.named_parameters()
        )
    del before, model, loss
    manifest = load_manifest(CACHE)
    for name, expected in manifest["files"].items():
        assert sha256(CACHE / name) == expected
    data = TokenData(CACHE, "cuda", 10017)
    batches = list(data.validation(16, 128, 158))
    assert len(batches) == 158 and batches[-1][0].shape == (9, 128)
    assert torch.equal(torch.cat([y.flatten() for _, y in batches]), data.valid[1:322689])
    x, y = (t[:2, :16] for t in batches[0])
    old_result = json.loads(Path("results/wikitext_screen_v1/result.json").read_text())
    archive_records = []
    critical = (
        "src/core/data.py",
        "src/core/optimization.py",
        "src/dense_ffn/__init__.py",
        "src/blockshuffle_ffn/__init__.py",
        "src/grouped_ffn/__init__.py",
    )
    for trial in old_result["trials"]:
        path = Path("results/runs") / trial["run"]
        raw = json.loads((path / "metrics.json").read_text())
        assert sha256(path / "metrics.json") == trial["metrics_sha256"]
        assert raw["data"]["files"] == manifest["files"]
        with zipfile.ZipFile(path / "source.zip") as archive:
            import hashlib

            assert all(
                hashlib.sha256(archive.read(name)).hexdigest() == value
                for name, value in raw["provenance"]["source_files"].items()
            )
            assert all(archive.read(name) == Path(name).read_bytes() for name in critical)

            def training_loop(source):
                tree = ast.parse(source)
                loops = [
                    n
                    for n in ast.walk(tree)
                    if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Name)
                    and n.target.id == "step"
                ]
                assert len(loops) == 1
                return ast.dump(loops[0], include_attributes=False)

            assert training_loop(archive.read("src/core/trainer.py")) == training_loop(
                Path("src/core/trainer.py").read_bytes()
            )
        checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert checkpoint["step"] == 200
        assert sum(t.numel() for t in checkpoint["model"].values()) == raw["total_parameters"]
        archive_records.append(
            {
                "run": path.name,
                "metrics_sha256": sha256(path / "metrics.json"),
                "checkpoint_sha256": sha256(path / "checkpoint.pt"),
                "all_source_hashes_match": True,
            }
        )
        del checkpoint
    records = {}
    for base in ("calibrated_narrow", "blockshuffle"):
        mc, tc = configuration(base, RATES[1])
        original = Transformer(mc, 17).cuda()
        initialize_dense_width(original, tc)
        set_execution(original, tc)
        with autocast("cuda", "bf16"):
            baseline_logits, baseline_loss = original(x), original.loss(x, y)
        baseline_loss.backward()
        bp = dict(original.named_parameters())
        original_groups = {
            name: (g["weight_decay"], g["lr_scale"])
            for g in parameter_groups(original, tc)
            for name, p in bp.items()
            if any(p is member for member in g["params"])
        }
        for family in families:
            recipe = f"{base}_{family}"
            mc, tc = configuration(recipe, RATES[1])
            mc.validate()
            tc.validate()
            adaptive = Transformer(mc, 17).cuda()
            initialize_dense_width(adaptive, tc)
            set_execution(adaptive, tc)
            ap = dict(adaptive.named_parameters())
            assert all(torch.equal(p, ap[name]) for name, p in bp.items())
            with autocast("cuda", "bf16"):
                logits, loss = adaptive(x), adaptive.loss(x, y)
            assert torch.equal(logits, baseline_logits) and torch.equal(loss, baseline_loss)
            loss.backward()
            assert all(torch.equal(p.grad, ap[name].grad) for name, p in bp.items())
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ap.values())
            groups = parameter_groups(adaptive, tc)
            ag = {
                name: (g["weight_decay"], g["lr_scale"])
                for g in groups
                for name, p in ap.items()
                if any(p is member for member in g["params"])
            }
            assert all(ag[name] == value for name, value in original_groups.items())
            assert all(ag[name] == (0.0, 1.0) for name in ap if name not in bp)
            records[recipe] = {
                "model": asdict(mc),
                "training": asdict(tc),
                "actual_parameters": sum(p.numel() for p in adaptive.parameters()),
                "real_token_cuda_bf16_initial_logits_loss_common_gradients_exact": True,
                "all_shape_gradients_finite": True,
                "common_optimizer_assignments_identical": True,
                "optimizer_groups": group_summary(groups),
            }
            del adaptive, ap, groups, logits, loss
        del original, bp, baseline_logits, baseline_loss
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "environment": environment(),
        "provenance": provenance(),
        "old_variants_exact": original_variant_count,
        "before_snapshot_sha256": sha256(snapshot),
        "reused_controls": archive_records,
        "unchanged_projection_data_optimizer_sources": list(critical),
        "cache_hashes": manifest["files"],
        "validation_targets": 322688,
        "validation_target_stream_exact": True,
        "final_batch_shape": [9, 128],
        "recipes": records,
    }


def audit(output, suite="curves"):
    recipes = AFFINE_RECIPES if suite == "affine" else RECIPES
    plan = Path("research/affine_activation_plan.md") if suite == "affine" else PLAN
    output.mkdir(parents=True, exist_ok=False)
    try:
        initial = (
            preflight(Path("results/verification/affine_before_v1.pt"), ("affine",))
            if suite == "affine"
            else preflight()
        )
        write_json(output / "preflight.json", initial)
        protocol = {
            "plan_sha256": sha256(plan),
            "suite": suite,
            "provenance": provenance(),
            "environment": initial["environment"],
            "rates": RATES,
            "recipes": recipes,
            "steps": 200,
            "seed": 17,
            "train_tokens_per_trial": 409600,
            "validation_targets": 322688,
            "worker_timeout_seconds": 900,
            "controls": {k: v["run"] for k, v in controls().items()},
        }
        write_json(output / "protocol.json", protocol)
        with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in protocol["provenance"]["source_files"]:
                archive.write(name, name)
            archive.write(plan, plan.as_posix())
        rows = {k: [] for k in recipes}
        completed = []
        for index, rate in enumerate(RATES):
            offset = index % len(recipes)
            for recipe in recipes[offset:] + recipes[:offset]:
                path = output_path(recipe, rate)
                if path.exists():
                    raise ValueError(f"Existing run must not be overwritten: {path}")
                print(f"Starting {recipe}, peak LR {rate:g}", flush=True)
                with (output / f"{path.name}.log").open("x", encoding="utf-8") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "-X",
                            "faulthandler",
                            "-m",
                            "src.core.activation_screen",
                            "worker",
                            "--recipe",
                            recipe,
                            "--rate",
                            str(rate),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=900,
                    )
                metrics = json.loads((path / "metrics.json").read_text())
                assert (
                    metrics["training_tokens"] == 409600 and metrics["validation_tokens"] == 322688
                )
                assert metrics["data"]["files"] == initial["cache_hashes"]
                assert math.isfinite(metrics["validation_loss"]) and math.isfinite(
                    metrics["final_gradient_norm"]
                )
                for stage in ("initial", "final"):
                    diag = json.loads((path / f"{stage}_diagnostics.json").read_text())
                    assert all(r["finite"] and r.get("slope_finite", True) for r in diag.values())
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
        refs = controls()
        gates = {k: promotion(v, k.rsplit("_", 1)[0], refs) for k, v in selected.items()}
        result = {
            "status": "complete",
            "protocol": protocol,
            "trials": completed,
            "selected": {k: v["run"] for k, v in selected.items()},
            "promotion_gates": gates,
            "all_trial_diagnostics_finite": True,
            "earns_longer_training": {k: all(v.values()) for k, v in gates.items()},
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
            output / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()}
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "audit", "preflight"))
    parser.add_argument("--recipe", choices=RECIPES + AFFINE_RECIPES)
    parser.add_argument("--suite", choices=("curves", "affine"), default="curves")
    parser.add_argument("--rate", type=float, choices=RATES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        if args.recipe is None or args.rate is None:
            parser.error("worker requires recipe and rate")
        mc, tc = configuration(args.recipe, args.rate)
        train(mc, tc, CACHE, output_path(args.recipe, args.rate))
    elif args.command == "preflight":
        result = preflight()
        if args.output:
            write_json(args.output, result)
        print(json.dumps({k: v for k, v in result.items() if k != "provenance"}), flush=True)
    else:
        if args.output is None:
            parser.error("audit requires an output directory")
        audit(args.output, args.suite)
