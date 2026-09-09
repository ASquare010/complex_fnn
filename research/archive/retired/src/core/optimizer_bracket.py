"""Frozen H048 global learning-rate extension for four proven 800-step recipes."""

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

from src.core.affine_longer import configuration as original_configuration
from src.core.affine_longer import training_loop
from src.core.benchmark import evaluate, forward_flops
from src.core.config import ModelConfig, TrainConfig
from src.core.data import TokenData, load_manifest
from src.core.multihead_screen import verify_archive
from src.core.optimization import group_summary, initialize_dense_width, parameter_groups
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.trainer import learning_rate, train
from src.core.transformer import Transformer
from src.core.wikitext_screen import promotion, select_best

PLAN = Path("research/optimizer_bracket_plan.md")
CACHE = Path("data/wikitext2_v1")
RECIPES = ("full_swiglu", "calibrated_narrow", "blockshuffle", "full_gelu")
RATES = (0.0012, 0.0024, 0.0048)
NEW_RATES = RATES[1:]
COMPUTATION = (
    "src/core/config.py",
    "src/core/transformer.py",
    "src/core/trainer.py",
    "src/core/optimization.py",
    "src/core/data.py",
    "src/core/benchmark.py",
    "src/core/diagnostics.py",
    "src/dense_ffn/__init__.py",
    "src/blockshuffle_ffn/__init__.py",
    "src/grouped_ffn/__init__.py",
    "src/learnable_activation_ffn/__init__.py",
)
CRITICAL = (
    *COMPUTATION,
    "src/core/optimizer_bracket.py",
    "src/core/affine_longer.py",
    *(f"configs/wikitext2_{r}_800.json" for r in RECIPES),
)


def configuration(recipe, rate):
    if recipe not in RECIPES or rate not in RATES:
        raise ValueError("Recipe and rate must belong to the frozen grid")
    mc, tc = original_configuration(recipe, 17)
    return mc, replace(tc, learning_rate=rate)


def output_path(recipe, rate):
    return Path(f"results/runs/wikitext2_{recipe}_lr{round(rate * 1e6)}_s17_800")


def reference_record(recipe):
    root = "affine_rate_control_v1" if recipe == "blockshuffle" else "affine_longer_v2"
    old = json.loads((Path("results") / root / "result.json").read_text())
    assert old["status"] == "complete"
    row = next(t for t in old["trials"] if t["recipe"] == recipe)
    assert row["run"] == output_path(recipe, RATES[0]).name
    p = output_path(recipe, RATES[0])
    assert sha256(p / "metrics.json") == row["metrics_sha256"]
    assert sha256(p / "checkpoint.pt") == row["checkpoint_sha256"]
    return {
        "run": p.name,
        "metrics_sha256": row["metrics_sha256"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "source_archive_sha256": sha256(p / "source.zip"),
        "source_decision": root,
    }


def selected_decision(matrix):
    selected = {r: select_best(cells) for r, cells in matrix.items()}
    gates = promotion(selected)
    return {
        "selected": {r: m["run"] for r, m in selected.items()},
        "gates": gates,
        "quality_passes": all(v for k, v in gates.items() if not k.startswith("memory_")),
        "full_promotion_passes": all(gates.values()),
        "narrow_margin_at_least_point_two_percent": selected["blockshuffle"]["validation_loss"]
        <= 0.998 * selected["calibrated_narrow"]["validation_loss"],
        "selected_relative_nll_percent": {
            r: 100
            * (selected["blockshuffle"]["validation_loss"] / selected[r]["validation_loss"] - 1)
            for r in RECIPES
            if r != "blockshuffle"
        },
        "grid_positions": {
            r: (
                "lower_boundary"
                if m["training"]["learning_rate"] == RATES[0]
                else "upper_boundary"
                if m["training"]["learning_rate"] == RATES[-1]
                else "interior"
            )
            for r, m in selected.items()
        },
    }


def same_rate_comparisons(matrix):
    results = {}
    for rate in RATES:
        cells = {
            r: next(
                (
                    m
                    for m in values
                    if m["training"]["learning_rate"] == rate and m.get("status") != "FAILED"
                ),
                None,
            )
            for r, values in matrix.items()
        }
        candidate = cells["blockshuffle"]
        results[str(round(rate * 1e6))] = {
            r: 100 * (candidate["validation_loss"] / m["validation_loss"] - 1)
            if candidate is not None and m is not None
            else None
            for r, m in cells.items()
            if r != "blockshuffle"
        }
    return results


def verify_run(recipe, rate, initial):
    p = output_path(recipe, rate)
    m = json.loads((p / "metrics.json").read_text())
    mc, tc = configuration(recipe, rate)
    ref = initial["references"][recipe]
    rp = output_path(recipe, RATES[0])
    assert sha256(rp / "metrics.json") == ref["metrics_sha256"]
    reference = json.loads((rp / "metrics.json").read_text())
    assert ModelConfig(**m["model"]) == mc and TrainConfig(**m["training"]) == tc
    assert m["training_tokens"] == 1638400 and m["validation_tokens"] == 322688
    assert (
        m["data"]["files"] == initial["data_hashes"] and m["environment"] == initial["environment"]
    )
    assert abs(m["initial_validation_loss"] - reference["initial_validation_loss"]) < 1e-7
    assert m["optimizer_parameter_groups"] == reference["optimizer_parameter_groups"]
    assert m["ffn_width_initialization"] == reference["ffn_width_initialization"]
    assert m["activation_compilation"] is None and m["precision"] == "bf16"
    assert (
        m["ffn_parameters"] == mc.unique_ffn_parameters
        and m["total_parameters"] == mc.total_parameters
    )
    assert all(m[k] == v for k, v in forward_flops(mc).items())
    history = [json.loads(s) for s in (p / "history.jsonl").read_text().splitlines()]
    assert [r["step"] for r in history] == [1, 200, 400, 600, 800]
    assert history[-1]["validation_loss"] == m["validation_loss"]
    for row in history:
        assert all(
            math.isfinite(row[k])
            for k in ("training_loss", "validation_loss", "gradient_norm_pre_clip")
        )
        assert all(math.isfinite(v) for v in row["layer_gradient_norms_post_clip"].values())
        assert abs(row["learning_rate"] - learning_rate(row["step"] - 1, tc)) < 1e-15
    for stage in ("initial", "final"):
        d = json.loads((p / f"{stage}_diagnostics.json").read_text())
        assert all(r["finite"] for r in d.values())
    verify_archive(p, m)
    ckpt = torch.load(p / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert (
        ckpt["step"] == 800
        and ckpt["model_config"] == m["model"]
        and ckpt["training_config"] == m["training"]
    )
    assert sum(t.numel() for t in ckpt["model"].values()) == mc.total_parameters
    assert (
        hashlib.sha256(ckpt["sampling_rng"].numpy().tobytes()).hexdigest()
        == initial["sampling_rng_sha256"]
    )
    assert all(torch.isfinite(t).all() for t in ckpt["model"].values())
    return m


def preserved_functions(archive):
    for n in (
        "src/core/trainer.py",
        "src/core/data.py",
        "src/core/optimization.py",
        "src/dense_ffn/__init__.py",
        "src/blockshuffle_ffn/__init__.py",
        "src/grouped_ffn/__init__.py",
    ):
        assert archive.read(n) == Path(n).read_bytes(), n
    assert training_loop(archive.read("src/core/trainer.py")) == training_loop(
        Path("src/core/trainer.py").read_bytes()
    )
    for file, excluded in (
        ("src/core/benchmark.py", {"forward_flops"}),
        ("src/core/diagnostics.py", {"inspect_layers"}),
    ):
        functions = []
        for source in (archive.read(file), Path(file).read_bytes()):
            functions.append(
                {
                    n.name: ast.dump(n, include_attributes=False)
                    for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name not in excluded
                }
            )
        assert functions[0] == functions[1], file
    classes = []
    file = "src/core/transformer.py"
    for source in (archive.read(file), Path(file).read_bytes()):
        classes.append({n.name: n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)})
    for name in ("RMSNorm", "Attention"):
        assert ast.dump(classes[0][name], include_attributes=False) == ast.dump(
            classes[1][name], include_attributes=False
        )
    for name in ("Block", "Transformer"):
        methods = []
        for cls in classes:
            methods.append(
                {
                    n.name: ast.dump(n, include_attributes=False)
                    for n in cls[name].body
                    if isinstance(n, ast.FunctionDef) and n.name in ("forward", "loss")
                }
            )
        assert methods[0] == methods[1]


def preflight():
    torch.set_num_threads(4)
    tested = json.loads(Path("results/verification/headwise_tests_v1.json").read_text())
    final = json.loads(Path("results/verification/headwise_final_v1.json").read_text())
    assert (
        tested["returncode"] == 0
        and tested["source_unchanged_through_tests"]
        and final["status"] == "PASS"
    )
    assert all(sha256(Path(n)) == tested["source_files"][n] for n in COMPUTATION)
    snapshot = Path("results/verification/headwise_before_v1.pt")
    meta = json.loads(snapshot.with_suffix(".json").read_text())
    assert sha256(snapshot) == meta["sha256"]
    before = torch.load(snapshot, map_location="cpu", weights_only=True)
    for row in before.values():
        c = ModelConfig(**row["config"])
        m = Transformer(c, 17)
        x = torch.arange(16).reshape(2, 8)
        loss = m.loss(x, x + 1)
        loss.backward()
        assert torch.equal(m(x), row["logits"]) and torch.equal(loss, row["loss"])
        assert all(torch.equal(t, row["state"][n]) for n, t in m.state_dict().items())
        assert all(torch.equal(p.grad, row["gradients"][n]) for n, p in m.named_parameters())
        assert forward_flops(c) == row["flops"]
    assert len(before) == 30
    del before, m
    manifest = load_manifest(CACHE)
    assert all(sha256(CACHE / n) == h for n, h in manifest["files"].items())
    data = TokenData(CACHE, "cpu", 10017)
    batches = list(data.validation(16, 128, 158))
    assert len(batches) == 158 and batches[-1][0].shape == (9, 128)
    assert torch.equal(torch.cat([y.flatten() for _, y in batches]), data.valid[1:322689])
    records = {r: reference_record(r) for r in RECIPES}
    ckpt = torch.load(
        output_path(RECIPES[0], RATES[0]) / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    sampling_hash = hashlib.sha256(ckpt["sampling_rng"].numpy().tobytes()).hexdigest()
    del ckpt
    initial = {
        "references": records,
        "sampling_rng_sha256": sampling_hash,
        "data_hashes": manifest["files"],
        "environment": environment(),
        "old_variants_exact": 30,
        "old_snapshot_sha256": sha256(snapshot),
        "current_computation_matches_220_test_snapshot": True,
        "validation_stream_exact": True,
    }
    shared = None
    qualifications = {}
    for r in RECIPES:
        p = output_path(r, RATES[0])
        m = verify_run(r, RATES[0], initial)
        changed = {n for n in COMPUTATION if sha256(Path(n)) != m["provenance"]["source_files"][n]}
        assert changed <= {
            "src/core/config.py",
            "src/core/transformer.py",
            "src/core/benchmark.py",
            "src/core/diagnostics.py",
        }
        with zipfile.ZipFile(p / "source.zip") as archive:
            preserved_functions(archive)
        mc, tc = configuration(r, RATES[0])
        model = Transformer(mc, 17)
        init = initialize_dense_width(model, tc)
        assert init == m["ffn_width_initialization"]
        assert group_summary(parameter_groups(model, tc)) == m["optimizer_parameter_groups"]
        state = {n: t.detach().clone() for n, t in model.named_parameters() if ".ffn." not in n}
        if shared is None:
            shared = state
        else:
            assert all(torch.equal(t, shared[n]) for n, t in state.items())
        qualifications[r] = {
            "historical_changed_computation_files": sorted(changed),
            "literal_sources_and_execution_asts_match": True,
            "optimizer_and_count_metadata_match": True,
        }
        del model
    initial["source_qualifications"] = qualifications
    initial["historical_branch_qualification"] = (
        "Registry, factory/initialization, matrix counting and diagnostic additions for other variants; all 30 prior tiny CPU functions/gradients exact, four large initial functions checked separately on GPU. Trainer/data/optimizer/base FFNs remain literal matches."
    )
    initial["common_non_ffn_initial_parameters_exact"] = True
    return initial


def initial_worker(output):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = TokenData(CACHE, "cuda", 10017)
    rows = []
    for r in RECIPES:
        mc, tc = configuration(r, RATES[0])
        m = Transformer(mc, 17).cuda()
        initialize_dense_width(m, tc)
        if tc.recompute_gate:
            for b in m.blocks:
                b.ffn.recompute_gate = True
                b.ffn.gate_recompute_method = tc.gate_recompute_method
        loss, count = evaluate(m, data.validation(16, 128, 158), "cuda", "bf16")
        original = json.loads((output_path(r, RATES[0]) / "metrics.json").read_text())[
            "initial_validation_loss"
        ]
        assert count == 322688 and abs(loss - original) < 1e-7
        rows.append(
            {"recipe": r, "initial_nll": loss, "reference_initial_nll": original, "targets": count}
        )
        del m
        gc.collect()
        torch.cuda.empty_cache()
    write_json(
        output,
        {
            "status": "PASS",
            "rows": rows,
            "provenance": provenance(),
            "environment": environment(),
            "data_hashes": data.manifest["files"],
        },
    )
    print(json.dumps(rows), flush=True)


def numerical_failure(path):
    """Only a trainer-recorded nonfinite clip norm qualifies; import exits do not."""
    failure = path / "failure.json"
    if not failure.exists() or (path / "metrics.json").exists():
        return False
    row = json.loads(failure.read_text())
    return (
        row.get("status") == "FAILED"
        and "non-finite" in row.get("error", "")
        and "clip_grad_norm_" in row.get("traceback", "")
    )


def run_worker(output, name, command, timeout, attempt):
    with (output / f"{name}.attempt_{attempt}.log").open("x", encoding="utf-8") as log:
        return subprocess.run(
            [sys.executable, "-X", "faulthandler", *command],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        ).returncode


def audit(output, resume_qualification=None):
    completed = []
    attempt = 1
    if resume_qualification is None:
        output.mkdir(parents=True, exist_ok=False)
    else:
        assert output.is_dir() and not (output / "result.json").exists()
        qualification = json.loads(resume_qualification.read_text())
        assert qualification["plan_sha256"] == sha256(PLAN)
        assert qualification["protocol_sha256"] == sha256(output / "protocol.json")
        assert qualification["failure_sha256"] == sha256(output / "failure.json")
        assert qualification["no_live_worker_verified"] and qualification["reason"]
        prior = json.loads((output / "failure.json").read_text())
        completed = prior["completed"]
        attempt = prior["attempt"] + 1
        write_json(output / f"resume_{attempt}.json", qualification)
    try:
        if resume_qualification is None:
            initial = preflight()
            write_json(output / "preflight.json", initial)
            protocol = {
                "plan_sha256": sha256(PLAN),
                "provenance": provenance(),
                "environment": environment(),
                "rates": RATES,
                "new_rates": NEW_RATES,
                "recipes": RECIPES,
                "max_new_training_tokens": 13107200,
                "worker_timeout_seconds": 2400,
                "reference_records": initial["references"],
                "configurations": {
                    r: {
                        str(round(rate * 1e6)): {
                            "model": asdict(configuration(r, rate)[0]),
                            "training": asdict(configuration(r, rate)[1]),
                        }
                        for rate in RATES
                    }
                    for r in RECIPES
                },
            }
            write_json(output / "protocol.json", protocol)
            with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for n in protocol["provenance"]["source_files"]:
                    archive.write(n, n)
                archive.write(PLAN, PLAN.as_posix())
        else:
            initial = json.loads((output / "preflight.json").read_text())
            protocol = json.loads((output / "protocol.json").read_text())
            assert protocol["plan_sha256"] == sha256(PLAN)
            assert all(
                sha256(Path(n)) == protocol["provenance"]["source_files"][n] for n in CRITICAL
            )
            assert initial["environment"] == environment()
            assert all(sha256(CACHE / n) == h for n, h in initial["data_hashes"].items())
        if not (output / "baseline_initial.json").exists():
            print("Reproducing all four 800-step reference initial validation losses", flush=True)
            write_json(
                output / "progress.json",
                {"status": "baseline_initial", "attempt": attempt, "completed": completed},
            )
            code = run_worker(
                output,
                "baseline_initial",
                [
                    "-m",
                    "src.core.optimizer_bracket",
                    "baseline-initial",
                    "--output",
                    str(output / "baseline_initial.json"),
                ],
                600,
                attempt,
            )
            assert code == 0, f"Initial worker failed with returncode {code}"
        baseline = json.loads((output / "baseline_initial.json").read_text())
        assert baseline["status"] == "PASS" and all(
            baseline["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
            for n in CRITICAL
        )
        matrix = {r: [verify_run(r, RATES[0], initial)] for r in RECIPES}
        for index, rate in enumerate(NEW_RATES):
            for r in RECIPES if index == 0 else RECIPES[::-1]:
                p = output_path(r, rate)
                retained = next((t for t in completed if t["run"] == p.name), None)
                if retained is not None:
                    if retained["status"] == "NUMERICAL_FAIL":
                        assert (
                            numerical_failure(p)
                            and sha256(p / "failure.json") == retained["failure_sha256"]
                        )
                        continue
                    assert (
                        sha256(p / "metrics.json") == retained["metrics_sha256"]
                        and sha256(p / "checkpoint.pt") == retained["checkpoint_sha256"]
                    )
                    m = verify_run(r, rate, initial)
                else:
                    assert not p.exists(), f"Unqualified existing run directory: {p}"
                    write_json(
                        output / "progress.json",
                        {
                            "status": "running",
                            "attempt": attempt,
                            "active": p.name,
                            "completed": completed,
                        },
                    )
                    print(f"Starting {r}, peak LR {rate}, 800 steps", flush=True)
                    code = run_worker(
                        output,
                        p.name,
                        [
                            "-m",
                            "src.core.optimizer_bracket",
                            "worker",
                            "--recipe",
                            r,
                            "--rate",
                            str(rate),
                        ],
                        2400,
                        attempt,
                    )
                    if code != 0:
                        if not numerical_failure(p):
                            raise RuntimeError(
                                f"Unclassified worker failure for {p.name}; returncode {code}"
                            )
                        failure = json.loads((p / "failure.json").read_text())
                        assert (
                            ModelConfig(**failure["model"]) == configuration(r, rate)[0]
                            and TrainConfig(**failure["training"]) == configuration(r, rate)[1]
                        )
                        assert all(
                            failure["provenance"]["source_files"][n]
                            == protocol["provenance"]["source_files"][n]
                            for n in CRITICAL
                        )
                        verify_archive(p, failure)
                        completed.append(
                            {
                                "recipe": r,
                                "rate": rate,
                                "run": p.name,
                                "status": "NUMERICAL_FAIL",
                                "failure_sha256": sha256(p / "failure.json"),
                            }
                        )
                        print(json.dumps(completed[-1]), flush=True)
                        continue
                    m = verify_run(r, rate, initial)
                    completed.append(
                        {
                            "recipe": r,
                            "rate": rate,
                            "run": p.name,
                            "status": "COMPLETE",
                            "nll": m["validation_loss"],
                            "peak_mib": m["peak_allocated_vram_bytes"] / 2**20,
                            "metrics_sha256": sha256(p / "metrics.json"),
                            "checkpoint_sha256": sha256(p / "checkpoint.pt"),
                        }
                    )
                    print(json.dumps(completed[-1]), flush=True)
                assert all(
                    m["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
                    for n in CRITICAL
                )
                matrix[r].append(m)
        assert len(completed) == 8
        result = {
            "status": "complete",
            "trials": completed,
            "decision": selected_decision(matrix),
            "same_rate_relative_nll_percent": same_rate_comparisons(matrix),
            "plan_sha256": sha256(PLAN),
            "all_complete_trials_sampling_matches_references": True,
            "complete_trials_diagnostics_finite": True,
            "numerical_failures": sum(t["status"] == "NUMERICAL_FAIL" for t in completed),
        }
        write_json(output / "result.json", result)
        write_json(
            output / "progress.json",
            {"status": "complete", "attempt": attempt, "completed": completed},
        )
        print(json.dumps(result), flush=True)
    except Exception as exc:
        failure = {
            "status": "failed",
            "attempt": attempt,
            "completed": completed,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output / f"failure_attempt_{attempt}.json", failure)
        write_json(output / "failure.json", failure)
        write_json(output / "progress.json", failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "baseline-initial", "worker", "audit"))
    parser.add_argument("--recipe", choices=RECIPES)
    parser.add_argument("--rate", type=float, choices=NEW_RATES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-qualification", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        if args.recipe is None or args.rate is None:
            parser.error("worker requires recipe and rate")
        train(*configuration(args.recipe, args.rate), CACHE, output_path(args.recipe, args.rate))
    elif args.output is None:
        parser.error("output is required")
    elif args.command == "baseline-initial":
        assert not args.output.exists()
        initial_worker(args.output)
    elif args.command == "preflight":
        assert not args.output.exists()
        write_json(args.output, preflight())
    else:
        audit(args.output, args.resume_qualification)
