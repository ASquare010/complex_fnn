"""H049: fill three lower-rate cells without rerunning thirteen retained controls."""

import argparse
import hashlib
import json
import math
import traceback
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import torch

from src.core.affine_longer import configuration as original_configuration
from src.core.benchmark import forward_flops
from src.core.config import ModelConfig, TrainConfig
from src.core.data import TokenData
from src.core.multihead_screen import verify_archive
from src.core.optimizer_bracket import (
    CACHE,
    RECIPES,
    numerical_failure,
    output_path,
    preserved_functions,
    run_worker,
)
from src.core.optimizer_bracket import CRITICAL as PRIOR_CRITICAL
from src.core.optimizer_bracket import same_rate_comparisons as prior_same_rate
from src.core.optimizer_bracket_report import load_verified
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.trainer import learning_rate, train
from src.core.wikitext_screen import promotion, select_best

PLAN = Path("research/optimizer_lower_plan.md")
RATES = (0.0006, 0.0012, 0.0024, 0.0048)
NEW_RECIPES = ("full_swiglu", "calibrated_narrow", "full_gelu")
CRITICAL = (*PRIOR_CRITICAL, "src/core/optimizer_lower.py")


def configuration(recipe):
    if recipe not in RECIPES:
        raise ValueError("Unknown frozen recipe")
    mc, tc = original_configuration(recipe, 17)
    return mc, replace(tc, learning_rate=RATES[0])


def verify_lower(recipe, initial):
    p = output_path(recipe, RATES[0])
    m = json.loads((p / "metrics.json").read_text())
    mc, tc = configuration(recipe)
    ref = output_path(recipe, 0.0012)
    assert sha256(ref / "metrics.json") == initial["references"][recipe]["metrics_sha256"]
    reference = json.loads((ref / "metrics.json").read_text())
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
    h = [json.loads(s) for s in (p / "history.jsonl").read_text().splitlines()]
    assert [r["step"] for r in h] == [1, 200, 400, 600, 800]
    assert h[-1]["validation_loss"] == m["validation_loss"]
    for row in h:
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
    assert all(torch.isfinite(t).all() for t in ckpt["model"].values())
    assert (
        hashlib.sha256(ckpt["sampling_rng"].numpy().tobytes()).hexdigest()
        == initial["sampling_rng_sha256"]
    )
    return m


def decisions(matrix):
    selected = {r: select_best(v) for r, v in matrix.items()}
    gates = promotion(selected)
    return {
        "selected": {r: m["run"] for r, m in selected.items()},
        "gates": gates,
        "quality_passes": all(v for k, v in gates.items() if not k.startswith("memory_")),
        "full_promotion_passes": all(gates.values()),
        "narrow_margin_at_least_point_two_percent": selected["blockshuffle"]["validation_loss"]
        <= 0.998 * selected["calibrated_narrow"]["validation_loss"],
        "selected_relative_nll_percent": {
            r: 100 * (selected["blockshuffle"]["validation_loss"] / m["validation_loss"] - 1)
            for r, m in selected.items()
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
        "all_selected_rates_unchanged_at_0012": all(
            m["training"]["learning_rate"] == 0.0012 for m in selected.values()
        ),
    }


def same_rates(matrix):
    cells = {
        r: next((m for m in v if m["training"]["learning_rate"] == RATES[0]), None)
        for r, v in matrix.items()
    }
    candidate = cells["blockshuffle"]
    lower = {
        r: 100 * (candidate["validation_loss"] / m["validation_loss"] - 1)
        if m is not None
        else None
        for r, m in cells.items()
        if r != "blockshuffle"
    }
    return {"600": lower, **prior_same_rate(matrix)}


def preflight():
    torch.set_num_threads(4)
    tested = json.loads(Path("results/verification/optimizer_bracket_tests_v1.json").read_text())
    final = json.loads(Path("results/verification/optimizer_bracket_final_v1.json").read_text())
    assert (
        tested["returncode"] == 0
        and tested["source_unchanged_through_tests"]
        and final["status"] == "PASS"
    )
    assert all(sha256(Path(n)) == tested["source_files"][n] for n in PRIOR_CRITICAL)
    prior, initial, protocol, matrix = load_verified()
    assert prior["decision"]["full_promotion_passes"]
    assert all(v == "lower_boundary" for v in prior["decision"]["grid_positions"].values())
    assert all(sha256(Path(n)) == protocol["provenance"]["source_files"][n] for n in PRIOR_CRITICAL)
    assert initial["environment"] == environment()
    assert all(sha256(CACHE / n) == h for n, h in initial["data_hashes"].items())
    data = TokenData(CACHE, "cpu", 10017)
    batches = list(data.validation(16, 128, 158))
    assert len(batches) == 158 and batches[-1][0].shape == (9, 128)
    assert torch.equal(torch.cat([y.flatten() for _, y in batches]), data.valid[1:322689])
    old = json.loads(Path("results/affine_longer_v2/result.json").read_text())
    retained = next(t for t in old["trials"] if t["recipe"] == "blockshuffle")
    p = output_path("blockshuffle", RATES[0])
    assert retained["run"] == p.name
    assert sha256(p / "metrics.json") == retained["metrics_sha256"]
    assert sha256(p / "checkpoint.pt") == retained["checkpoint_sha256"]
    m = verify_lower("blockshuffle", initial)
    with zipfile.ZipFile(p / "source.zip") as archive:
        preserved_functions(archive)
    matrix["blockshuffle"].append(m)
    return {
        "prior_initial": initial,
        "prior_result_sha256": sha256(Path("results/optimizer_bracket_v1/result.json")),
        "prior_protocol_sha256": sha256(Path("results/optimizer_bracket_v1/protocol.json")),
        "lower_blockshuffle_record": retained,
        "lower_blockshuffle_archive_sha256": sha256(p / "source.zip"),
        "current_computation_matches_223_test_snapshot": True,
        "validation_stream_exact": True,
        "retained_cells": 13,
    }, matrix


def audit(output, resume_qualification=None):
    completed = []
    attempt = 1
    if resume_qualification is None:
        output.mkdir(parents=True, exist_ok=False)
    else:
        assert output.is_dir() and not (output / "result.json").exists()
        q = json.loads(resume_qualification.read_text())
        assert q["plan_sha256"] == sha256(PLAN) and q["protocol_sha256"] == sha256(
            output / "protocol.json"
        )
        assert (
            q["failure_sha256"] == sha256(output / "failure.json")
            and q["no_live_worker_verified"]
            and q["reason"]
        )
        prior = json.loads((output / "failure.json").read_text())
        completed = prior["completed"]
        attempt = prior["attempt"] + 1
        write_json(output / f"resume_{attempt}.json", q)
    try:
        initial, matrix = preflight()
        if resume_qualification is None:
            write_json(output / "preflight.json", initial)
            protocol = {
                "plan_sha256": sha256(PLAN),
                "provenance": provenance(),
                "environment": environment(),
                "rates": RATES,
                "new_recipes": NEW_RECIPES,
                "new_training_tokens": 4915200,
                "worker_timeout_seconds": 2400,
                "configurations": {
                    r: {
                        "model": asdict(configuration(r)[0]),
                        "training": asdict(configuration(r)[1]),
                    }
                    for r in NEW_RECIPES
                },
            }
            write_json(output / "protocol.json", protocol)
            with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for n in protocol["provenance"]["source_files"]:
                    archive.write(n, n)
                archive.write(PLAN, PLAN.as_posix())
        else:
            protocol = json.loads((output / "protocol.json").read_text())
            assert protocol["plan_sha256"] == sha256(PLAN)
            assert all(
                sha256(Path(n)) == protocol["provenance"]["source_files"][n] for n in CRITICAL
            )
            assert initial == json.loads((output / "preflight.json").read_text())
        print(
            "Thirteen retained cells verified; lower-rate source/data preflight passed", flush=True
        )
        for r in NEW_RECIPES:
            p = output_path(r, RATES[0])
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
                m = verify_lower(r, initial["prior_initial"])
            else:
                assert not p.exists(), f"Unqualified existing run directory: {p}"
                print(f"Starting {r}, LR 0.0006, 800 steps", flush=True)
                write_json(
                    output / "progress.json",
                    {
                        "status": "running",
                        "attempt": attempt,
                        "active": p.name,
                        "completed": completed,
                    },
                )
                code = run_worker(
                    output,
                    p.name,
                    ["-m", "src.core.optimizer_lower", "worker", "--recipe", r],
                    2400,
                    attempt,
                )
                if code != 0:
                    if not numerical_failure(p):
                        raise RuntimeError(
                            f"Unclassified worker failure: {p.name}, returncode {code}"
                        )
                    failure = json.loads((p / "failure.json").read_text())
                    assert (
                        ModelConfig(**failure["model"]) == configuration(r)[0]
                        and TrainConfig(**failure["training"]) == configuration(r)[1]
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
                            "rate": RATES[0],
                            "run": p.name,
                            "status": "NUMERICAL_FAIL",
                            "failure_sha256": sha256(p / "failure.json"),
                        }
                    )
                    continue
                m = verify_lower(r, initial["prior_initial"])
                completed.append(
                    {
                        "recipe": r,
                        "rate": RATES[0],
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
        result = {
            "status": "complete",
            "trials": completed,
            "decision": decisions(matrix),
            "same_rate_relative_nll_percent": same_rates(matrix),
            "plan_sha256": sha256(PLAN),
            "numerical_failures": sum(t["status"] == "NUMERICAL_FAIL" for t in completed),
            "all_complete_sampling_states_match": True,
            "all_complete_diagnostics_finite": True,
        }
        write_json(output / "result.json", result)
        write_json(output / "progress.json", {"status": "complete", "completed": completed})
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
    parser.add_argument("command", choices=("preflight", "worker", "audit"))
    parser.add_argument("--recipe", choices=NEW_RECIPES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-qualification", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        if args.recipe is None:
            parser.error("worker requires recipe")
        train(*configuration(args.recipe), CACHE, output_path(args.recipe, RATES[0]))
    elif args.output is None:
        parser.error("output is required")
    elif args.command == "preflight":
        assert not args.output.exists()
        write_json(args.output, preflight()[0])
    else:
        audit(args.output, args.resume_qualification)
