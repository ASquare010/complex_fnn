"""Fill the missing same-rate WikiText comparisons without changing model code."""

import argparse
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

from src.core.affine_longer import CRITICAL, verify_trial
from src.core.affine_longer import configuration as original_configuration
from src.core.affine_longer import output_path as original_path
from src.core.config import ModelConfig, TrainConfig
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.trainer import train
from src.core.wikitext_screen import promotion

PLAN = Path("research/affine_activation_rate_control_plan.md")
CASES = ("blockshuffle", "blockshuffle_affine")
RATES = {"blockshuffle": 0.0012, "blockshuffle_affine": 0.0006}


def configuration(recipe):
    mc, tc = original_configuration(recipe)
    return mc, replace(tc, learning_rate=RATES[recipe])


def output_path(recipe):
    _, tc = configuration(recipe)
    return Path(f"results/runs/wikitext2_{recipe}_lr{round(tc.learning_rate * 1e6)}_s17_800")


def same_rate_comparisons(matrix):
    results = {}
    for rate, pair in matrix.items():
        base = pair["blockshuffle"]["validation_loss"]
        affine = pair["blockshuffle_affine"]["validation_loss"]
        assert math.isfinite(base) and math.isfinite(affine) and base > 0
        results[rate] = {
            "plain_nll": base,
            "affine_nll": affine,
            "affine_minus_plain_nll": affine - base,
            "relative_affine_change_percent": 100 * (affine / base - 1),
            "at_least_point_two_percent_better": affine <= 0.998 * base,
        }
    return results


def verify_new(recipe, initial, reference):
    path = output_path(recipe)
    m = json.loads((path / "metrics.json").read_text())
    mc, tc = configuration(recipe)
    assert ModelConfig(**m["model"]) == mc and TrainConfig(**m["training"]) == tc
    assert m["training_tokens"] == 1638400 and m["validation_tokens"] == 322688
    assert m["data"]["files"] == initial["cache_hashes"]
    assert m["environment"] == initial["environment"] == environment()
    assert abs(m["initial_validation_loss"] - reference["initial_validation_loss"]) < 1e-7
    assert m["optimizer_parameter_groups"] == reference["optimizer_parameter_groups"]
    assert m["ffn_width_initialization"] == reference["ffn_width_initialization"]
    assert m["activation_compilation"] is None
    assert (
        m["ffn_parameters"] == mc.unique_ffn_parameters
        and m["total_parameters"] == mc.total_parameters
    )
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
        diag = json.loads((path / f"{stage}_diagnostics.json").read_text())
        assert all(v["finite"] and v.get("slope_finite", True) for v in diag.values())
    with zipfile.ZipFile(path / "source.zip") as archive:
        assert all(
            hashlib.sha256(archive.read(n)).hexdigest() == h
            for n, h in m["provenance"]["source_files"].items()
        )
    checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert (
        checkpoint["step"] == 800
        and checkpoint["model_config"] == m["model"]
        and checkpoint["training_config"] == m["training"]
    )
    assert sum(t.numel() for t in checkpoint["model"].values()) == mc.total_parameters
    return m, checkpoint["sampling_rng"].clone()


def audit(output):
    output.mkdir(parents=True, exist_ok=False)
    completed = []
    try:
        initial = json.loads(Path("results/affine_longer_v2/preflight.json").read_text())
        previous = json.loads(Path("results/affine_longer_v2/protocol.json").read_text())
        assert all(sha256(Path(n)) == previous["provenance"]["source_files"][n] for n in CRITICAL)
        assert json.loads(Path("results/affine_replication_v1/result.json").read_text())[
            "replication_passes"
        ]
        references, sampling = {}, None
        matrix = {"600": {}, "1200": {}}
        for recipe in ("full_swiglu", "full_gelu", "calibrated_narrow", *CASES):
            m, rng = verify_trial(recipe, initial)
            references[recipe] = m
            if sampling is None:
                sampling = rng
            assert torch.equal(sampling, rng)
            if recipe in CASES:
                rate = str(round(m["training"]["learning_rate"] * 1e6))
                matrix[rate][recipe] = m
                assert not output_path(recipe).exists(), f"Do not overwrite {output_path(recipe)}"
        protocol = {
            "plan_sha256": sha256(PLAN),
            "provenance": provenance(),
            "environment": environment(),
            "new_training_tokens": 3276800,
            "worker_timeout_seconds": 2400,
            "configurations": {
                r: {"model": asdict(configuration(r)[0]), "training": asdict(configuration(r)[1])}
                for r in CASES
            },
            "source_hashes": {
                r: {
                    "metrics": sha256(original_path(r) / "metrics.json"),
                    "checkpoint": sha256(original_path(r) / "checkpoint.pt"),
                }
                for r in references
            },
            "all_critical_computation_hashes_match_original": True,
        }
        write_json(output / "protocol.json", protocol)
        with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in protocol["provenance"]["source_files"]:
                archive.write(name, name)
            archive.write(PLAN, PLAN.as_posix())
        for recipe in CASES:
            print(f"Starting {recipe} at {RATES[recipe]}, seed17, 800 steps", flush=True)
            write_json(
                output / "progress.json",
                {"status": "running", "active": recipe, "completed": completed},
            )
            with (output / f"{recipe}.log").open("x", encoding="utf-8") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-X",
                        "faulthandler",
                        "-m",
                        "src.core.affine_rate_control",
                        "worker",
                        "--recipe",
                        recipe,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=2400,
                )
            m, rng = verify_new(recipe, initial, references[recipe])
            assert torch.equal(rng, sampling)
            assert all(
                m["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
                for n in (*CRITICAL, "src/core/affine_rate_control.py")
            )
            rate = str(round(RATES[recipe] * 1e6))
            matrix[rate][recipe] = m
            completed.append(
                {
                    "recipe": recipe,
                    "rate": RATES[recipe],
                    "run": output_path(recipe).name,
                    "nll": m["validation_loss"],
                    "metrics_sha256": sha256(output_path(recipe) / "metrics.json"),
                    "checkpoint_sha256": sha256(output_path(recipe) / "checkpoint.pt"),
                }
            )
            print(json.dumps(completed[-1]), flush=True)
        comparisons = same_rate_comparisons(matrix)
        quality = {
            rate: {r: promotion({**references, "blockshuffle": m}) for r, m in pair.items()}
            for rate, pair in matrix.items()
        }
        result = {
            "status": "complete",
            "trials": completed,
            "same_rate_comparisons": comparisons,
            "material_benefit_at_both_rates": all(
                v["at_least_point_two_percent_better"] for v in comparisons.values()
            ),
            "reference_gates": quality,
            "final_sampling_rng_matches_all_four": True,
            "final_sampling_rng_sha256": hashlib.sha256(sampling.numpy().tobytes()).hexdigest(),
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
    parser.add_argument("command", choices=("audit", "worker"))
    parser.add_argument("--recipe", choices=CASES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        if args.recipe is None:
            parser.error("worker requires recipe")
        mc, tc = configuration(args.recipe)
        train(mc, tc, Path("data/wikitext2_v1"), output_path(args.recipe))
    else:
        if args.output is None:
            parser.error("audit requires output")
        audit(args.output)
