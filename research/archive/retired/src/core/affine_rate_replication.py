"""Independent seeds for the stronger plain same-rate BlockShuffle control."""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import traceback
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import torch

from src.core.affine_longer import CRITICAL, verify_trial
from src.core.affine_longer import configuration as original_configuration
from src.core.affine_replication import paired_statistics
from src.core.config import ModelConfig, TrainConfig
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.trainer import train
from src.core.wikitext_screen import promotion

PLAN = Path("research/affine_rate_replication_plan.md")
SEEDS = (17, 29, 43)


def configuration(seed):
    mc, tc = original_configuration("blockshuffle", seed)
    return mc, replace(tc, learning_rate=0.0012)


def output_path(seed):
    return Path(f"results/runs/wikitext2_blockshuffle_lr1200_s{seed}_800")


def verify_plain(seed, initial, reference):
    path = output_path(seed)
    m = json.loads((path / "metrics.json").read_text())
    mc, tc = configuration(seed)
    assert ModelConfig(**m["model"]) == mc and TrainConfig(**m["training"]) == tc
    assert m["training_tokens"] == 1638400 and m["validation_tokens"] == 322688
    assert (
        m["data"]["files"] == initial["cache_hashes"] and m["environment"] == initial["environment"]
    )
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


def load_references(initial):
    rows, rngs = {}, {}
    for seed in SEEDS:
        rows[seed] = {}
        for recipe in (
            "full_swiglu",
            "full_gelu",
            "calibrated_narrow",
            "blockshuffle",
            "blockshuffle_affine",
        ):
            m, rng = verify_trial(recipe, initial, seed)
            rows[seed][recipe] = m
            if seed not in rngs:
                rngs[seed] = rng
            assert torch.equal(rngs[seed], rng)
    return rows, rngs


def summarize(rows, plain):
    pairs = [
        rows[s]["blockshuffle_affine"]["validation_loss"] - plain[s]["validation_loss"]
        for s in SEEDS
    ]
    affine_mean = statistics.mean(rows[s]["blockshuffle_affine"]["validation_loss"] for s in SEEDS)
    plain_mean = statistics.mean(plain[s]["validation_loss"] for s in SEEDS)
    gates = {s: promotion({**rows[s], "blockshuffle": plain[s]}) for s in SEEDS}
    return {
        "plain_nlls": {s: plain[s]["validation_loss"] for s in SEEDS},
        "affine_nlls": {s: rows[s]["blockshuffle_affine"]["validation_loss"] for s in SEEDS},
        "plain_mean_nll": plain_mean,
        "plain_sample_sd_nll": statistics.stdev(plain[s]["validation_loss"] for s in SEEDS),
        "affine_mean_nll": affine_mean,
        "affine_sample_sd_nll": statistics.stdev(
            rows[s]["blockshuffle_affine"]["validation_loss"] for s in SEEDS
        ),
        "relative_mean_affine_change_percent": 100 * (affine_mean / plain_mean - 1),
        "paired_statistics": paired_statistics(pairs),
        "plain_reference_gates": gates,
        "plain_passes_all_seeds": all(all(g.values()) for g in gates.values()),
        "material_activation_benefit": affine_mean <= 0.998 * plain_mean
        and all(d < 0 for d in pairs),
        "plain_relative_to_reference_means_percent": {
            r: 100
            * (plain_mean / statistics.mean(rows[s][r]["validation_loss"] for s in SEEDS) - 1)
            for r in ("full_swiglu", "full_gelu", "calibrated_narrow", "blockshuffle")
        },
    }


def audit(output):
    output.mkdir(parents=True, exist_ok=False)
    completed = []
    try:
        prior = json.loads(Path("results/affine_rate_control_v1/result.json").read_text())
        assert prior["status"] == "complete" and not prior["material_benefit_at_both_rates"]
        initial = json.loads(Path("results/affine_longer_v2/preflight.json").read_text())
        old = json.loads(Path("results/affine_longer_v2/protocol.json").read_text())
        assert all(sha256(Path(n)) == old["provenance"]["source_files"][n] for n in CRITICAL)
        assert initial["environment"] == environment()
        rows, rngs = load_references(initial)
        plain = {}
        plain[17], rng = verify_plain(17, initial, rows[17]["blockshuffle"])
        assert torch.equal(rng, rngs[17])
        source = next(r for r in prior["trials"] if r["recipe"] == "blockshuffle")
        assert sha256(output_path(17) / "metrics.json") == source["metrics_sha256"]
        assert sha256(output_path(17) / "checkpoint.pt") == source["checkpoint_sha256"]
        for seed in SEEDS[1:]:
            assert not output_path(seed).exists(), f"Do not overwrite {output_path(seed)}"
        protocol = {
            "plan_sha256": sha256(PLAN),
            "provenance": provenance(),
            "environment": environment(),
            "prior_result_sha256": sha256(Path("results/affine_rate_control_v1/result.json")),
            "seed17": source,
            "new_training_tokens": 3276800,
            "worker_timeout_seconds": 2400,
            "configurations": {
                s: {"model": asdict(configuration(s)[0]), "training": asdict(configuration(s)[1])}
                for s in SEEDS[1:]
            },
            "all_critical_computation_hashes_match_original": True,
        }
        write_json(output / "protocol.json", protocol)
        with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for n in protocol["provenance"]["source_files"]:
                archive.write(n, n)
            archive.write(PLAN, PLAN.as_posix())
        for seed in SEEDS[1:]:
            print(f"Starting plain BlockShuffle LR0.0012, seed{seed}, 800 steps", flush=True)
            write_json(
                output / "progress.json",
                {"status": "running", "active_seed": seed, "completed": completed},
            )
            with (output / f"s{seed}.log").open("x", encoding="utf-8") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-X",
                        "faulthandler",
                        "-m",
                        "src.core.affine_rate_replication",
                        "worker",
                        "--seed",
                        str(seed),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=2400,
                )
            m, rng = verify_plain(seed, initial, rows[seed]["blockshuffle"])
            assert torch.equal(rng, rngs[seed])
            assert all(
                m["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
                for n in (*CRITICAL, "src/core/affine_rate_replication.py")
            )
            plain[seed] = m
            completed.append(
                {
                    "seed": seed,
                    "run": output_path(seed).name,
                    "nll": m["validation_loss"],
                    "metrics_sha256": sha256(output_path(seed) / "metrics.json"),
                    "checkpoint_sha256": sha256(output_path(seed) / "checkpoint.pt"),
                }
            )
            print(json.dumps(completed[-1]), flush=True)
        result = {
            "status": "complete",
            "trials": completed,
            **summarize(rows, plain),
            "final_sampling_rng_matches_each_seed": True,
            "sampling_rng_hashes": {
                s: hashlib.sha256(r.numpy().tobytes()).hexdigest() for s, r in rngs.items()
            },
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
    parser.add_argument("command", choices=("worker", "audit"))
    parser.add_argument("--seed", type=int, choices=SEEDS[1:])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        if args.seed is None:
            parser.error("worker requires seed")
        mc, tc = configuration(args.seed)
        train(mc, tc, Path("data/wikitext2_v1"), output_path(args.seed))
    else:
        if args.output is None:
            parser.error("audit requires output")
        audit(args.output)
