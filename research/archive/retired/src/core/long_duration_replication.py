"""H051: independently seeded fixed-duration comparison; no NN or optimizer change."""

import argparse
import hashlib
import json
import math
import statistics
import traceback
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import torch

from src.core.affine_longer import output_path as original_path
from src.core.affine_longer import verify_trial as verify_original
from src.core.affine_rate_replication import output_path as short_plain_path
from src.core.affine_rate_replication import verify_plain
from src.core.benchmark import forward_flops
from src.core.config import ModelConfig, TrainConfig
from src.core.data import TokenData
from src.core.long_duration import CRITICAL as PRIOR_CRITICAL
from src.core.long_duration import RECIPES, STEPS, decisions, trajectory_screen
from src.core.long_duration import configuration as original_configuration
from src.core.long_duration_report import load_verified as verify_seed17
from src.core.multihead_screen import verify_archive
from src.core.optimizer_bracket import CACHE, numerical_failure, run_worker
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.trainer import learning_rate, train

PLAN = Path("research/long_duration_replication_plan.md")
SEEDS = (17, 29, 43)
ORDER = tuple((s, r) for s in SEEDS[1:] for r in (tuple(reversed(RECIPES)) if s == 29 else RECIPES))
CRITICAL = (*PRIOR_CRITICAL, "src/core/long_duration_replication.py")


def configuration(recipe, seed):
    if seed not in SEEDS:
        raise ValueError("Seed is outside the frozen replication cohort")
    mc, tc = original_configuration(recipe)
    return mc, replace(tc, seed=seed)


def output_path(recipe, seed):
    return Path(f"results/runs/wikitext2_{recipe}_lr1200_s{seed}_3200")


def paired_description(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = 4.302652729911275 * sd / len(values) ** 0.5
    return {
        "differences": values,
        "mean": mean,
        "sample_sd": sd,
        "exploratory_t95_interval": [mean - half, mean + half],
        "strict_candidate_wins": sum(v < 0 for v in values),
    }


def summarize(rows):
    per_seed = {str(s): decisions(rows.get(s, {})) for s in SEEDS}
    complete = all(d["all_four_complete"] for d in per_seed.values())
    result = {
        "all_twelve_complete": complete,
        "per_seed": per_seed,
        "all_seeds_pass_primary_gates": complete
        and all(d["full_promotion_passes"] for d in per_seed.values()),
    }
    if not complete:
        return result
    means = {r: statistics.mean(rows[s][r]["validation_loss"] for s in SEEDS) for r in RECIPES}
    result.update(
        mean_nll=means,
        sample_sd_nll={
            r: statistics.stdev(rows[s][r]["validation_loss"] for s in SEEDS) for r in RECIPES
        },
        relative_mean_nll_percent={
            r: 100 * (means["blockshuffle"] / means[r] - 1) for r in RECIPES if r != "blockshuffle"
        },
        paired={
            r: paired_description(
                [
                    rows[s]["blockshuffle"]["validation_loss"] - rows[s][r]["validation_loss"]
                    for s in SEEDS
                ]
            )
            for r in RECIPES
            if r != "blockshuffle"
        },
        mean_narrow_margin_at_least_point_two_percent=means["blockshuffle"]
        <= 0.998 * means["calibrated_narrow"],
        every_seed_narrow_margin_at_least_point_two_percent=all(
            d["narrow_margin_at_least_point_two_percent"] for d in per_seed.values()
        ),
    )
    return result


def preflight():
    torch.set_num_threads(4)
    tested = json.loads(Path("results/verification/long_duration_tests_v2.json").read_text())
    final = json.loads(Path("results/verification/long_duration_final_v1.json").read_text())
    assert tested["returncode"] == 0 and tested["source_unchanged"] and final["status"] == "PASS"
    assert all(sha256(Path(n)) == tested["source_files"][n] for n in PRIOR_CRITICAL)
    result17, initial17, _, _ = verify_seed17()
    assert result17["decision"]["full_promotion_passes"]
    old_initial = json.loads(Path("results/affine_longer_v2/preflight.json").read_text())
    assert old_initial["environment"] == environment() == initial17["environment"]
    old17 = json.loads(Path("results/affine_longer_v2/result.json").read_text())
    old_seeds = json.loads(Path("results/affine_replication_v1/result.json").read_text())
    plain_result = json.loads(Path("results/affine_rate_replication_v1/result.json").read_text())
    plain_protocol = json.loads(
        Path("results/affine_rate_replication_v1/protocol.json").read_text()
    )
    references, sampling = {}, {}
    for seed in SEEDS:
        refs, rng = {}, None
        for recipe in RECIPES:
            if recipe == "blockshuffle":
                old_plain, _ = verify_original(recipe, old_initial, seed)
                m, state = verify_plain(seed, old_initial, old_plain)
                p = short_plain_path(seed)
                t = (
                    plain_protocol["seed17"]
                    if seed == 17
                    else next(t for t in plain_result["trials"] if t["seed"] == seed)
                )
            else:
                m, state = verify_original(recipe, old_initial, seed)
                p = original_path(recipe, seed)
                t = next(
                    t
                    for t in (old17 if seed == 17 else old_seeds)["trials"]
                    if t["recipe"] == recipe and (seed == 17 or t["seed"] == seed)
                )
            assert sha256(p / "metrics.json") == t["metrics_sha256"]
            assert sha256(p / "checkpoint.pt") == t["checkpoint_sha256"]
            mc, tc = configuration(recipe, seed)
            assert mc == ModelConfig(**m["model"])
            assert tc == replace(TrainConfig(**m["training"]), steps=STEPS, log_every=800)
            if rng is None:
                rng = state
            assert torch.equal(rng, state)
            refs[recipe] = {
                "run": p.name,
                "metrics_sha256": sha256(p / "metrics.json"),
                "checkpoint_sha256": sha256(p / "checkpoint.pt"),
                "source_archive_sha256": sha256(p / "source.zip"),
            }
        references[str(seed)] = refs
        sampling[str(seed)] = hashlib.sha256(rng.numpy().tobytes()).hexdigest()
    return {
        "references": references,
        "sampling_800_sha256": sampling,
        "seed17_trials": result17["trials"],
        "seed17_result_sha256": sha256(Path("results/long_duration_v1/result.json")),
        "data_hashes": initial17["data_hashes"],
        "environment": environment(),
        "validation_stream_exact": True,
        "current_computation_matches_227_test_snapshot": True,
    }


def sampling_worker(seed, output, initial_path):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    initial = json.loads(initial_path.read_text())
    data = TokenData(CACHE, "cuda", 10000 + seed)
    states, first = {}, None
    for index in range(1, STEPS + 1):
        x, y = data.batch(16, 128)
        if index == 1:
            first = hashlib.sha256(
                x.cpu().numpy().tobytes() + y.cpu().numpy().tobytes()
            ).hexdigest()
        if index in (800, 3200):
            states[f"after_{index}_batches"] = data.generator.get_state().clone()
    torch.cuda.synchronize()
    hashes = {
        k + "_sha256": hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in states.items()
    }
    assert hashes["after_800_batches_sha256"] == initial["sampling_800_sha256"][str(seed)]
    assert data.manifest["files"] == initial["data_hashes"]
    torch.save(states, output / "states.pt")
    record = {
        "status": "PASS",
        "seed": seed,
        "generator_seed": 10000 + seed,
        "generator_device": "cuda",
        "training_batch_calls": STEPS,
        "optimizer_updates": 0,
        "validation_targets_scored": 0,
        "first_batch_sha256": first,
        **hashes,
        "states_sha256": sha256(output / "states.pt"),
        "data_hashes": data.manifest["files"],
        "environment": environment(),
        "provenance": provenance(),
    }
    write_json(output / "result.json", record)
    print(
        json.dumps(
            {
                k: v
                for k, v in record.items()
                if k not in ("provenance", "data_hashes", "environment")
            }
        ),
        flush=True,
    )


def verify_trial(recipe, seed, initial, sampling):
    p = output_path(recipe, seed)
    m = json.loads((p / "metrics.json").read_text())
    mc, tc = configuration(recipe, seed)
    rp = Path("results/runs") / initial["references"][str(seed)][recipe]["run"]
    assert sha256(rp / "metrics.json") == initial["references"][str(seed)][recipe]["metrics_sha256"]
    reference = json.loads((rp / "metrics.json").read_text())
    assert ModelConfig(**m["model"]) == mc and TrainConfig(**m["training"]) == tc
    assert m["training_tokens"] == 6553600 and m["validation_tokens"] == 322688
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
    assert [r["step"] for r in h] == [1, 800, 1600, 2400, 3200]
    assert h[-1]["validation_loss"] == m["validation_loss"]
    old_first = json.loads((rp / "history.jsonl").read_text().splitlines()[0])
    assert abs(h[0]["training_loss"] - old_first["training_loss"]) < 1e-7
    for row in h:
        assert all(
            math.isfinite(row[k])
            for k in ("training_loss", "validation_loss", "gradient_norm_pre_clip")
        )
        assert all(math.isfinite(v) for v in row["layer_gradient_norms_post_clip"].values())
        assert abs(row["learning_rate"] - learning_rate(row["step"] - 1, tc)) < 1e-15
    for stage in ("initial", "final"):
        diag = json.loads((p / f"{stage}_diagnostics.json").read_text())
        assert all(r["finite"] for r in diag.values())
    verify_archive(p, m)
    ckpt = torch.load(p / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert (
        ckpt["step"] == STEPS
        and ckpt["model_config"] == m["model"]
        and ckpt["training_config"] == m["training"]
    )
    assert sum(t.numel() for t in ckpt["model"].values()) == mc.total_parameters
    assert all(torch.isfinite(t).all() for t in ckpt["model"].values())
    assert (
        hashlib.sha256(ckpt["sampling_rng"].numpy().tobytes()).hexdigest()
        == sampling["after_3200_batches_sha256"]
    )
    return m, trajectory_screen(h)


def verify_sampling(output, seed, initial, protocol):
    p = output / "sampling" / f"s{seed}"
    record = json.loads((p / "result.json").read_text())
    assert (
        record["status"] == "PASS"
        and record["seed"] == seed
        and record["generator_seed"] == 10000 + seed
    )
    assert record["generator_device"] == "cuda" and record["training_batch_calls"] == STEPS
    assert record["optimizer_updates"] == record["validation_targets_scored"] == 0
    assert (
        record["data_hashes"] == initial["data_hashes"]
        and record["environment"] == initial["environment"]
    )
    assert record["after_800_batches_sha256"] == initial["sampling_800_sha256"][str(seed)]
    assert sha256(p / "states.pt") == record["states_sha256"]
    states = torch.load(p / "states.pt", map_location="cpu", weights_only=True)
    for count in (800, 3200):
        assert (
            hashlib.sha256(states[f"after_{count}_batches"].numpy().tobytes()).hexdigest()
            == record[f"after_{count}_batches_sha256"]
        )
    assert all(
        record["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
        for n in CRITICAL
    )
    return record


def audit(output, resume_qualification=None):
    completed, attempt = [], 1
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
        failure = json.loads((output / "failure.json").read_text())
        completed, attempt = failure["completed"], failure["attempt"] + 1
        write_json(output / f"resume_{attempt}.json", q)
    try:
        initial = preflight()
        if resume_qualification is None:
            write_json(output / "preflight.json", initial)
            protocol = {
                "plan_sha256": sha256(PLAN),
                "provenance": provenance(),
                "environment": environment(),
                "order": ORDER,
                "new_training_tokens": 52428800,
                "worker_timeout_seconds": 2400,
                "configurations": {
                    f"s{s}_{r}": {
                        "model": asdict(configuration(r, s)[0]),
                        "training": asdict(configuration(r, s)[1]),
                    }
                    for s, r in ORDER
                },
            }
            write_json(output / "protocol.json", protocol)
            with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for n in protocol["provenance"]["source_files"]:
                    archive.write(n, n)
                archive.write(PLAN, PLAN.as_posix())
        else:
            protocol = json.loads((output / "protocol.json").read_text())
            assert initial == json.loads((output / "preflight.json").read_text())
            assert all(
                sha256(Path(n)) == protocol["provenance"]["source_files"][n] for n in CRITICAL
            )
        _, _, seed17_rows, seed17_trajectories = verify_seed17()
        rows, trajectories, sampling = {17: seed17_rows}, {"17": seed17_trajectories}, {}
        for seed in SEEDS[1:]:
            p = output / "sampling" / f"s{seed}"
            if not (p / "result.json").exists():
                assert not p.exists(), "Partial sampler requires separate qualification"
                print(f"Reconstructing CUDA sampling states for seed {seed}", flush=True)
                code = run_worker(
                    output,
                    f"sampling_s{seed}",
                    [
                        "-m",
                        "src.core.long_duration_replication",
                        "sampling",
                        "--seed",
                        str(seed),
                        "--output",
                        str(p),
                        "--initial",
                        str(output / "preflight.json"),
                    ],
                    240,
                    attempt,
                )
                assert code == 0, f"Sampling worker failed: {code}"
            sampling[seed] = verify_sampling(output, seed, initial, protocol)
        for seed, recipe in ORDER:
            p = output_path(recipe, seed)
            retained = next(
                (t for t in completed if t["seed"] == seed and t["recipe"] == recipe), None
            )
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
                m, trajectory = verify_trial(recipe, seed, initial, sampling[seed])
            else:
                assert not p.exists(), f"Unqualified existing trial: {p}"
                print(f"Starting {recipe}, seed {seed}, LR 0.0012, 3200 steps", flush=True)
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
                    [
                        "-m",
                        "src.core.long_duration_replication",
                        "worker",
                        "--recipe",
                        recipe,
                        "--seed",
                        str(seed),
                    ],
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
                        ModelConfig(**failure["model"]) == configuration(recipe, seed)[0]
                        and TrainConfig(**failure["training"]) == configuration(recipe, seed)[1]
                    )
                    assert all(
                        failure["provenance"]["source_files"][n]
                        == protocol["provenance"]["source_files"][n]
                        for n in CRITICAL
                    )
                    verify_archive(p, failure)
                    completed.append(
                        {
                            "seed": seed,
                            "recipe": recipe,
                            "run": p.name,
                            "status": "NUMERICAL_FAIL",
                            "failure_sha256": sha256(p / "failure.json"),
                        }
                    )
                    print(json.dumps(completed[-1]), flush=True)
                    continue
                m, trajectory = verify_trial(recipe, seed, initial, sampling[seed])
                completed.append(
                    {
                        "seed": seed,
                        "recipe": recipe,
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
            rows.setdefault(seed, {})[recipe] = m
            trajectories.setdefault(str(seed), {})[recipe] = trajectory
        result = {
            "status": "complete",
            "trials": completed,
            "summary": summarize(rows),
            "trajectory_screens": trajectories,
            "all_late_plateau_screens_pass": sum(len(v) for v in trajectories.values()) == 12
            and all(
                t["late_plateau_screen_passes"] for v in trajectories.values() for t in v.values()
            ),
            "plan_sha256": sha256(PLAN),
            "numerical_failures": sum(t["status"] == "NUMERICAL_FAIL" for t in completed),
            "all_complete_sampling_states_match_reconstruction": True,
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
    parser.add_argument("command", choices=("preflight", "sampling", "worker", "audit"))
    parser.add_argument("--seed", type=int, choices=SEEDS[1:])
    parser.add_argument("--recipe", choices=RECIPES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--resume-qualification", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        if args.recipe is None or args.seed is None:
            parser.error("worker requires recipe and seed")
        train(*configuration(args.recipe, args.seed), CACHE, output_path(args.recipe, args.seed))
    elif args.output is None:
        parser.error("output is required")
    elif args.command == "preflight":
        assert not args.output.exists()
        write_json(args.output, preflight())
    elif args.command == "sampling":
        if args.seed is None or args.initial is None:
            parser.error("sampling requires seed and initial record")
        sampling_worker(args.seed, args.output, args.initial)
    else:
        audit(args.output, args.resume_qualification)
