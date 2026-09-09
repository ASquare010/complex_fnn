"""H050: four-times-longer, fixed-recipe WikiText comparison with CUDA RNG proof."""

import argparse
import hashlib
import json
import math
import traceback
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import torch

from src.core.benchmark import forward_flops
from src.core.config import ModelConfig, TrainConfig
from src.core.data import TokenData
from src.core.multihead_screen import verify_archive
from src.core.optimizer_bracket import CACHE, RECIPES, numerical_failure, run_worker
from src.core.optimizer_bracket import configuration as rate_configuration
from src.core.optimizer_lower import CRITICAL as PRIOR_CRITICAL
from src.core.optimizer_lower_report import load_verified
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.trainer import learning_rate, train
from src.core.wikitext_screen import promotion

PLAN = Path("research/long_duration_plan.md")
STEPS = 3200
CRITICAL = (*PRIOR_CRITICAL, "src/core/long_duration.py")


def configuration(recipe):
    mc, tc = rate_configuration(recipe, 0.0012)
    return mc, replace(tc, steps=STEPS, log_every=800)


def output_path(recipe):
    return Path(f"results/runs/wikitext2_{recipe}_lr1200_s17_3200")


def trajectory_screen(history):
    """Operational late stability only; it cannot prove optimal convergence."""
    by_step = {r["step"]: r["validation_loss"] for r in history}
    assert 2400 in by_step and 3200 in by_step
    assert all(math.isfinite(v) and v > 0 for v in by_step.values())
    change = 100 * (by_step[3200] / by_step[2400] - 1)
    degradation = 100 * (by_step[3200] / min(by_step.values()) - 1)
    return {
        "relative_last_800_step_nll_change_percent": change,
        "relative_final_cost_over_best_recorded_percent": degradation,
        "late_plateau_screen_passes": abs(change) <= 0.2 and degradation <= 0.2,
    }


def decisions(rows):
    if set(rows) != set(RECIPES):
        return {"all_four_complete": False, "quality_passes": False, "full_promotion_passes": False}
    gates = promotion(rows)
    return {
        "all_four_complete": True,
        "gates": gates,
        "quality_passes": all(v for k, v in gates.items() if not k.startswith("memory_")),
        "full_promotion_passes": all(gates.values()),
        "narrow_margin_at_least_point_two_percent": rows["blockshuffle"]["validation_loss"]
        <= 0.998 * rows["calibrated_narrow"]["validation_loss"],
        "relative_nll_percent": {
            r: 100 * (rows["blockshuffle"]["validation_loss"] / m["validation_loss"] - 1)
            for r, m in rows.items()
            if r != "blockshuffle"
        },
    }


def preflight():
    torch.set_num_threads(4)
    tested = json.loads(Path("results/verification/optimizer_lower_tests_v1.json").read_text())
    final = json.loads(Path("results/verification/optimizer_lower_final_v1.json").read_text())
    assert (
        tested["returncode"] == 0
        and tested["source_unchanged_through_tests"]
        and final["status"] == "PASS"
    )
    assert all(sha256(Path(n)) == tested["source_files"][n] for n in PRIOR_CRITICAL)
    lower, prior, matrix = load_verified()
    d = lower["decision"]
    assert d["full_promotion_passes"] and d["all_selected_rates_unchanged_at_0012"]
    assert all(v == "interior" for v in d["grid_positions"].values())
    initial = prior["prior_initial"]
    assert initial["environment"] == environment()
    assert all(sha256(CACHE / n) == h for n, h in initial["data_hashes"].items())
    data = TokenData(CACHE, "cpu", 10017)
    batches = list(data.validation(16, 128, 158))
    assert len(batches) == 158 and batches[-1][0].shape == (9, 128)
    assert torch.equal(torch.cat([y.flatten() for _, y in batches]), data.valid[1:322689])
    references = {}
    for r in RECIPES:
        p = Path("results/runs") / d["selected"][r]
        m = next(v for v in matrix[r] if v["run"] == p.name)
        mc, tc = configuration(r)
        assert mc == ModelConfig(**m["model"])
        assert tc == replace(TrainConfig(**m["training"]), steps=STEPS, log_every=800)
        references[r] = {
            "run": p.name,
            "metrics_sha256": sha256(p / "metrics.json"),
            "checkpoint_sha256": sha256(p / "checkpoint.pt"),
            "source_archive_sha256": sha256(p / "source.zip"),
        }
    return {
        "references": references,
        "data_hashes": initial["data_hashes"],
        "environment": environment(),
        "sampling_800_sha256": initial["sampling_rng_sha256"],
        "lower_result_sha256": sha256(Path("results/optimizer_lower_v1/result.json")),
        "validation_stream_exact": True,
        "current_computation_matches_225_test_snapshot": True,
    }


def sampling_worker(output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    data = TokenData(CACHE, "cuda", 10017)
    state800 = None
    first = None
    for index in range(1, STEPS + 1):
        x, y = data.batch(16, 128)
        if index == 1:
            first = hashlib.sha256(
                x.cpu().numpy().tobytes() + y.cpu().numpy().tobytes()
            ).hexdigest()
        if index == 800:
            state800 = data.generator.get_state().clone()
    torch.cuda.synchronize()
    final = data.generator.get_state().clone()
    previous = json.loads(Path("results/optimizer_bracket_v1/preflight.json").read_text())
    hash800 = hashlib.sha256(state800.numpy().tobytes()).hexdigest()
    assert hash800 == previous["sampling_rng_sha256"]
    assert first == "258e6c6132c5ac9d202dd965ee64a530055aa414fd62b396b79c4d1be392362e"
    torch.save({"after_800_batches": state800, "after_3200_batches": final}, output / "states.pt")
    record = {
        "status": "PASS",
        "first_batch_sha256": first,
        "after_800_batches_sha256": hash800,
        "after_3200_batches_sha256": hashlib.sha256(final.numpy().tobytes()).hexdigest(),
        "generator_device": "cuda",
        "training_batch_calls": STEPS,
        "optimizer_updates": 0,
        "validation_targets_scored": 0,
        "data_hashes": data.manifest["files"],
        "provenance": provenance(),
        "environment": environment(),
        "states_sha256": sha256(output / "states.pt"),
    }
    write_json(output / "result.json", record)
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "status",
                    "first_batch_sha256",
                    "after_800_batches_sha256",
                    "after_3200_batches_sha256",
                )
            }
        ),
        flush=True,
    )


def verify_trial(recipe, initial, sampling):
    p = output_path(recipe)
    m = json.loads((p / "metrics.json").read_text())
    mc, tc = configuration(recipe)
    rp = Path("results/runs") / initial["references"][recipe]["run"]
    assert sha256(rp / "metrics.json") == initial["references"][recipe]["metrics_sha256"]
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
        previous = json.loads((output / "failure.json").read_text())
        completed = previous["completed"]
        attempt = previous["attempt"] + 1
        write_json(output / f"resume_{attempt}.json", q)
    try:
        initial = preflight()
        if resume_qualification is None:
            write_json(output / "preflight.json", initial)
            protocol = {
                "plan_sha256": sha256(PLAN),
                "provenance": provenance(),
                "environment": environment(),
                "recipes": RECIPES,
                "new_training_tokens": 26214400,
                "steps": STEPS,
                "worker_timeout_seconds": 2400,
                "configurations": {
                    r: {
                        "model": asdict(configuration(r)[0]),
                        "training": asdict(configuration(r)[1]),
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
            protocol = json.loads((output / "protocol.json").read_text())
            assert protocol["plan_sha256"] == sha256(PLAN)
            assert all(
                sha256(Path(n)) == protocol["provenance"]["source_files"][n] for n in CRITICAL
            )
            assert initial == json.loads((output / "preflight.json").read_text())
        if not (output / "sampling/result.json").exists():
            assert not (output / "sampling").exists(), "Unqualified partial sampling directory"
            print("Reconstructing the exact 800/3200-batch CUDA sampling states", flush=True)
            write_json(
                output / "progress.json",
                {"status": "sampling", "attempt": attempt, "completed": completed},
            )
            code = run_worker(
                output,
                "sampling",
                ["-m", "src.core.long_duration", "sampling", "--output", str(output / "sampling")],
                600,
                attempt,
            )
            assert code == 0, f"Sampling worker failed with returncode {code}"
        sampling = json.loads((output / "sampling/result.json").read_text())
        assert (
            sampling["status"] == "PASS"
            and sampling["after_800_batches_sha256"] == initial["sampling_800_sha256"]
        )
        assert sampling["data_hashes"] == initial["data_hashes"]
        assert all(
            sampling["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
            for n in CRITICAL
        )
        rows, trajectories = {}, {}
        for r in RECIPES:
            p = output_path(r)
            retained = next((t for t in completed if t["recipe"] == r), None)
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
                m, trajectory = verify_trial(r, initial, sampling)
            else:
                assert not p.exists(), f"Unqualified existing trial: {p}"
                print(f"Starting {r}, LR 0.0012, 3200 steps", flush=True)
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
                    ["-m", "src.core.long_duration", "worker", "--recipe", r],
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
                            "run": p.name,
                            "status": "NUMERICAL_FAIL",
                            "failure_sha256": sha256(p / "failure.json"),
                        }
                    )
                    print(json.dumps(completed[-1]), flush=True)
                    continue
                m, trajectory = verify_trial(r, initial, sampling)
                completed.append(
                    {
                        "recipe": r,
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
            rows[r] = m
            trajectories[r] = trajectory
        result = {
            "status": "complete",
            "trials": completed,
            "decision": decisions(rows),
            "trajectory_screens": trajectories,
            "all_late_plateau_screens_pass": len(trajectories) == 4
            and all(v["late_plateau_screen_passes"] for v in trajectories.values()),
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
    parser.add_argument("--recipe", choices=RECIPES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-qualification", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        if args.recipe is None:
            parser.error("worker requires recipe")
        train(*configuration(args.recipe), CACHE, output_path(args.recipe))
    elif args.output is None:
        parser.error("output is required")
    elif args.command == "preflight":
        assert not args.output.exists()
        write_json(args.output, preflight())
    elif args.command == "sampling":
        sampling_worker(args.output)
    else:
        audit(args.output, args.resume_qualification)
