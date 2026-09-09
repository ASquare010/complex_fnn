"""H047 qualification and frozen three-rate screen for router-free headwise SwiGLU."""

import argparse
import gc
import hashlib
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
from src.core.config import VARIANTS, ModelConfig, TrainConfig
from src.core.data import TokenData, load_manifest
from src.core.diagnostics import inspect_layers
from src.core.multihead_screen import (
    COMPUTATION,
    RATES,
    candidate_gates,
    configuration,
    output_path,
    verify_archive,
    verify_trial,
)
from src.core.optimization import group_summary, parameter_groups
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.transformer import Transformer
from src.core.wikitext_screen import RECIPES as CONTROLS
from src.core.wikitext_screen import output_path as control_path
from src.core.wikitext_screen import select_best

PLAN = Path("research/headwise_screen_plan.md")
CACHE = Path("data/wikitext2_v1")
RECIPE = "headwise"
CRITICAL = (
    *COMPUTATION,
    "src/multihead_ffn/headwise.py",
    "src/core/headwise_screen.py",
    "src/core/multihead_screen.py",
    "configs/wikitext2_headwise_screen.json",
)


def preflight():
    torch.set_num_threads(4)
    snapshot = Path("results/verification/headwise_before_v1.pt")
    meta = json.loads(snapshot.with_suffix(".json").read_text())
    assert sha256(snapshot) == meta["sha256"]
    changed = {n for n, h in meta["provenance"]["source_files"].items() if sha256(Path(n)) != h}
    assert changed == {"src/core/config.py", "src/core/transformer.py", "src/core/benchmark.py"}
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
    assert len(before) == 30 and set(VARIANTS) - set(before) == {"headwise_swiglu"}
    del before, m
    previous = json.loads(Path("results/multihead_screen_v2/preflight.json").read_text())
    outcome = json.loads(Path("results/multihead_screen_v2/result.json").read_text())
    assert outcome["status"] == "complete" and not any(
        v["quality_investigation_passes"] for v in outcome["decisions"].values()
    )
    data = TokenData(CACHE, "cpu", 10017)
    manifest = load_manifest(CACHE)
    assert manifest["files"] == previous["data_hashes"]
    assert all(sha256(CACHE / n) == h for n, h in manifest["files"].items())
    batches = list(data.validation(16, 128, 158))
    assert len(batches) == 158 and batches[-1][0].shape == (9, 128)
    assert torch.equal(torch.cat([y.flatten() for _, y in batches]), data.valid[1:322689])
    for row in previous["controls"]:
        p = Path("results/runs") / row["run"]
        assert sha256(p / "metrics.json") == row["metrics_sha256"]
        assert sha256(p / "checkpoint.pt") == row["checkpoint_sha256"]
        assert sha256(p / "source.zip") == row["source_archive_sha256"]
        verify_archive(p, json.loads((p / "metrics.json").read_text()))
    for row in outcome["trials"]:
        p = Path("results/runs") / row["run"]
        assert sha256(p / "metrics.json") == row["metrics_sha256"]
        assert sha256(p / "checkpoint.pt") == row["checkpoint_sha256"]
    gaussian = torch.randn(4096, 384, generator=torch.Generator().manual_seed(73))
    moments, common = {}, None
    for r in ("full", RECIPE):
        c = (
            configuration(RECIPE, 0.0006)[0]
            if r == RECIPE
            else ModelConfig(variant="swiglu", width=384, layers=8, heads=6)
        )
        m = Transformer(c, 17)
        state = {n: p.detach().clone() for n, p in m.named_parameters() if ".ffn." not in n}
        if common is None:
            common = state
        else:
            assert all(torch.equal(p, common[n]) for n, p in state.items())
        with torch.no_grad():
            y = m.blocks[0].ffn(gaussian)
        assert torch.isfinite(y).all()
        moments[r] = y.square().mean().sqrt().item()
        if r == RECIPE:
            _, tc = configuration(RECIPE, 0.0006)
            groups = group_summary(parameter_groups(m, tc))
        del m, y
    ratio = moments[RECIPE] / moments["full"]
    assert 0.5 <= ratio <= 2.0
    return {
        "old_variants_exact": 30,
        "old_snapshot_sha256": sha256(snapshot),
        "qualified_changed_files": sorted(changed),
        "data_hashes": manifest["files"],
        "environment": environment(),
        "controls": previous["controls"],
        "parallel_trials": outcome["trials"],
        "sampling_rng_sha256": previous["sampling_rng_sha256"],
        "validation_stream_exact": True,
        "moments": moments,
        "calibrated_full_rms_ratio": ratio,
        "new_cpu_checks": [{"recipe": RECIPE, "optimizer_groups": groups}],
        "common_non_ffn_initial_parameters_exact": True,
    }


def qualify(output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    c, _ = configuration(RECIPE, 0.0006)
    tc = TrainConfig(
        steps=10,
        batch_size=16,
        learning_rate=0.0006,
        seed=17,
        precision="bf16",
        eval_batches=1,
        log_every=10,
    )
    m = Transformer(c, 17).cuda()
    data = TokenData(CACHE, "cuda", 10017)
    x, y = data.batch(16, 128)
    batch_hash = hashlib.sha256(x.cpu().numpy().tobytes() + y.cpu().numpy().tobytes()).hexdigest()
    assert batch_hash == "258e6c6132c5ac9d202dd965ee64a530055aa414fd62b396b79c4d1be392362e"
    with torch.no_grad():
        expected = m(x).float().cpu()
        with autocast("cuda", "bf16"):
            actual = m(x).float().cpu()
    relative = (actual - expected).norm().item() / expected.norm().item()
    assert relative <= 0.05 and torch.isfinite(actual).all()
    del actual, expected
    optimizer = torch.optim.AdamW(
        parameter_groups(m, tc), lr=tc.learning_rate, betas=(0.9, 0.95), eps=1e-8
    )
    diagnostics = inspect_layers(m, x, "cuda", "bf16")
    assert all(v["finite"] for v in diagnostics.values())
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    history = []
    for step in range(1, 11):
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", "bf16"):
            loss = m.loss(x, y)
        loss.backward()
        norms = {n: p.grad.float().norm().item() for n, p in m.named_parameters()}
        assert all(math.isfinite(v) and v > 0 for v in norms.values())
        assert all(torch.isfinite(p.grad).all() for p in m.parameters())
        norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0).item()
        assert math.isfinite(loss.item()) and math.isfinite(norm)
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in m.parameters())
        history.append(
            {
                "step": step,
                "loss": loss.item(),
                "gradient_norm_pre_clip": norm,
                "gradient_norms_pre_clip": norms,
            }
        )
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    assert history[-1]["loss"] < history[0]["loss"]
    torch.save(
        {
            "step": 10,
            "model": m.state_dict(),
            "model_config": asdict(c),
            "training_config": asdict(tc),
            "batch_sha256": batch_hash,
            "scope": "fixed training batch qualification; no validation",
        },
        output / "checkpoint.pt",
    )
    record = {
        "status": "PASS",
        "model": asdict(c),
        "training": asdict(tc),
        "provenance": provenance(),
        "environment": environment(),
        "precision_relative_l2": relative,
        "history": history,
        "peak_training_allocated_vram_bytes": peak,
        "initial_diagnostics": diagnostics,
        "batch_sha256": batch_hash,
        "training_token_exposures": 20480,
        "validation_targets": 0,
        "checkpoint_sha256": sha256(output / "checkpoint.pt"),
    }
    write_json(output / "result.json", record)
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "status",
                    "precision_relative_l2",
                    "peak_training_allocated_vram_bytes",
                    "batch_sha256",
                )
            }
        )
    )


def run_worker(output, name, command, timeout):
    with (output / f"{name}.log").open("x", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, "-X", "faulthandler", *command],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )


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
            "new_screen_training_tokens": 1228800,
            "qualification_training_token_exposures": 20480,
        }
        write_json(output / "protocol.json", protocol)
        with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for n in protocol["provenance"]["source_files"]:
                archive.write(n, n)
            archive.write(PLAN, PLAN.as_posix())
        print("CPU qualification passed; starting fixed-batch GPU qualification", flush=True)
        write_json(output / "progress.json", {"status": "qualifying", "completed": completed})
        run_worker(
            output,
            "qualification",
            [
                "-m",
                "src.core.headwise_screen",
                "qualify",
                "--output",
                str(output / "qualification"),
            ],
            600,
        )
        qualification = json.loads((output / "qualification/result.json").read_text())
        assert qualification["status"] == "PASS"
        assert all(
            qualification["provenance"]["source_files"][n]
            == protocol["provenance"]["source_files"][n]
            for n in CRITICAL
        )
        print(
            "GPU qualification passed; reproducing four original initial validation losses",
            flush=True,
        )
        run_worker(
            output,
            "baseline_initial",
            [
                "-m",
                "src.core.multihead_screen",
                "baseline-initial",
                "--output",
                str(output / "baseline_initial.json"),
            ],
            600,
        )
        baseline = json.loads((output / "baseline_initial.json").read_text())
        assert baseline["status"] == "PASS"
        assert all(
            baseline["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
            for n in CRITICAL
        )
        rows = []
        for rate in RATES:
            p = output_path(RECIPE, rate)
            assert not p.exists(), f"Do not overwrite {p}"
            print(f"Starting headwise, LR {rate}, 200 steps", flush=True)
            write_json(
                output / "progress.json",
                {"status": "running", "active": p.name, "completed": completed},
            )
            run_worker(
                output,
                p.name,
                ["-m", "src.core.headwise_screen", "worker", "--rate", str(rate)],
                1200,
            )
            m = verify_trial(RECIPE, rate, initial)
            assert all(
                m["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
                for n in CRITICAL
            )
            rows.append(m)
            completed.append(
                {
                    "recipe": RECIPE,
                    "rate": rate,
                    "run": p.name,
                    "nll": m["validation_loss"],
                    "peak_mib": m["peak_allocated_vram_bytes"] / 2**20,
                    "metrics_sha256": sha256(p / "metrics.json"),
                    "checkpoint_sha256": sha256(p / "checkpoint.pt"),
                }
            )
            print(json.dumps(completed[-1]), flush=True)
        controls = {
            r: [json.loads((control_path(r, rate) / "metrics.json").read_text()) for rate in RATES]
            for r in CONTROLS
        }
        references = {r: select_best(v) for r, v in controls.items()}
        selected = select_best(rows)
        decision = candidate_gates(selected, references)
        same_rate = {
            str(round(rate * 1e6)): {
                r: 100 * (rows[i]["validation_loss"] / cells[i]["validation_loss"] - 1)
                for r, cells in controls.items()
            }
            for i, rate in enumerate(RATES)
        }
        assert (
            max(m["initial_validation_loss"] for m in rows)
            - min(m["initial_validation_loss"] for m in rows)
            < 1e-7
        )
        result = {
            "status": "complete",
            "trials": completed,
            "selected": selected["run"],
            "decisions": decision,
            "same_rate_relative_nll_percent": same_rate,
            "grid_boundary_winner": selected["training"]["learning_rate"] in (RATES[0], RATES[-1]),
            "plan_sha256": sha256(PLAN),
            "all_diagnostics_finite": True,
            "sampling_rng_matches_controls": True,
            "old_variants_exact": 30,
            "qualification_passes": True,
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
    parser.add_argument("command", choices=("preflight", "qualify", "worker", "audit"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rate", type=float, choices=RATES)
    args = parser.parse_args()
    if args.command == "worker":
        if args.rate is None:
            parser.error("worker requires rate")
        from src.core.trainer import train

        train(*configuration(RECIPE, args.rate), CACHE, output_path(RECIPE, args.rate))
    elif args.output is None:
        parser.error("output is required")
    elif args.command == "preflight":
        assert not args.output.exists()
        write_json(args.output, preflight())
    elif args.command == "qualify":
        qualify(args.output)
    else:
        audit(args.output)
