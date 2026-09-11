"""Freeze H110 and dispatch one GPU worker at a time; stop on paired fidelity failure."""

import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import torch

from results.token_memory_duration_v1.source.common import POLICIES, SEEDS, SHAPES, qualify
from src.core.data import load_manifest
from src.core.reproducibility import environment, provenance, sha256, write_json

ROOT = Path("results/token_memory_duration_v1")


def read(path):
    return json.loads(Path(path).read_text())


def run():
    assert not (ROOT / "protocol.json").exists(), "Preserve prior/partial science"
    assert read("results/verification/sobolev_learning_final_v1.json")["status"] == "PASS"
    prior = read("results/verification/token_memory_final_v1.json")
    assert prior["status"] == "PASS"
    for path, digest in prior["files"].items():
        assert sha256(Path(path)) == digest, path
    prov = provenance()
    sources = dict(prov["source_files"])
    for path in [
        *(ROOT / "source").glob("*.py"),
        Path("research/token_memory_duration_plan.md"),
        Path("results/token_memory_v1/source/execution.py"),
    ]:
        sources[path.as_posix()] = sha256(path)
    protocol = {
        "sources": sources,
        "provenance": prov,
        "environment": environment(),
        "data": load_manifest(Path("data/wikitext2_v1")),
        "seeds": SEEDS,
        "shapes": SHAPES,
        "policies": POLICIES,
        "steps": 800,
        "chunk_size": 512,
        "timing_warmup": 20,
        "planned_updates": 9600,
        "learning_rate": 0.0006,
        "schedule": "constant",
        "precision": "bf16",
        "allocator": "malloc",
        "pythonhashseed": 107,
        "stop_on_pair_nll_failure": True,
        "previous_goal_turn": "progress",
    }
    write_json(ROOT / "protocol.json", protocol)
    with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path)
    checks = qualify()
    write_json(ROOT / "qualification.json", {"passed": True, "checks": checks})
    print(f"Qualification: {len(checks)} model/shape/device cases passed", flush=True)
    # Qualification objects are gone; workers have independent CUDA contexts.
    rows, pairs = [], []
    started = time.perf_counter()
    status, reason = "COMPLETE", None
    for batch, context in SHAPES:
        for seed_index, seed in enumerate(SEEDS):
            order = POLICIES[seed_index % 2 :] + POLICIES[: seed_index % 2]
            completed = {}
            for policy in order:
                label = f"b{batch}_t{context}_s{seed}_{policy}"
                log_path = ROOT / f"{label}.log"
                with log_path.open("x") as log:
                    worker = subprocess.Popen(
                        [
                            sys.executable,
                            "-X",
                            "faulthandler",
                            "-u",
                            "-m",
                            "results.token_memory_duration_v1.source.worker",
                            str(batch),
                            str(context),
                            str(seed),
                            policy,
                        ],
                        env=os.environ.copy(),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    write_json(
                        ROOT / "coordinator_status.json",
                        {
                            "status": "RUNNING",
                            "worker_pid": worker.pid,
                            "label": label,
                            "completed_trials": len(rows),
                        },
                    )
                    code = worker.wait()
                (ROOT / f"{label}_exit.txt").write_text(str(code))
                if code:
                    write_json(
                        ROOT / "coordinator_status.json",
                        {
                            "status": "WORKER_FAILED",
                            "label": label,
                            "exit_code": code,
                            "completed_trials": len(rows),
                        },
                    )
                    raise RuntimeError(f"Worker {label} failed with {code}; do not retry silently")
                row = read(ROOT / "runs" / label / "metrics.json")
                rows.append(row)
                completed[policy] = row
                print(
                    json.dumps(
                        {
                            "completed_trials": len(rows),
                            "label": label,
                            "nll": row["full_validation"]["nll"],
                            "training_mib": row["peak_training_allocated_bytes"] / 2**20,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
            reference, candidate = completed["block"], completed["loss_chunks"]
            assert reference["initial_state_hashes"] == candidate["initial_state_hashes"]
            assert reference["initial_sampler_sha256"] == candidate["initial_sampler_sha256"]
            assert reference["final_sampler_sha256"] == candidate["final_sampler_sha256"]
            assert reference["data_order_sha256"] == candidate["data_order_sha256"]
            full_error = abs(
                candidate["full_validation"]["nll"] / reference["full_validation"]["nll"] - 1
            )
            trajectory = {
                str(a["step"]): abs(b["nll"] / a["nll"] - 1)
                for a, b in zip(reference["evaluations"], candidate["evaluations"], strict=True)
            }
            pair = {
                "batch": batch,
                "context": context,
                "seed": seed,
                "full_relative_nll_difference_abs": full_error,
                "trajectory_relative_nll_difference_abs": trajectory,
                "fidelity_passes": full_error <= 0.01
                and all(v <= 0.01 for v in trajectory.values()),
            }
            pairs.append(pair)
            write_json(ROOT / "paired_progress.json", {"pairs": pairs})
            if not pair["fidelity_passes"]:
                status = "STOPPED_FIDELITY_GATE"
                reason = pair
                break
        if reason:
            break
    write_json(
        ROOT / "result.json",
        {
            "status": status,
            "cases": rows,
            "pairs": pairs,
            "stop_reason": reason,
            "elapsed_seconds": time.perf_counter() - started,
            "updates": 800 * len(rows),
            "planned_updates": 9600,
            "broad_goal_achieved": False,
        },
    )
    write_json(ROOT / "coordinator_status.json", {"status": status, "completed_trials": len(rows)})
    print(f"H110 {status}: {len(rows)} completed trials", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        run()
    except Exception:
        import traceback

        write_json(ROOT / "study_failure.json", {"traceback": traceback.format_exc()})
        raise
