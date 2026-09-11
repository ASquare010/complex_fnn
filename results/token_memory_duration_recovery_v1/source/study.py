"""Resume only never-trained trials with explicit dependency-preload recovery."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("results/token_memory_duration_recovery_v1")
OLD = Path("results/token_memory_duration_v1")


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def run():
    assert (ROOT / "before.json").exists() and not (ROOT / "coordinator_status.json").exists()
    # Two explicit CPU dependency checks, not model trials or an automatic retry loop.
    for index in (1, 2):
        with (ROOT / f"dependency_probe{index}.log").open("x") as log:
            code = subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "faulthandler",
                    "-u",
                    "-m",
                    "results.token_memory_duration_recovery_v1.source.worker",
                ],
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        (ROOT / f"dependency_probe{index}_exit.txt").write_text(str(code))
        assert code == 0, "Dependency check failed; do not start training"
    rows, pairs = [], []
    started = time.perf_counter()
    status, reason = "COMPLETE", None
    for batch, context in ((16, 128), (8, 512)):
        for seed_index, seed in enumerate((17, 29, 43)):
            policies = ("block", "loss_chunks") if seed_index % 2 == 0 else ("loss_chunks", "block")
            completed = {}
            for policy in policies:
                label = f"b{batch}_t{context}_s{seed}_{policy}"
                if label == "b16_t128_s17_block":
                    row = read(OLD / "runs" / label / "metrics.json")
                    row["source_root"] = OLD.as_posix()
                else:
                    with (ROOT / f"{label}.log").open("x") as log:
                        process = subprocess.Popen(
                            [
                                sys.executable,
                                "-X",
                                "faulthandler",
                                "-u",
                                "-m",
                                "results.token_memory_duration_recovery_v1.source.worker",
                                str(batch),
                                str(context),
                                str(seed),
                                policy,
                            ],
                            env=os.environ.copy(),
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                        write(
                            "coordinator_status.json",
                            {
                                "status": "RUNNING",
                                "worker_pid": process.pid,
                                "label": label,
                                "completed_trials": len(rows),
                            },
                        )
                        code = process.wait()
                    (ROOT / f"{label}_exit.txt").write_text(str(code))
                    if code:
                        write(
                            "coordinator_status.json",
                            {
                                "status": "WORKER_FAILED",
                                "label": label,
                                "exit_code": code,
                                "completed_trials": len(rows),
                            },
                        )
                        raise RuntimeError(f"Recovery worker {label} failed: {code}")
                    row = read(ROOT / "runs" / label / "metrics.json")
                    row["source_root"] = ROOT.as_posix()
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
            ref, candidate = completed["block"], completed["loss_chunks"]
            for key in (
                "initial_state_hashes",
                "initial_sampler_sha256",
                "final_sampler_sha256",
                "data_order_sha256",
            ):
                assert ref[key] == candidate[key], key
            full_error = abs(
                candidate["full_validation"]["nll"] / ref["full_validation"]["nll"] - 1
            )
            trajectory = {
                str(a["step"]): abs(b["nll"] / a["nll"] - 1)
                for a, b in zip(ref["evaluations"], candidate["evaluations"], strict=True)
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
            write("paired_progress.json", {"pairs": pairs})
            if not pair["fidelity_passes"]:
                status, reason = "STOPPED_FIDELITY_GATE", pair
                break
        if reason:
            break
    write(
        "result.json",
        {
            "status": status,
            "cases": rows,
            "pairs": pairs,
            "stop_reason": reason,
            "elapsed_seconds": time.perf_counter() - started,
            "updates": 800 * len(rows),
            "planned_updates": 9600,
            "original_completed_control_reused": True,
            "repeated_scientific_updates": 0,
            "broad_goal_achieved": False,
        },
    )
    write("coordinator_status.json", {"status": status, "completed_trials": len(rows)})
    print(f"Recovered H110 {status}: {len(rows)} trials", flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        import traceback

        write("study_failure.json", {"traceback": traceback.format_exc()})
        raise
