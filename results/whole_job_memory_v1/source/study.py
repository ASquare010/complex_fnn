"""One sequential fresh process per fixed corpus/seed/policy; no automatic retries."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("results/whole_job_memory_v1")


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + "\n")


def command(*args):
    return [
        sys.executable,
        "-B",
        "-X",
        "pycache_prefix=" + str(ROOT.resolve() / "unused_worker_cache"),
        "-X",
        "faulthandler",
        "-u",
        "-m",
        "results.whole_job_memory_v1.source.worker",
        *args,
    ]


def child(label, args):
    with (ROOT / f"{label}.log").open("x") as log:
        process = subprocess.Popen(
            command(*args), env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT
        )
        write(
            "coordinator_status.json",
            {"status": "RUNNING", "worker_pid": process.pid, "label": label},
        )
        code = process.wait()
    (ROOT / f"{label}_exit.txt").write_text(str(code))
    if code:
        write(
            "coordinator_status.json",
            {"status": "WORKER_FAILED", "label": label, "exit_code": code},
        )
        raise RuntimeError(f"H112 worker {label} failed: {code}")


def run():
    assert (ROOT / "protocol.json").exists() and not (ROOT / "coordinator_status.json").exists()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    child("qualification", ["qualify"])
    rows = []
    start = time.perf_counter()
    for corpus_index, dataset in enumerate(("wikitext2", "tinystories")):
        for seed_index, seed in enumerate((61, 73, 89)):
            policies = (
                ("block", "loss_chunks")
                if (seed_index + corpus_index) % 2 == 0
                else ("loss_chunks", "block")
            )
            pair = []
            for policy in policies:
                trial = f"b8_t512_s{seed}_{policy}"
                child(f"{dataset}_{trial}", [dataset, str(seed), policy])
                row = json.loads((ROOT / dataset / "runs" / trial / "metrics.json").read_text())
                adapter = json.loads((ROOT / dataset / "runs" / trial / "adapter.json").read_text())
                row["extra_checkpoints"] = adapter["extra_checkpoints"]
                row.update(
                    dataset=dataset,
                    evaluation_policy=protocol["evaluation_policy"],
                    source_root=(ROOT / dataset).as_posix(),
                )
                rows.append(row)
                pair.append(row)
                write(
                    "completed_progress.json",
                    {
                        "completed_trials": len(rows),
                        "labels": [f"{r['dataset']}_{r['label']}" for r in rows],
                        "updates": 800 * len(rows),
                    },
                )
                print(
                    json.dumps(
                        {
                            "completed_trials": len(rows),
                            "dataset": dataset,
                            "label": trial,
                            "streamed_nll": row["full_validation"]["nll"],
                            "training_mib": row["peak_training_allocated_bytes"] / 2**20,
                            "job_mib": row["peak_job_allocated_bytes"] / 2**20,
                            "elapsed_seconds": time.perf_counter() - start,
                        }
                    ),
                    flush=True,
                )
            for key in (
                "initial_state_hashes",
                "initial_sampler_sha256",
                "final_sampler_sha256",
                "data_order_sha256",
            ):
                assert pair[0][key] == pair[1][key], key
    assert len(rows) == 12
    write(
        "result.json",
        {
            "status": "COMPLETE",
            "cases": rows,
            "updates": 9600,
            "training_targets": sum(r["training_targets"] for r in rows),
            "elapsed_seconds": time.perf_counter() - start,
            "primary_quality_requires_native_audit": True,
            "repeated_scientific_updates": 0,
            "broad_goal_achieved": False,
        },
    )
    write("coordinator_status.json", {"status": "COMPLETE", "completed_trials": 12})


if __name__ == "__main__":
    try:
        run()
    except Exception:
        import traceback

        write("failure.json", {"traceback": traceback.format_exc()})
        raise
