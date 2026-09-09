"""H075 one bounded construction diagnostic and operational recovery, with durable records."""

import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from results.ungated_resource_recovery_v1.source.recovery import PLAN, RECOVERY_CELLS, ROOT
from results.ungated_resource_v1.source import study as original
from src.core.native_recompute_audit import read, sha, write_new
from src.core.reproducibility import environment, provenance, write_json

before = read(ROOT / "before.json")
old = Path("results/ungated_resource_v1")
assert not (ROOT / "protocol.json").exists()
assert sha(PLAN) == before["plan_sha256"]
assert sha("results/verification/ungated_resource_final_v1.json") == before["h074_final_sha256"]
assert sha(old / "failure.json") == before["h074_failure_sha256"]
assert all(sha(n) == h for n, h in before["h074_sources"].items())
previous = read(old / "protocol.json")
assert environment() == previous["environment"]
source_probe = read("results/verification/ungated_resource_tokenizer_probe_v1.json")
assert all(sha(n) == v["sha256"] for n, v in source_probe["files"].items())
sources = dict(before["h074_sources"])
sources.update({p.as_posix(): sha(p) for p in (ROOT / "source").glob("*.py")})
assert len(sources) == 103
preserved = read("results/verification/ungated_resource_partial_analysis_v1.json")[
    "worker_artifact_hashes"
]
assert len(preserved) == 17
assert set(original.CELLS) - set(preserved) == set(RECOVERY_CELLS)
protocol = {
    "plan_sha256": previous["plan_sha256"],
    "recovery_plan_sha256": sha(PLAN),
    "sources": sources,
    "environment": environment(),
    "provenance": provenance(),
    "cells": original.CELLS,
    "token_record": previous["token_record"],
    "token_file_sha256": previous["token_file_sha256"],
    "origins": {
        cell: (ROOT if cell in RECOVERY_CELLS else old).as_posix() for cell in original.CELLS
    },
    "preserved_h074_worker_hashes": preserved,
    "h074_final_sha256": before["h074_final_sha256"],
    "h074_failure_sha256": before["h074_failure_sha256"],
    "original_protocol_sha256": sha(old / "protocol.json"),
    "new_optimizer_updates": 80,
    "combined_optimizer_updates": 420,
    "corpus_targets": 0,
}
write_new(ROOT / "protocol.json", protocol)
assert sha(old / "tokens.pt") == protocol["token_file_sha256"]
with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as z:
    for n in sources:
        z.write(n, n)
    z.write(PLAN, PLAN.as_posix())
    z.write(original.PLAN, original.PLAN.as_posix())


def execute(label, args):
    path = ROOT / "processes" / (label + ".json")
    logpath = path.with_suffix(".log")
    command = [sys.executable, "-X", "faulthandler", *args]
    record = {
        "status": "RUNNING",
        "command": command,
        "coordinator_pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha(ROOT / "protocol.json"),
        "source_archive_sha256": sha(ROOT / "source.zip"),
    }
    write_new(path, record)
    start = time.perf_counter()
    with logpath.open("x", encoding="utf-8") as log:
        child = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"),
        )
        record["pid"] = child.pid
        write_json(path, record)
        print(json.dumps({"label": label, "pid": child.pid, "status": "STARTED"}), flush=True)
        code = child.wait()
    record.update(
        status="PASS" if code == 0 else "FAIL",
        returncode=code,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter() - start,
        log_sha256=sha(logpath),
        source_unchanged=all(sha(n) == h for n, h in sources.items()),
    )
    write_json(path, record)
    print(json.dumps(record), flush=True)
    print(logpath.read_text(encoding="utf-8")[-3500:], flush=True)
    if code or not record["source_unchanged"]:
        raise SystemExit(code or 1)


terminal = {
    "status": "RUNNING",
    "coordinator_pid": os.getpid(),
    "started_utc": datetime.now(timezone.utc).isoformat(),
}
write_new(ROOT / "coordinator_status.json", terminal)
try:
    for probe in ("cpu_small", "full_1", "full_2"):
        execute(probe, ["-m", "results.ungated_resource_recovery_v1.source.probe", probe])
    for cell in RECOVERY_CELLS:
        execute(
            cell,
            [
                "-m",
                "results.ungated_resource_recovery_v1.source.recovery",
                "worker",
                "--cell",
                cell,
            ],
        )
    execute("finish", ["-m", "results.ungated_resource_recovery_v1.source.recovery", "finish"])
except BaseException as exc:
    terminal.update(status="FAIL", exception_type=type(exc).__name__, exception=str(exc))
    raise
else:
    terminal["status"] = "PASS"
finally:
    terminal["finished_utc"] = datetime.now(timezone.utc).isoformat()
    terminal["source_unchanged"] = all(sha(n) == h for n, h in sources.items())
    write_json(ROOT / "coordinator_status.json", terminal)
