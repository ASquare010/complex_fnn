"""Freeze and run H076's five integration checks and 21 sequential language cells once."""

import json
import os
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone

from results.ungated_lm_screen_v1.source.study import CACHE, CELLS, PLAN, ROOT, configuration
from src.core.native_recompute_audit import read, sha, write_new
from src.core.reproducibility import environment, provenance, write_json

before = read(ROOT / "before.json")
assert not (ROOT / "protocol.json").exists()
assert sha(PLAN) == before["plan_sha256"]
assert (
    sha("results/verification/ungated_resource_recovery_final_v1.json")
    == before["h075_final_sha256"]
)
assert sha("results/ungated_resource_recovery_v1/result.json") == before["h075_result_sha256"]
assert all(sha(n) == h for n, h in before["h075_sources"].items())
assert all(sha(CACHE / n) == h for n, h in before["data_files"].items())
assert sha(CACHE / "manifest.json") == before["data_manifest_sha256"]
sources = dict(before["h075_sources"])
sources.update({p.as_posix(): sha(p) for p in (ROOT / "source").glob("*.py")})
assert len(sources) == 106
configs = {
    cell: dict(zip(("model", "training"), map(asdict, configuration(spec["form"], spec["rate"]))))
    for cell, spec in CELLS.items()
}
protocol = {
    "plan_sha256": sha(PLAN),
    "sources": sources,
    "environment": environment(),
    "provenance": provenance(),
    "cells": CELLS,
    "configurations": configs,
    "data_files": before["data_files"],
    "data_manifest_sha256": before["data_manifest_sha256"],
    "h075_final_sha256": before["h075_final_sha256"],
    "language_trials": 21,
    "language_training_targets": 8601600,
    "language_optimizer_updates": 4200,
    "shared_validation_target_exposures": 47435136,
    "harness_checks": 5,
    "qualification_cpu_updates": 8,
    "qualification_cpu_training_targets": 256,
    "official_test_scored": False,
}
write_new(ROOT / "protocol.json", protocol)
with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as z:
    for n in sources:
        z.write(n, n)
    z.write(PLAN, PLAN.as_posix())


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
    started = time.perf_counter()
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
        elapsed_seconds=time.perf_counter() - started,
        log_sha256=sha(logpath),
        source_unchanged=all(sha(n) == h for n, h in sources.items()),
    )
    write_json(path, record)
    print(json.dumps(record), flush=True)
    print(logpath.read_text(encoding="utf-8")[-4500:], flush=True)
    if code or not record["source_unchanged"]:
        raise SystemExit(code or 1)


terminal = {
    "status": "RUNNING",
    "coordinator_pid": os.getpid(),
    "started_utc": datetime.now(timezone.utc).isoformat(),
}
write_new(ROOT / "coordinator_status.json", terminal)
try:
    execute(
        "harness",
        [
            "-m",
            "pytest",
            "-p",
            "no:anyio",
            "-q",
            "--tb=short",
            str(ROOT / "source/test_harness.py"),
        ],
    )
    for cell in CELLS:
        execute(cell, ["-m", "results.ungated_lm_screen_v1.source.study", "worker", "--cell", cell])
    execute("finish", ["-m", "results.ungated_lm_screen_v1.source.study", "finish"])
except BaseException as exc:
    terminal.update(status="FAIL", exception_type=type(exc).__name__, exception=str(exc))
    raise
else:
    terminal["status"] = "PASS"
finally:
    terminal["finished_utc"] = datetime.now(timezone.utc).isoformat()
    terminal["source_unchanged"] = all(sha(n) == h for n, h in sources.items())
    write_json(ROOT / "coordinator_status.json", terminal)
