"""Freeze and run H077's five integration checks and 18 sequential language cells once."""

import json
import os
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone

from results.ungated_lm_replication_v1.source.study import CACHE, CELLS, PLAN, ROOT, configuration
from src.core.native_recompute_audit import read, sha, write_new
from src.core.reproducibility import environment, provenance, write_json

before = read(ROOT / "before.json")
assert not (ROOT / "protocol.json").exists()
assert sha(PLAN) == before["plan_sha256"]
assert sha("results/verification/ungated_lm_screen_final_v1.json") == before["h076_final_sha256"]
assert sha("results/ungated_lm_screen_v1/result.json") == before["h076_result_sha256"]
assert all(sha(n) == h for n, h in before["h076_sources"].items())
assert sha("results/ungated_lm_screen_v1/source.zip") == before["h076_source_archive_sha256"]
assert all(sha(CACHE / n) == h for n, h in before["data_files"].items())
assert sha(CACHE / "manifest.json") == before["data_manifest_sha256"]
sources = dict(before["h076_sources"])
sources.update({p.as_posix(): sha(p) for p in (ROOT / "source").glob("*.py")})
assert len(sources) == 109
configs = {
    cell: dict(zip(("model", "training"), map(asdict, configuration(spec["form"], spec["seed"]))))
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
    "h076_final_sha256": before["h076_final_sha256"],
    "language_trials": 18,
    "language_training_targets": 29491200,
    "language_optimizer_updates": 14400,
    "shared_validation_target_exposures": 40658688,
    "harness_checks": 5,
    "qualification_optimizer_updates": 0,
    "qualification_unscored_sampler_target_elements": 4915200,
    "seed_values": [17, 29, 43],
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
    qualification = {n.as_posix(): sha(n) for n in (ROOT / "qualification").glob("*.json")}
    assert len(qualification) == 2
    write_new(ROOT / "qualification_manifest.json", qualification)
    for cell in CELLS:
        assert all(sha(n) == h for n, h in qualification.items())
        execute(
            cell, ["-m", "results.ungated_lm_replication_v1.source.study", "worker", "--cell", cell]
        )
    execute("finish", ["-m", "results.ungated_lm_replication_v1.source.study", "finish"])
except BaseException as exc:
    terminal.update(status="FAIL", exception_type=type(exc).__name__, exception=str(exc))
    raise
else:
    terminal["status"] = "PASS"
finally:
    terminal["finished_utc"] = datetime.now(timezone.utc).isoformat()
    terminal["source_unchanged"] = all(sha(n) == h for n, h in sources.items())
    write_json(ROOT / "coordinator_status.json", terminal)
