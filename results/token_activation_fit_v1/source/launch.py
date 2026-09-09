"""Freeze, check and run H068 once with durable terminal process records."""

import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from results.token_activation_fit_v1.source.study import PLAN, ROOT, write_new
from src.core.reproducibility import environment, provenance, sha256, write_json

before = json.loads((ROOT / "before.json").read_text(encoding="utf-8"))
assert not (ROOT / "protocol.json").exists()
assert sha256(Path("results/token_activation_v1/result.json")) == before["h067_result_sha256"]
assert (
    sha256(Path("results/verification/token_activation_final_v1.json"))
    == before["h067_final_sha256"]
)
assert all(sha256(Path(n)) == h for n, h in before["h067_sources"].items())
sources = dict(before["h067_sources"])
sources.update({p.as_posix(): sha256(p) for p in (ROOT / "source").glob("*.py")})
assert len(sources) == len(before["h067_sources"]) + 3
protocol = {
    "plan_sha256": sha256(PLAN),
    "sources": sources,
    "environment": environment(),
    "provenance": provenance(),
    "cells": 252,
    "updates_per_cell": 300,
    "corpus_targets": 0,
    "optimizer_updates": 75600,
    "harness_checks": 4,
    "h067_result_sha256": before["h067_result_sha256"],
}
write_new(ROOT / "protocol.json", protocol)
with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as z:
    for name in sources:
        z.write(name, name)
    z.write(PLAN, PLAN.as_posix())


def execute(label, args):
    command = [sys.executable, "-X", "faulthandler", *args]
    process_path, log_path = ROOT / (label + "_process.json"), ROOT / (label + ".log")
    record = {
        "command": command,
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(ROOT / "protocol.json"),
        "source_zip_sha256": sha256(ROOT / "source.zip"),
    }
    write_new(process_path, record)
    start = time.perf_counter()
    with log_path.open("x", encoding="utf-8") as log:
        child = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"),
        )
        record["pid"] = child.pid
        write_json(process_path, record)
        print(json.dumps({"label": label, "pid": child.pid, "status": "STARTED"}), flush=True)
        code = child.wait()
    record.update(
        status="PASS" if code == 0 else "FAIL",
        returncode=code,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter() - start,
        log_sha256=sha256(log_path),
        source_unchanged=all(sha256(Path(n)) == h for n, h in sources.items()),
    )
    write_json(process_path, record)
    print(json.dumps(record), flush=True)
    print(log_path.read_text(encoding="utf-8")[-4000:], flush=True)
    if code or not record["source_unchanged"]:
        raise SystemExit(code or 1)


execute(
    "harness",
    ["-m", "pytest", "-p", "no:anyio", "-q", "--tb=short", str(ROOT / "source/test_harness.py")],
)
execute("fitting", ["-m", "results.token_activation_fit_v1.source.study"])
