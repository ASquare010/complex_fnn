"""Freeze and run H069 once in a hidden coordinator; retain failures."""

import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from results.factor_balance_v1.source.diagnosis import PLAN, ROOT, read, write_new
from src.core.reproducibility import environment, provenance, sha256, write_json

before = read(ROOT / "before.json")
assert not (ROOT / "protocol.json").exists()
assert (
    sha256(Path("results/verification/token_activation_fit_final_v1.json"))
    == before["h068_final_sha256"]
)
assert all(sha256(Path(n)) == h for n, h in before["h068_sources"].items())
sources = dict(before["h068_sources"])
sources.update({p.as_posix(): sha256(p) for p in (ROOT / "source").glob("*.py")})
assert len(sources) == len(before["h068_sources"]) + 3
protocol = {
    "plan_sha256": sha256(PLAN),
    "sources": sources,
    "checkpoints": before["checkpoints"],
    "environment": environment(),
    "provenance": provenance(),
    "projection_records": 216,
    "paired_ffn_cases": 24,
    "harness_checks": 4,
    "optimizer_updates": 0,
    "corpus_targets": 0,
}
write_new(ROOT / "protocol.json", protocol)
with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as z:
    for n in sources:
        z.write(n, n)
    z.write(PLAN, PLAN.as_posix())


def execute(label, args):
    command = [sys.executable, "-X", "faulthandler", *args]
    path, logpath = ROOT / (label + "_process.json"), ROOT / (label + ".log")
    record = {
        "status": "RUNNING",
        "command": command,
        "coordinator_pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(ROOT / "protocol.json"),
        "source_zip_sha256": sha256(ROOT / "source.zip"),
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
        code = child.wait()
    record.update(
        status="PASS" if code == 0 else "FAIL",
        returncode=code,
        elapsed_seconds=time.perf_counter() - start,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        source_unchanged=all(sha256(Path(n)) == h for n, h in sources.items()),
        log_sha256=sha256(logpath),
    )
    write_json(path, record)
    print(json.dumps(record), flush=True)
    if code or not record["source_unchanged"]:
        raise SystemExit(code or 1)


execute(
    "harness",
    ["-m", "pytest", "-p", "no:anyio", "-q", "--tb=short", str(ROOT / "source/test_balance.py")],
)
execute("diagnosis", ["-m", "results.factor_balance_v1.source.diagnosis"])
