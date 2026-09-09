"""Freeze and run H074 once, retaining terminal records for every process."""

import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone

import torch

from results.ungated_resource_v1.source.study import CELLS, PLAN, ROOT
from src.core.native_recompute_audit import read, sha, tensor_record, tokens, write_new
from src.core.reproducibility import environment, provenance, write_json

before = read(ROOT / "before.json")
assert not (ROOT / "protocol.json").exists()
assert sha(PLAN) == before["plan_sha256"]
assert sha("results/verification/ungated_fit_final_v1.json") == before["h073_final_sha256"]
assert sha("results/ungated_fit_v1/result.json") == before["h073_result_sha256"]
assert all(sha(n) == h for n, h in before["h073_sources"].items())
sources = dict(before["h073_sources"])
sources.update({p.as_posix(): sha(p) for p in (ROOT / "source").glob("*.py")})
assert len(sources) == 100
stream = tokens()
torch.save(stream, ROOT / "tokens.pt")
protocol = {
    "plan_sha256": sha(PLAN),
    "sources": sources,
    "environment": environment(),
    "provenance": provenance(),
    "cells": CELLS,
    "token_record": tensor_record(stream),
    "token_file_sha256": sha(ROOT / "tokens.pt"),
    "synthetic_updates": 420,
    "synthetic_training_targets": 860160,
    "initial_probe_targets": 43008,
    "corpus_targets": 0,
    "harness_checks": 5,
    "h073_final_sha256": before["h073_final_sha256"],
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
    print(logpath.read_text(encoding="utf-8")[-3000:], flush=True)
    if code or not record["source_unchanged"]:
        raise SystemExit(code or 1)


execute(
    "harness",
    ["-m", "pytest", "-p", "no:anyio", "-q", "--tb=short", str(ROOT / "source/test_harness.py")],
)
for cell in CELLS:
    execute(cell, ["-m", "results.ungated_resource_v1.source.study", "worker", "--cell", cell])
execute("finish", ["-m", "results.ungated_resource_v1.source.study", "finish"])
