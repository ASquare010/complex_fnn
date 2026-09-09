"""Freeze and launch H066's six fresh sequential workers, preserving every outcome."""

import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from results.staged_resource_v1.source.study import CELLS, PLAN, PLAN_SHA, ROOT
from src.core.native_recompute_audit import read, sha, tensor_record, tokens, write_new
from src.core.reproducibility import environment

assert not (ROOT / "protocol.json").exists()
before = read(ROOT / "before.json")
assert sha(PLAN) == before["plan_sha256"] == PLAN_SHA
assert sha("results/verification/rational_staged_final_v1.json") == before["h065_final_sha256"]
assert all(sha(n) == h for n, h in before["h065_source_hashes"].items())
for row in before["inputs"].values():
    assert all(sha(Path("results/runs") / row["run"] / n) == h for n, h in row["files"].items())
sources = dict(before["h065_source_hashes"])
sources.update({p.as_posix(): sha(p) for p in (ROOT / "source").glob("*.py")})
assert len(sources) == len(before["h065_source_hashes"]) + 2
protocol = {
    "plan_sha256": PLAN_SHA,
    "provenance": {"source_files": sources},
    "inputs": before["inputs"],
    "environment": environment(),
    "cells": CELLS,
    "token_record": tensor_record(tokens()),
    "synthetic_updates": 120,
    "synthetic_training_targets": 245760,
    "initial_probe_targets": 12288,
    "corpus_scoring": False,
}
write_new(ROOT / "protocol.json", protocol)
with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
    for n in sources:
        archive.write(n, n)
    archive.write(PLAN, PLAN.as_posix())
write_new(
    ROOT / "preflight.json",
    {
        "status": "PASS",
        "protocol_sha256": sha(ROOT / "protocol.json"),
        "h065_qualification_preserved": True,
        "existing_sources_unchanged": True,
    },
)


def run(label, args):
    path = ROOT / "processes" / f"{label}.json"
    logpath = path.with_suffix(".log")
    assert not path.exists() and not logpath.exists()
    command = [
        sys.executable,
        "-X",
        "faulthandler",
        "-m",
        "results.staged_resource_v1.source.study",
        *args,
    ]
    record = {
        "status": "RUNNING",
        "command": command,
        "protocol_sha256": sha(ROOT / "protocol.json"),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_new(path, record)
    start = time.perf_counter()
    with logpath.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"),
        )
        record["pid"] = child.pid
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"phase": label, "status": "STARTED", "pid": child.pid}), flush=True)
        code = child.wait()
    record.update(
        status="PASS" if code == 0 else "FAIL",
        returncode=code,
        elapsed_seconds=time.perf_counter() - start,
        source_unchanged=all(sha(n) == h for n, h in sources.items()),
        log_sha256=sha(logpath),
    )
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "command"}), flush=True)
    print(logpath.read_text(encoding="utf-8")[-4500:], flush=True)
    if code or not record["source_unchanged"]:
        raise SystemExit(code or 1)


for cell in CELLS:
    run(cell, ["worker", "--cell", cell])
run("finish", ["finish"])
