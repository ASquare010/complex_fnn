"""Freeze H081 sources, qualify once, then execute 24 GPU cells sequentially."""

import json
import os
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone

from results.blast_learning_screen_v1.source.study import CACHE, CELLS, PLAN, ROOT, configuration
from results.blast_operator_recovery_v1.source.storage import read, sha, write_json
from src.core.reproducibility import environment, provenance

before = read(ROOT / "before.json")
assert not (ROOT / "protocol.json").exists()
assert sha(PLAN) == before["plan_sha256"]
assert sha("results/verification/blast_operator_recovery_final_v1.json") == before["h080_final_sha256"]
assert sha("results/blast_operator_recovery_v1/source.zip") == before["h080_source_sha256"]
assert sha("results/blast_operator_recovery_v1/result.json") == before["h080_result_sha256"]
assert all(sha(n) == h for n, h in before["prior_sources"].items())
assert all(sha(CACHE / n) == h for n, h in before["data_files"].items())
assert sha(CACHE / "manifest.json") == before["data_manifest_sha256"]
sources = {**before["prior_sources"], **{p.as_posix(): sha(p) for p in (ROOT / "source").glob("*.py")}}
assert len(sources) == 121
protocol = {
    "plan_sha256": sha(PLAN), "sources": sources, "environment": environment(), "provenance": provenance(),
    "cells": CELLS, "configurations": {cell: dict(zip(("model", "training"), map(asdict, configuration(**spec))))
                                      for cell, spec in CELLS.items()},
    "data_files": before["data_files"], "data_manifest_sha256": before["data_manifest_sha256"],
    "h080_final_sha256": before["h080_final_sha256"], "language_trials": 24,
    "language_optimizer_updates": 4800, "language_training_targets": 9830400,
    "shared_validation_target_exposures": 54211584, "preflight_checks": 9,
    "qualification_optimizer_updates": 24, "qualification_cpu_targets": 384,
    "qualification_cuda_targets": 24576, "scientific_retries": 0, "official_test_scored": False,
}
write_json(ROOT / "protocol.json", protocol)
with (ROOT / "source.zip").open("xb") as handle:
    with zipfile.ZipFile(handle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(name, name)
        archive.write(PLAN, PLAN.as_posix())
    handle.flush()
    os.fsync(handle.fileno())
with zipfile.ZipFile(ROOT / "source.zip") as archive:
    assert archive.testzip() is None


def execute(label, args):
    path = ROOT / "processes" / (label + ".json")
    logpath = path.with_suffix(".log")
    command = [sys.executable, "-X", "faulthandler", *args]
    record = {"status": "RUNNING", "command": command, "coordinator_pid": os.getpid(),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": sha(ROOT / "protocol.json"), "source_archive_sha256": sha(ROOT / "source.zip")}
    write_json(path, record)
    started = time.perf_counter()
    with logpath.open("xb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"))
        record["pid"] = child.pid
        write_json(path, record, exclusive=False)
        print(json.dumps({"label": label, "pid": child.pid, "status": "STARTED"}), flush=True)
        code = child.wait()
        log.flush()
        os.fsync(log.fileno())
    record.update(status="PASS" if code == 0 else "FAIL", returncode=code,
                  finished_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.perf_counter() - started, log_sha256=sha(logpath),
                  source_unchanged=all(sha(n) == h for n, h in sources.items()))
    write_json(path, record, exclusive=False)
    print(json.dumps(record), flush=True)
    print(logpath.read_text(encoding="utf-8")[-2000:], flush=True)
    if code or not record["source_unchanged"]:
        raise SystemExit(code or 1)


terminal = {"status": "RUNNING", "coordinator_pid": os.getpid(),
            "started_utc": datetime.now(timezone.utc).isoformat()}
write_json(ROOT / "coordinator_status.json", terminal)
try:
    execute("preflight", ["-m", "pytest", "-p", "no:anyio", "-q", "--tb=short",
                          str(ROOT / "source/test_harness.py")])
    preflight = list((ROOT / "preflight").glob("*.json"))
    assert len(preflight) == 9
    assert sum(read(p)["optimizer_updates"] for p in preflight) == 24
    for cell in CELLS:
        execute(cell, ["-m", "results.blast_learning_screen_v1.source.study", "worker", "--cell", cell])
    execute("finish", ["-m", "results.blast_learning_screen_v1.source.study", "finish"])
except BaseException as exc:
    terminal.update(status="FAIL", exception_type=type(exc).__name__, exception=str(exc))
    raise
else:
    terminal["status"] = "PASS"
finally:
    terminal.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                    source_unchanged=all(sha(n) == h for n, h in sources.items()))
    write_json(ROOT / "coordinator_status.json", terminal, exclusive=False)
