"""One explicit logging-corrected H069 attempt with the original scientific code."""

import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from results.factor_balance_v1.source import diagnosis as study
from results.factor_balance_v2.source.logging_fix import tensor_hash
from src.core.reproducibility import sha256, write_json

ROOT = Path("results/factor_balance_v2")
OLD = Path("results/factor_balance_v1")
FIX_PLAN = Path("research/factor_balance_logging_plan.md")

if "--worker" in sys.argv:
    study.ROOT = ROOT
    study.tensor_hash = tensor_hash
    study.run()
    old = study.read(OLD / "checks/s17_0_l0_cpu.json")
    new = study.read(ROOT / "checks/s17_0_l0_cpu.json")
    assert old["comparisons"] == new["comparisons"] and old["input_sha256"] == new["input_sha256"]
    study.write_new(ROOT / "first_cpu_reproduction.json", {"original_cpu_case_exact": True})
    raise SystemExit(0)

before = study.read(ROOT / "before.json")
assert not (ROOT / "protocol.json").exists()
assert sha256(OLD / "protocol.json") == before["original_protocol_sha256"]
assert sha256(OLD / "diagnosis_process.json") == before["original_failure_sha256"]
assert study.read(OLD / "harness_process.json")["status"] == "PASS"
protocol = study.read(OLD / "protocol.json")
assert all(sha256(Path(n)) == h for n, h in protocol["sources"].items())
sources = dict(protocol["sources"])
sources.update({p.as_posix(): sha256(p) for p in (ROOT / "source").glob("*.py")})
sources[FIX_PLAN.as_posix()] = sha256(FIX_PLAN)
protocol.update(
    sources=sources,
    logging_plan_sha256=sha256(FIX_PLAN),
    logging_checks=4,
    original_protocol_sha256=before["original_protocol_sha256"],
    original_failure_sha256=before["original_failure_sha256"],
    corrected_attempt=1,
    output_root=ROOT.as_posix(),
)
study.write_new(ROOT / "protocol.json", protocol)
with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as z:
    for n in sources:
        z.write(n, n)
    z.write(study.PLAN, study.PLAN.as_posix())


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
    study.write_new(path, record)
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
    "logging",
    ["-m", "pytest", "-p", "no:anyio", "-q", "--tb=short", str(ROOT / "source/test_logging.py")],
)
execute("diagnosis", ["-m", "results.factor_balance_v2.source.launch", "--worker"])
