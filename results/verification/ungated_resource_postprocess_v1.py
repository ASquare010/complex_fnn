"""Durable H074 postprocessing; never reruns scientific workers."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

root = Path("results/ungated_resource_v1")
parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=("analysis", "report"))
phase = parser.parse_args().phase


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


assert read(root / "result.json")["status"] == "complete"
assert read(root / "processes/finish.json")["status"] == "PASS"
protocol = read(root / "protocol.json")
assert all(sha(n) == h for n, h in protocol["sources"].items())
source = Path(f"results/verification/ungated_resource_{phase}_v1.py")
record_path, log_path = root / (phase + "_process.json"), root / (phase + ".log")
assert not record_path.exists() and not log_path.exists()
record = {
    "status": "RUNNING",
    "source_sha256": sha(source),
    "coordinator_pid": os.getpid(),
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "scientific_reruns": 0,
    "scientific_result_sha256": sha(root / "result.json"),
    "scientific_finish_process_sha256": sha(root / "processes/finish.json"),
}
record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
if phase == "report":
    env["MPLCONFIGDIR"] = str((root / "report_mplconfig").resolve())
start = time.perf_counter()
with log_path.open("x", encoding="utf-8") as log:
    child = subprocess.Popen(
        [
            sys.executable,
            "-X",
            "faulthandler",
            "-m",
            f"results.verification.ungated_resource_{phase}_v1",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    record["pid"] = child.pid
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    code = child.wait()
record.update(
    status="PASS" if code == 0 else "FAIL",
    returncode=code,
    finished_utc=datetime.now(timezone.utc).isoformat(),
    elapsed_seconds=time.perf_counter() - start,
    source_unchanged=sha(source) == record["source_sha256"],
    scientific_source_unchanged=all(sha(n) == h for n, h in protocol["sources"].items()),
    log_sha256=sha(log_path),
)
record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record), flush=True)
raise SystemExit(code)
