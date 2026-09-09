"""Run H073 read-only postprocessing under a durable hidden coordinator."""

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("results/ungated_fit_v1")
phase = sys.argv[1]
assert phase in ("analysis", "report")
source = Path(f"results/verification/ungated_fit_{phase}_v1.py")
record_path, log_path = ROOT / f"{phase}_process.json", ROOT / f"{phase}.log"
assert not record_path.exists() and not log_path.exists()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(record):
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
assert all(sha(n) == h for n, h in protocol["sources"].items())
command = [
    sys.executable,
    "-X",
    "faulthandler",
    "-m",
    f"results.verification.ungated_fit_{phase}_v1",
]
record = {
    "status": "RUNNING",
    "command": command,
    "coordinator_pid": os.getpid(),
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "source_sha256": sha(source),
    "scientific_fitting_process_sha256": sha(ROOT / "fitting_process.json"),
    "scientific_reruns": 0,
}
save(record)
post_env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
if phase == "report":
    config_dir = ROOT / "report_mplconfig"
    config_dir.mkdir(exist_ok=True)
    post_env["MPLCONFIGDIR"] = str(config_dir.resolve())
start = time.perf_counter()
with log_path.open("x", encoding="utf-8") as log:
    child = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=post_env,
    )
    record["pid"] = child.pid
    save(record)
    code = child.wait()
record.update(
    status="PASS" if code == 0 else "FAIL",
    returncode=code,
    finished_utc=datetime.now(timezone.utc).isoformat(),
    elapsed_seconds=time.perf_counter() - start,
    log_sha256=sha(log_path),
    postprocess_source_unchanged=sha(source) == record["source_sha256"],
    scientific_source_unchanged=all(sha(n) == h for n, h in protocol["sources"].items()),
)
save(record)
print(json.dumps(record), flush=True)
raise SystemExit(code)
