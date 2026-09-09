"""Durable one-shot analysis/report launcher; invoke only after H085 training ends."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("results/latent_offset_fit_v1")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value, exclusive=True):
    payload = (json.dumps(value, indent=2) + "\n").encode()
    with path.open("xb" if exclusive else "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    assert path.read_bytes() == payload


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("phase", choices=("analysis", "report"))
args = parser.parse_args()
assert json.loads((ROOT / "coordinator_status.json").read_bytes())["status"] == "PASS"
if args.phase == "report":
    assert json.loads((ROOT / "analysis_process.json").read_bytes())["status"] == "PASS"
source = Path(f"results/verification/latent_offset_fit_{args.phase}_v1.py")
sources = json.loads((ROOT / "protocol.json").read_bytes())["sources"]
assert all(sha(n) == h for n, h in sources.items())
record_path = ROOT / (args.phase + "_process.json")
log_path = ROOT / (args.phase + ".log")
command = [sys.executable, "-X", "faulthandler", "-m", f"results.verification.latent_offset_fit_{args.phase}_v1"]
record = {"status": "RUNNING", "coordinator_pid": os.getpid(), "command": command,
          "source_sha256": sha(source), "started_utc": datetime.now(timezone.utc).isoformat(),
          "training_repetitions": 0, "optimizer_updates": 0}
write(record_path, record)
started = time.perf_counter()
with log_path.open("xb") as log:
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                             env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"))
    record["pid"] = child.pid
    write(record_path, record, exclusive=False)
    code = child.wait()
    log.flush()
    os.fsync(log.fileno())
record.update(status="PASS" if code == 0 else "FAIL", returncode=code,
              finished_utc=datetime.now(timezone.utc).isoformat(),
              elapsed_seconds=time.perf_counter()-started, log_sha256=sha(log_path),
              source_unchanged=sha(source) == record["source_sha256"],
              scientific_sources_unchanged=all(sha(n) == h for n, h in sources.items()))
write(record_path, record, exclusive=False)
print(json.dumps(record), flush=True)
print(log_path.read_text(encoding="utf-8")[-3000:], flush=True)
raise SystemExit(code or int(not record["source_unchanged"] or not record["scientific_sources_unchanged"]))
