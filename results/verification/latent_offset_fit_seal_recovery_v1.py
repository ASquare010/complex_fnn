"""One explicitly recorded administrative seal retry; no numerical work."""

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("results/latent_offset_fit_v1")
RECORD = Path("results/verification/latent_offset_fit_seal_recovery_process_v1.json")
LOG = Path("results/verification/latent_offset_fit_seal_recovery_v1.log")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(value, exclusive):
    data = (json.dumps(value, indent=2)+"\n").encode()
    with RECORD.open("xb" if exclusive else "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    assert RECORD.read_bytes() == data


plan = json.loads((ROOT / "finalization_recovery_plan.json").read_bytes())
source = Path("results/verification/latent_offset_fit_final_recovery_v1.py")
assert sha(source) == plan["finalizer_source_sha256"]
assert not (ROOT / "support_recovery_v1.zip").exists() and not (ROOT / "evidence_metadata_recovery_v1.zip").exists()
assert sha(ROOT / "evidence_metadata.zip") == plan["original_metadata_archive_sha256"]
command = [sys.executable, "-X", "faulthandler", "-m", "results.verification.latent_offset_fit_final_recovery_v1"]
record = {"status": "RUNNING", "coordinator_pid": os.getpid(), "command": command,
          "started_utc": datetime.now(timezone.utc).isoformat(), "administrative_repetition": 1,
          "scientific_retries": 0, "optimizer_updates": 0, "source_sha256": sha(source),
          "recovery_plan_sha256": sha(ROOT / "finalization_recovery_plan.json")}
write(record, True)
started = time.perf_counter()
with LOG.open("xb") as log:
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    record["pid"] = child.pid
    write(record, False)
    code = child.wait()
    log.flush()
    os.fsync(log.fileno())
record.update(status="PASS" if code == 0 else "FAIL", returncode=code,
              finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-started,
              log_sha256=sha(LOG), source_unchanged=sha(source) == record["source_sha256"])
write(record, False)
print(json.dumps(record), flush=True)
print(LOG.read_text(encoding="utf-8")[-3500:], flush=True)
raise SystemExit(code or int(not record["source_unchanged"]))
