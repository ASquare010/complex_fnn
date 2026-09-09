"""Launch one independent H090 audit with durable terminal evidence."""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from results.blast_operator_recovery_v1.source.storage import read, sha, write_json

root = Path("results/neuron_geometry_audit_v1")
before = read(root / "before.json")
assert all(sha(p) == h for p, h in before["source_hashes"].items())
assert all(sha(p) == h for p, h in before["anchors"].items())
assert read("results/neuron_geometry_recovery_v1/coordinator_status.json")["status"] == "PASS"
record = {
    "status": "RUNNING",
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "coordinator_pid": os.getpid(),
    "before_sha256": sha(root / "before.json"),
}
command = [
    sys.executable,
    "-X",
    "faulthandler",
    "-m",
    "results.neuron_geometry_audit_v1.source.audit",
]
record["command"] = command
write_json(root / "process.json", record)
started = time.perf_counter()
with (root / "audit.log").open("xb") as stream:
    child = subprocess.Popen(
        command,
        stdout=stream,
        stderr=subprocess.STDOUT,
        env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"),
    )
    record["pid"] = child.pid
    write_json(root / "process.json", record, exclusive=False)
    code = child.wait()
    stream.flush()
    os.fsync(stream.fileno())
record.update(
    status="PASS" if code == 0 else "FAIL",
    returncode=code,
    elapsed_seconds=time.perf_counter() - started,
    finished_utc=datetime.now(timezone.utc).isoformat(),
    log_sha256=sha(root / "audit.log"),
)
write_json(root / "process.json", record, exclusive=False)
assert code == 0, (root / "audit.log").read_text()[-12000:]
