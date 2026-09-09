"""Durable H089 diagnostic launcher; never retries."""

import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from results.blast_operator_recovery_v1.source.storage import read, sha, write_json
from src.core.reproducibility import environment

root = Path("results/neuron_precision_v1")
before = read(root / "before.json")
assert all(sha(name) == value for name, value in before["source_hashes"].items())
write_json(
    root / "protocol.json",
    {**before, "environment": environment(), "before_sha256": sha(root / "before.json")},
)
with (root / "source.zip").open("xb") as stream:
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in before["source_hashes"]:
            archive.write(name, name)
    stream.flush()
    os.fsync(stream.fileno())
command = [
    sys.executable,
    "-X",
    "faulthandler",
    "-m",
    "results.neuron_precision_v1.source.diagnose",
]
record = {
    "status": "RUNNING",
    "coordinator_pid": os.getpid(),
    "command": command,
    "started_utc": datetime.now(timezone.utc).isoformat(),
}
write_json(root / "process.json", record)
started = time.perf_counter()
with (root / "diagnostic.log").open("xb") as stream:
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
    finished_utc=datetime.now(timezone.utc).isoformat(),
    elapsed_seconds=time.perf_counter() - started,
    log_sha256=sha(root / "diagnostic.log"),
)
write_json(root / "process.json", record, exclusive=False)
assert code == 0, (root / "diagnostic.log").read_text()[-12000:]
