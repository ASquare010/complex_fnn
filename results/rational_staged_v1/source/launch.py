"""Freeze H065 qualification inputs and run exactly one recorded child process."""

import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from src.core.native_recompute_audit import read, sha, write_new
from src.core.reproducibility import environment

ROOT = Path("results/rational_staged_v1")
PLAN = Path("research/rational_staged_plan.md")
EXPECTED_PLAN = "eecf4e93cce53c3a24c01f8f0d28f692311d4f2aa69a7751d9489178151319ad"
assert not (ROOT / "process.json").exists()
before = read(ROOT / "before.json")
assert sha(PLAN) == before["plan_sha256"] == EXPECTED_PLAN
assert sha("results/verification/rational_memory_final_v1.json") == before["h064_final_sha256"]
assert sha("results/rational_memory_v1/protocol.json") == before["h064_protocol_sha256"]
assert all(sha(n) == h for n, h in before["h064_source_hashes"].items())
source = Path("results/runs") / before["input"]["run"]
assert all(sha(source / n) == h for n, h in before["input"]["files"].items())
sources = dict(before["h064_source_hashes"])
sources.update({p.as_posix(): sha(p) for p in (ROOT / "source").glob("*.py")})
assert len(sources) == len(before["h064_source_hashes"]) + 3
protocol = {
    "plan_sha256": EXPECTED_PLAN,
    "sources": sources,
    "input": before["input"],
    "environment": environment(),
    "paired_cases": 12,
    "optimizer_updates": 0,
    "corpus_targets": 0,
    "previous_goal_turn": "PROGRESS",
}
write_new(ROOT / "protocol.json", protocol)
with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
    for name in sources:
        archive.write(name, name)
    archive.write(PLAN, PLAN.as_posix())
command = [
    sys.executable,
    "-X",
    "faulthandler",
    "-m",
    "results.rational_staged_v1.source.qualification",
]
record = {
    "status": "RUNNING",
    "command": command,
    "protocol_sha256": sha(ROOT / "protocol.json"),
    "source_archive_sha256": sha(ROOT / "source.zip"),
    "started_utc": datetime.now(timezone.utc).isoformat(),
}
write_new(ROOT / "process.json", record)
start = time.perf_counter()
with (ROOT / "process.log").open("w", encoding="utf-8") as log:
    child = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"),
    )
    record["pid"] = child.pid
    (ROOT / "process.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "STARTED", "pid": child.pid}), flush=True)
    code = child.wait()
record.update(
    status="PASS" if code == 0 else "FAIL",
    returncode=code,
    elapsed_seconds=time.perf_counter() - start,
    source_unchanged=all(sha(n) == h for n, h in sources.items()),
    log_sha256=sha(ROOT / "process.log"),
)
(ROOT / "process.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record), flush=True)
print((ROOT / "process.log").read_text(encoding="utf-8")[-6000:], flush=True)
raise SystemExit(code or (0 if record["source_unchanged"] else 1))
