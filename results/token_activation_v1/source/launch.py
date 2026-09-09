"""Freeze and execute H067's 21 checks once; retain process and scientific verdicts."""

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

ROOT = Path("results/token_activation_v1")
PLAN = Path("research/token_activation_plan.md")
THEORY = Path("research/token_activation_theory.md")
EXPECTED = "c493d8fb93a2bc7db9e3839e9e7f2438754f37201f9639e24d9ea0a2032d7191"
assert not (ROOT / "protocol.json").exists()
before = read(ROOT / "before.json")
assert sha(PLAN) == before["plan_sha256"] == EXPECTED
assert sha("results/verification/staged_resource_final_v1.json") == before["h066_final_sha256"]
assert all(sha(n) == h for n, h in before["h066_source_hashes"].items())
sources = dict(before["h066_source_hashes"])
sources.update({p.as_posix(): sha(p) for p in (ROOT / "source").glob("*.py")})
sources[THEORY.as_posix()] = sha(THEORY)
assert len(sources) == len(before["h066_source_hashes"]) + 4
protocol = {
    "plan_sha256": EXPECTED,
    "theory_sha256": sha(THEORY),
    "sources": sources,
    "environment": environment(),
    "qualification_checks": 21,
    "optimizer_updates": 0,
    "corpus_targets": 0,
    "new_active_variants": 0,
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
    "pytest",
    "-p",
    "no:anyio",
    "-q",
    "--tb=short",
    str(ROOT / "source/test_qualification.py"),
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
checks = {p.stem: read(p) for p in (ROOT / "checks").glob("*.json")}
passed = code == 0 and record["source_unchanged"] and len(checks) == 21
result = {
    "status": "complete" if len(checks) == 21 else "incomplete_qualification",
    "scientific_verdict": "LOCALLY_QUALIFIED" if passed else "REJECTED_OR_INCOMPLETE",
    "checks": checks,
    "all_checks_pass": passed,
    "plan_sha256": EXPECTED,
    "theory_sha256": sha(THEORY),
    "optimizer_updates": 0,
    "corpus_targets": 0,
    "full_model_resource_workers": 0,
    "earns_fitting_resource_comparison": passed,
    "research_goal_achieved": False,
}
write_new(ROOT / "result.json", result)
print(json.dumps(record), flush=True)
print((ROOT / "process.log").read_text(encoding="utf-8")[-6500:], flush=True)
print(json.dumps({k: v for k, v in result.items() if k != "checks"}), flush=True)
raise SystemExit(code or (0 if passed else 1))
