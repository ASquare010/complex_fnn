"""Freeze and execute H070 once; keep the 21-check result separate from promotion."""

import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from src.core.reproducibility import environment, provenance, sha256, write_json

ROOT = Path("results/rotated_shuffle_v1")
PLAN = Path("research/rotated_shuffle_plan.md")
THEORY = Path("research/rotated_shuffle_theory.md")


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def write_new(p, value):
    assert not p.exists(), p
    write_json(p, value)


before = read(ROOT / "before.json")
assert not (ROOT / "protocol.json").exists()
assert (
    sha256(Path("results/verification/factor_balance_final_v1.json")) == before["h069_final_sha256"]
)
assert all(sha256(Path(n)) == h for n, h in before["h069_sources"].items())
sources = dict(before["h069_sources"])
sources.update({p.as_posix(): sha256(p) for p in (ROOT / "source").glob("*.py")})
sources[THEORY.as_posix()] = sha256(THEORY)
assert len(sources) == len(before["h069_sources"]) + 4
protocol = {
    "plan_sha256": sha256(PLAN),
    "theory_sha256": sha256(THEORY),
    "sources": sources,
    "environment": environment(),
    "provenance": provenance(),
    "qualification_checks": 21,
    "optimizer_updates": 0,
    "corpus_targets": 0,
    "new_active_variants": 0,
}
write_new(ROOT / "protocol.json", protocol)
with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as z:
    for n in sources:
        z.write(n, n)
    z.write(PLAN, PLAN.as_posix())
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
    "coordinator_pid": os.getpid(),
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "protocol_sha256": sha256(ROOT / "protocol.json"),
    "source_zip_sha256": sha256(ROOT / "source.zip"),
}
write_new(ROOT / "process.json", record)
start = time.perf_counter()
with (ROOT / "process.log").open("x", encoding="utf-8") as log:
    child = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"),
    )
    record["pid"] = child.pid
    write_json(ROOT / "process.json", record)
    code = child.wait()
record.update(
    status="PASS" if code == 0 else "FAIL",
    returncode=code,
    elapsed_seconds=time.perf_counter() - start,
    finished_utc=datetime.now(timezone.utc).isoformat(),
    source_unchanged=all(sha256(Path(n)) == h for n, h in sources.items()),
    log_sha256=sha256(ROOT / "process.log"),
)
write_json(ROOT / "process.json", record)
checks = {p.stem: read(p) for p in (ROOT / "checks").glob("*.json")}
passed = (
    code == 0
    and record["source_unchanged"]
    and len(checks) == 21
    and all(r["passed"] for r in checks.values())
)
result = {
    "status": "complete" if len(checks) == 21 else "incomplete_qualification",
    "scientific_verdict": "LOCALLY_QUALIFIED" if passed else "REJECTED_OR_INCOMPLETE",
    "checks": checks,
    "all_checks_pass": passed,
    "plan_sha256": sha256(PLAN),
    "theory_sha256": sha256(THEORY),
    "optimizer_updates": 0,
    "corpus_targets": 0,
    "full_model_resource_workers": 0,
    "earns_separate_fitting_resource_comparison": passed,
    "research_goal_achieved": False,
}
write_new(ROOT / "result.json", result)
print(json.dumps(record), flush=True)
print(json.dumps({k: v for k, v in result.items() if k != "checks"}), flush=True)
raise SystemExit(code or (0 if passed else 1))
