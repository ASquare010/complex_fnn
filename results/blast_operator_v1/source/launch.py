"""Freeze and execute the 34-check H079 qualification once."""

import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from src.core.native_recompute_audit import read, sha, write_new
from src.core.reproducibility import environment, provenance, write_json

ROOT = Path("results/blast_operator_v1")
PLAN = Path("research/blast_operator_plan.md")
THEORY = Path("research/blast_operator_theory.md")
before = read(ROOT / "before.json")
assert not (ROOT / "protocol.json").exists() and not list((ROOT / "observations").iterdir())
assert sha(PLAN) == before["plan_sha256"]
assert sha("results/verification/ungated_duration_final_v1.json") == before["h078_final_sha256"]
for filename, key in (
    ("result.json", "h078_result_sha256"),
    ("source.zip", "h078_source_archive_sha256"),
    ("support.zip", "h078_support_archive_sha256"),
):
    assert sha(Path("results/ungated_duration_v1") / filename) == before[key]
assert all(sha(n) == h for n, h in before["prior_sources"].items())
assert all(sha(n) == h for n, h in before["prior_plans"].items())
sources = dict(before["prior_sources"])
sources.update({p.as_posix(): sha(p) for p in (ROOT / "source").glob("*.py")})
assert len(sources) == 115
protocol = {
    "plan_sha256": sha(PLAN), "theory_sha256": sha(THEORY), "sources": sources,
    "environment": environment(), "provenance": provenance(),
    "h078_final_sha256": before["h078_final_sha256"], "checks": 34,
    "optimizer_updates": 0, "corpus_targets": 0, "full_transformer_runs": 0,
    "seeds": [17, 29, 43], "ffn_tensor_comparisons": 114,
    "primary_source": "https://arxiv.org/html/2410.21262v1",
}
write_new(ROOT / "protocol.json", protocol)
with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as z:
    for n in sources:
        z.write(n, n)
    z.write(PLAN, PLAN.as_posix())
    z.write(THEORY, THEORY.as_posix())
command = [sys.executable, "-X", "faulthandler", "-m", "pytest", "-p", "no:anyio",
           "-q", "-x", "--tb=short", str(ROOT / "source/test_harness.py")]
record = {"status": "RUNNING", "command": command, "coordinator_pid": os.getpid(),
          "started_utc": datetime.now(timezone.utc).isoformat(),
          "protocol_sha256": sha(ROOT / "protocol.json"),
          "source_archive_sha256": sha(ROOT / "source.zip"), "scientific_attempts": 1,
          "scientific_retries": 0, "optimizer_updates": 0}
write_new(ROOT / "process.json", record)
start = time.perf_counter()
with (ROOT / "qualification.log").open("x", encoding="utf-8") as log:
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                             env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"))
    record["pid"] = child.pid
    write_json(ROOT / "process.json", record)
    print(json.dumps(record), flush=True)
    code = child.wait()
record.update(status="PASS" if code == 0 else "FAIL", returncode=code,
              finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-start,
              source_unchanged=all(sha(n) == h for n, h in sources.items()),
              log_sha256=sha(ROOT / "qualification.log"))
write_json(ROOT / "process.json", record)
print(json.dumps(record), flush=True)
if code or not record["source_unchanged"]:
    raise SystemExit(code or 1)
observations = {n.stem: read(n) for n in (ROOT / "observations").glob("*.json")}
assert len(observations) == 34 and all(v["status"] == "PASS" for v in observations.values())
assert "34 passed" in (ROOT / "qualification.log").read_text(encoding="utf-8")
write_new(ROOT / "result.json", {
    "status": "LOCALLY_QUALIFIED", "checks_passed": 34, "optimizer_updates": 0,
    "corpus_targets": 0, "full_transformer_runs": 0, "observations": observations,
    "observation_hashes": {n.as_posix(): sha(n) for n in (ROOT / "observations").iterdir()},
    "research_goal_achieved": False, "scientific_attempts": 1, "scientific_retries": 0,
})
