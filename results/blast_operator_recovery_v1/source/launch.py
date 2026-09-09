"""One explicit H080 qualification recovery; immutable H079 computations."""

import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from results.blast_operator_recovery_v1.source.storage import read, sha, write_json
from src.core.reproducibility import environment, provenance

ROOT = Path("results/blast_operator_recovery_v1")
OLD = Path("results/blast_operator_v1")
PLAN = Path("research/blast_operator_recovery_plan.md")
THEORY = Path("research/blast_operator_theory.md")
before = read(ROOT / "before.json")
assert not (ROOT / "protocol.json").exists() and not list((ROOT / "observations").iterdir())
assert sha(PLAN) == before["plan_sha256"]
assert sha(THEORY) == before["h079_theory_sha256"]
assert all(sha(n) == h for n, h in before["prior_sources"].items())
assert all(sha(n) == h for n, h in before["prior_plans"].items())
assert sha(OLD / "integrity_observation.json") == before["h079_integrity_observation_sha256"]
assert all(sha(n) == v["sha256"] for n, v in read(OLD / "integrity_observation.json")["files"].items())
for filename, key in (("protocol.json", "h079_protocol_sha256"),
                      ("source.zip", "h079_source_sha256"), ("process.json", "h079_process_sha256")):
    assert sha(OLD / filename) == before[key]
sources = dict(before["prior_sources"])
sources.update({n.as_posix(): sha(n) for n in (ROOT / "source").glob("*.py")})
assert len(sources) == 118
protocol = {"plan_sha256": sha(PLAN), "theory_sha256": sha(THEORY), "sources": sources,
            "environment": environment(), "provenance": provenance(), "checks": 34,
            "optimizer_updates": 0, "corpus_targets": 0, "full_transformer_runs": 0,
            "explicit_qualification_repetitions": 1, "total_h079_h080_attempts": 2,
            "original_integrity_sha256": before["h079_integrity_observation_sha256"],
            "unchanged_original_test_source_sha256": sources[(OLD / "source/test_harness.py").as_posix()]}
write_json(ROOT / "protocol.json", protocol)
with (ROOT / "source.zip").open("xb") as f:
    with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as z:
        for n in sources:
            z.write(n, n)
        for n in (PLAN, THEORY, Path("research/blast_operator_plan.md")):
            z.write(n, n.as_posix())
    f.flush()
    os.fsync(f.fileno())
with zipfile.ZipFile(ROOT / "source.zip") as z:
    assert z.testzip() is None


def execute(phase, flags):
    command = [sys.executable, "-X", "faulthandler", "-m", "pytest", "-p", "no:anyio",
               "-q", "-x", "--tb=short", *flags, str(ROOT / "source/test_recovery.py")]
    record = {"status": "RUNNING", "command": command, "coordinator_pid": os.getpid(),
              "started_utc": datetime.now(timezone.utc).isoformat(), "optimizer_updates": 0,
              "protocol_sha256": sha(ROOT / "protocol.json"), "source_archive_sha256": sha(ROOT / "source.zip")}
    path, logfile = ROOT / (phase + "_process.json"), ROOT / (phase + ".log")
    write_json(path, record)
    start = time.perf_counter()
    with logfile.open("xb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"))
        record["pid"] = child.pid
        write_json(path, record, exclusive=False)
        print(json.dumps(record), flush=True)
        code = child.wait()
        log.flush()
        os.fsync(log.fileno())
    contents = logfile.read_text(encoding="utf-8")
    record.update(status="PASS" if code == 0 else "FAIL", returncode=code,
                  elapsed_seconds=time.perf_counter()-start,
                  finished_utc=datetime.now(timezone.utc).isoformat(), log_sha256=sha(logfile),
                  source_unchanged=all(sha(n) == h for n, h in sources.items()))
    write_json(path, record, exclusive=False)
    print(json.dumps(record), flush=True)
    assert code == 0 and record["source_unchanged"], contents
    return contents


status = {"status": "RUNNING", "coordinator_pid": os.getpid(),
          "started_utc": datetime.now(timezone.utc).isoformat()}
write_json(ROOT / "coordinator_status.json", status)
try:
    collected = execute("collection", ["--collect-only"])
    assert "34 tests collected" in collected and not list((ROOT / "observations").iterdir())
    output = execute("qualification", [])
    assert "34 passed" in output
    observations = {n.stem: read(n) for n in (ROOT / "observations").glob("*.json")}
    assert len(observations) == 34 and all(v["status"] == "PASS" for v in observations.values())
    write_json(ROOT / "result.json", {
        "status": "LOCALLY_QUALIFIED", "checks_passed": 34, "optimizer_updates": 0,
        "corpus_targets": 0, "full_transformer_runs": 0, "observations": observations,
        "observation_hashes": {n.as_posix(): sha(n) for n in (ROOT / "observations").iterdir()},
        "research_goal_achieved": False, "explicit_qualification_repetitions": 1,
        "total_h079_h080_attempts": 2, "original_h079_evidence_status": "INCOMPLETE_ARTIFACT_INTEGRITY",
    })
except BaseException as exc:
    status.update(status="FAIL", exception_type=type(exc).__name__, exception=str(exc))
    raise
else:
    status["status"] = "PASS"
finally:
    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(ROOT / "coordinator_status.json", status, exclusive=False)
