"""Freeze sources, collect28 checks, run once, persist all observations."""

import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from results.blast_operator_recovery_v1.source.storage import read, sha, write_json
from src.core.reproducibility import environment, provenance

ROOT = Path("results/compact_sparse_operator_v1")
before = read(ROOT / "before.json")
assert not (ROOT / "protocol.json").exists()
assert not list((ROOT / "observations").iterdir())
assert all(sha(path) == value for path, value in before["prior_sources"].items())
assert all(sha(path) == value for path, value in before["prior_plans"].items())
assert all(sha(path) == value for path, value in before["anchors"].items())
assert all(sha(path) == value for path, value in before["new_sources"].items())
sources = {**before["prior_sources"], **before["new_sources"]}
plans = {**before["prior_plans"], **before["new_plans"]}
assert len(sources) == 139 and len(plans) == 75
assert all(sha(path) == value for path, value in plans.items())
protocol = {"sources": sources, "plans": plans, "anchors": before["anchors"],
            "environment": environment(), "provenance": provenance(), "checks": 28,
            "optimizer_updates": 0, "corpus_targets": 0, "training_examples": 0,
            "source_archive": "source.zip", "before_sha256": sha(ROOT / "before.json")}
write_json(ROOT / "protocol.json", protocol)
with (ROOT / "source.zip").open("xb") as stream:
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in dict.fromkeys([*sources, *plans]):
            archive.write(path, path)
    stream.flush()
    os.fsync(stream.fileno())
with zipfile.ZipFile(ROOT / "source.zip") as archive:
    assert archive.testzip() is None


def execute(phase, flags):
    command = [sys.executable, "-X", "faulthandler", "-m", "pytest", "-p", "no:anyio",
               "-q", "-x", "--tb=short", *flags, str(ROOT / "source/test_qualification.py")]
    record = {"status": "RUNNING", "command": command, "coordinator_pid": os.getpid(),
              "started_utc": datetime.now(timezone.utc).isoformat(), "optimizer_updates": 0}
    path, logfile = ROOT / f"{phase}_process.json", ROOT / f"{phase}.log"
    write_json(path, record)
    started = time.perf_counter()
    with logfile.open("xb") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"))
        record["pid"] = child.pid
        write_json(path, record, exclusive=False)
        print(record, flush=True)
        code = child.wait()
        stream.flush()
        os.fsync(stream.fileno())
    record.update(status="PASS" if code == 0 else "FAIL", returncode=code,
                  elapsed_seconds=time.perf_counter() - started,
                  finished_utc=datetime.now(timezone.utc).isoformat(), log_sha256=sha(logfile))
    write_json(path, record, exclusive=False)
    print(record, flush=True)
    assert code == 0, logfile.read_text(encoding="utf-8")
    return logfile.read_text(encoding="utf-8")


status = {"status": "RUNNING", "coordinator_pid": os.getpid(),
          "started_utc": datetime.now(timezone.utc).isoformat()}
write_json(ROOT / "coordinator_status.json", status)
try:
    assert "28 tests collected" in execute("collection", ["--collect-only"])
    assert not list((ROOT / "observations").iterdir())
    assert "28 passed" in execute("qualification", [])
    observations = {path.stem: read(path) for path in (ROOT / "observations").glob("*.json")}
    assert len(observations) == 28
    assert all(value["status"] == "PASS" or
               (value["kind"] == "hardware" and value["status"] == "UNAVAILABLE")
               for value in observations.values())
    assert all(sha(path) == value for path, value in sources.items())
    assert all(sha(path) == value for path, value in plans.items())
    assert all(sha(path) == value for path, value in before["anchors"].items())
    write_json(ROOT / "result.json", {"status": "NATIVE_LOCALLY_QUALIFIED", "checks": 28,
               "optimizer_updates": 0, "training_examples": 0, "corpus_targets": 0,
               "observations": observations, "research_goal_achieved": False,
               "protocol_sha256": sha(ROOT / "protocol.json"), "source_sha256": sha(ROOT / "source.zip"),
               "observation_hashes": {path.as_posix(): sha(path) for path in (ROOT / "observations").iterdir()}})
except BaseException as exc:
    status.update(status="FAIL", exception_type=type(exc).__name__, exception=str(exc))
    raise
else:
    status["status"] = "PASS"
finally:
    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(ROOT / "coordinator_status.json", status, exclusive=False)
