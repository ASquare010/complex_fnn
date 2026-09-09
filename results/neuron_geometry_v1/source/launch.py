"""H088 durable one-shot qualification and fitting coordinator."""

import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from results.blast_operator_recovery_v1.source.storage import read, sha, write_json
from src.core.reproducibility import environment, provenance

ROOT = Path("results/neuron_geometry_v1")
before = read(ROOT / "before.json")
assert not (ROOT / "protocol.json").exists()
assert all(sha(name) == value for name, value in before["source_hashes"].items())
assert all(sha(name) == value for name, value in before["anchors"].items())
protocol = {
    **before,
    "environment": environment(),
    "provenance": provenance(),
    "qualification_checks": 27,
    "training_runs": 336,
    "steps_per_run": 600,
    "batch_size": 256,
    "training_example_presentations": 51609600,
    "before_sha256": sha(ROOT / "before.json"),
    "scientific_retries": 0,
}
write_json(ROOT / "protocol.json", protocol)
with (ROOT / "source.zip").open("xb") as stream:
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in dict.fromkeys([*before["source_hashes"], *before["anchors"]]):
            archive.write(path, path)
    stream.flush()
    os.fsync(stream.fileno())
with zipfile.ZipFile(ROOT / "source.zip") as archive:
    assert archive.testzip() is None
    for path, expected in {**before["source_hashes"], **before["anchors"]}.items():
        import hashlib

        assert hashlib.sha256(archive.read(path)).hexdigest() == expected


def execute(phase, arguments):
    command = [sys.executable, "-X", "faulthandler", *arguments]
    record = {
        "status": "RUNNING",
        "command": command,
        "coordinator_pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha(ROOT / "protocol.json"),
    }
    path, logfile = ROOT / f"{phase}_process.json", ROOT / f"{phase}.log"
    write_json(path, record)
    started = time.perf_counter()
    with logfile.open("xb") as stream:
        child = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=dict(
                os.environ,
                OMP_NUM_THREADS="4",
                MKL_NUM_THREADS="4",
                MPLCONFIGDIR=str(Path(".cache/matplotlib").resolve()),
            ),
        )
        record["pid"] = child.pid
        write_json(path, record, exclusive=False)
        print(record, flush=True)
        code = child.wait()
        stream.flush()
        os.fsync(stream.fileno())
    record.update(
        status="PASS" if code == 0 else "FAIL",
        returncode=code,
        elapsed_seconds=time.perf_counter() - started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        log_sha256=sha(logfile),
    )
    write_json(path, record, exclusive=False)
    print(record, flush=True)
    assert code == 0, logfile.read_text(encoding="utf-8")[-12000:]
    return logfile.read_text(encoding="utf-8")


status = {
    "status": "RUNNING",
    "coordinator_pid": os.getpid(),
    "started_utc": datetime.now(timezone.utc).isoformat(),
}
write_json(ROOT / "coordinator_status.json", status)
try:
    tests = [
        "-m",
        "pytest",
        "-p",
        "no:anyio",
        "-q",
        "-x",
        "--tb=short",
        str(ROOT / "source/test_qualification.py"),
    ]
    assert "27 tests collected" in execute("collection", [*tests, "--collect-only"])
    assert not list((ROOT / "qualification").iterdir())
    assert "27 passed" in execute("qualification", tests)
    observations = [read(path) for path in (ROOT / "qualification").glob("*.json")]
    assert len(observations) == 27 and all(row["status"] == "PASS" for row in observations)
    execute("fitting", ["-m", "results.neuron_geometry_v1.source.study"])
    result = read(ROOT / "result.json")
    assert result["runs"] == 336 and result["updates"] == 201600
    assert all(sha(name) == value for name, value in before["source_hashes"].items())
except BaseException as exc:
    status.update(status="FAIL", exception_type=type(exc).__name__, exception=str(exc))
    raise
else:
    status["status"] = "PASS"
finally:
    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(ROOT / "coordinator_status.json", status, exclusive=False)
