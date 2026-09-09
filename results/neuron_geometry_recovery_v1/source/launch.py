"""One H090 recovery, CPU-payload preservation, then fresh fixed-budget fitting."""

import hashlib
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from results.blast_operator_recovery_v1.source.storage import equal_payload, read, sha, write_json
from src.core.reproducibility import environment, provenance

ROOT = Path("results/neuron_geometry_recovery_v1")
OLD = Path("results/neuron_geometry_v1")
before = read(ROOT / "before.json")
assert not (ROOT / "protocol.json").exists()
assert all(sha(name) == value for name, value in before["source_hashes"].items())
assert all(sha(name) == value for name, value in before["anchors"].items())
assert not (OLD / "data.pt").exists() and not list((OLD / "cells").iterdir())
protocol = {
    **before,
    "environment": environment(),
    "provenance": provenance(),
    "before_sha256": sha(ROOT / "before.json"),
    "qualification_checks": 27,
    "explicit_qualification_repetitions": 1,
    "training_runs": 336,
    "training_updates": 201600,
    "training_example_presentations": 51609600,
}
write_json(ROOT / "protocol.json", protocol)
with (ROOT / "source.zip").open("xb") as stream:
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in dict.fromkeys([*before["source_hashes"], *before["anchors"]]):
            archive.write(name, name)
    stream.flush()
    os.fsync(stream.fileno())
with zipfile.ZipFile(ROOT / "source.zip") as archive:
    assert archive.testzip() is None
    assert all(
        hashlib.sha256(archive.read(name)).hexdigest() == value
        for name, value in {**before["source_hashes"], **before["anchors"]}.items()
    )


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


def verify_original_cpu_results():
    observations, tensors = 0, 0
    for path in (OLD / "qualification").glob("*.json"):
        old, new = read(path), read(ROOT / "qualification" / path.name)
        if "tensor_file" in old:
            equal_payload(
                torch.load(
                    OLD / "qualification" / old["tensor_file"],
                    map_location="cpu",
                    weights_only=True,
                ),
                torch.load(
                    ROOT / "qualification" / new["tensor_file"],
                    map_location="cpu",
                    weights_only=True,
                ),
            )
            tensors += 1
            old, new = (
                {key: value for key, value in row.items() if key != "tensor_sha256"}
                for row in (old, new)
            )
        assert old == new
        observations += 1
    assert observations == 21 and tensors == 6
    for name, metadata in before["h088_files"].items():
        path = Path(name)
        assert sha(path) == metadata["sha256"]
        assert (path.stat().st_size, path.stat().st_mtime_ns) == (
            metadata["bytes"],
            metadata["mtime_ns"],
        )
    write_json(
        ROOT / "qualification_preservation.json",
        {
            "status": "PASS",
            "observations": observations,
            "tensor_payloads_exact": tensors,
            "original_files": len(before["h088_files"]),
            "optimizer_updates": 0,
        },
    )


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
        str(ROOT / "source/test_recovery.py"),
    ]
    assert "27 tests collected" in execute("collection", [*tests, "--collect-only"])
    assert not list((ROOT / "qualification").iterdir())
    assert "27 passed" in execute("qualification", tests)
    verify_original_cpu_results()
    execute("fitting", ["-m", "results.neuron_geometry_recovery_v1.source.recovery"])
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
