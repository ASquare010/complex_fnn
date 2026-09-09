"""Durable standard-library H057 launcher; no automatic retries."""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path("results/verification")
phases = [
    (
        "focused_tests",
        ["-m", "pytest", "-p", "no:anyio", "-q", "tests/test_residual_geometry.py"],
        180,
    ),
    ("preflight", ["-m", "src.core.residual_geometry", "preflight"], 180),
    ("cpu", ["-m", "src.core.residual_geometry", "cpu"], 600),
    ("gpu", ["-m", "src.core.residual_geometry", "gpu"], 300),
    ("finish", ["-m", "src.core.residual_geometry", "finish"], 60),
    ("tests", ["-m", "pytest", "-p", "no:anyio", "-q"], 240),
]
for label, args, timeout in phases:
    path = root / f"residual_geometry_{label}_v1.json"
    assert not path.exists()
    sources = {
        f.as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
        for base in ("src", "tests")
        for f in Path(base).rglob("*.py")
    }
    command = [sys.executable, "-X", "faulthandler", *args]
    record = {
        "status": "running",
        "command": command,
        "source_files": sources,
        "started_unix": time.time(),
        "plan_sha256": hashlib.sha256(
            Path("research/residual_geometry_plan.md").read_bytes()
        ).hexdigest(),
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    start = time.perf_counter()
    try:
        with (root / f"residual_geometry_{label}_v1.log").open("x", encoding="utf-8") as log:
            code = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout
            ).returncode
        record.update(status="complete", returncode=code)
    except BaseException as exc:
        record.update(status="failed", returncode=None, exception=repr(exc))
    record.update(
        seconds=time.perf_counter() - start,
        source_unchanged=all(
            hashlib.sha256(Path(n).read_bytes()).hexdigest() == h for n, h in sources.items()
        ),
    )
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "source_files"}), flush=True)
    if record["returncode"] != 0 or not record["source_unchanged"]:
        sys.exit(1)
