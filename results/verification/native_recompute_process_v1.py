"""Durable H063 subprocess records; process files and scientific outputs are separate."""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def run(label, arguments):
    assert re.fullmatch(r"[a-z0-9_]+", label)
    folder = Path("results/native_recompute_v1/processes")
    folder.mkdir(exist_ok=True)
    record_path, log_path = folder / f"{label}.json", folder / f"{label}.log"
    assert not record_path.exists() and not log_path.exists()
    paths = [
        *Path("src").rglob("*.py"),
        *Path("tests").glob("*.py"),
        *Path("configs").glob("*.json"),
        Path("pyproject.toml"),
        Path(__file__),
    ]
    sources = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    command = [sys.executable, "-X", "faulthandler", *arguments]
    record = {"status": "RUNNING", "command": command, "sources": sources}
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"phase": label, "status": "STARTED"}), flush=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"),
        )
    record.update(
        status="PASS" if process.returncode == 0 else "FAIL",
        returncode=process.returncode,
        elapsed_seconds=time.perf_counter() - start,
        source_unchanged=all(
            hashlib.sha256(Path(n).read_bytes()).hexdigest() == h for n, h in sources.items()
        ),
        log_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest(),
    )
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in record.items() if k not in ("sources", "command")}), flush=True
    )
    print(log_path.read_text()[-7000:], flush=True)
    if process.returncode or not record["source_unchanged"]:
        raise SystemExit(process.returncode or 1)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2:])
