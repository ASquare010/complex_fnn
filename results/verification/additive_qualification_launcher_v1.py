"""Run the frozen qualification once, recording each process before heavy imports."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path("results/additive_qualification_v1")
phases = [
    ("integration_tests", ["-m", "pytest", "-p", "no:anyio", "-q"]),
    ("preflight", ["-m", "src.core.additive_qualification", "preflight"]),
]
for seed in (17, 29, 43):
    for recipe in ("full_swiglu", "full_gelu", "calibrated_narrow", "blockshuffle", "additive"):
        phases.append(
            (
                f"{recipe}_s{seed}",
                [
                    "-m",
                    "src.core.additive_qualification",
                    "worker",
                    "--recipe",
                    recipe,
                    "--seed",
                    str(seed),
                ],
            )
        )
phases.append(("finish", ["-m", "src.core.additive_qualification", "finish"]))
for label, args in phases:
    record_path = root / f"process_{label}.json"
    log_path = root / f"process_{label}.log"
    assert not record_path.exists() and not log_path.exists(), (
        f"Never repeat a recorded phase: {label}"
    )
    command = [sys.executable, "-X", "faulthandler", *args]
    sources = {
        p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            *Path("src").rglob("*.py"),
            *Path("tests").glob("*.py"),
            *Path("configs").glob("*.json"),
            Path("pyproject.toml"),
        ]
    }
    record = {"status": "RUNNING", "phase": label, "command": command, "source_hashes": sources}
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"phase": label, "status": "STARTED"}), flush=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"),
        )
    record.update(
        status="PASS" if completed.returncode == 0 else "FAIL",
        returncode=completed.returncode,
        elapsed_seconds=time.perf_counter() - start,
        source_unchanged=all(
            hashlib.sha256(Path(n).read_bytes()).hexdigest() == h for n, h in sources.items()
        ),
        log_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest(),
    )
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in record.items() if k not in ("source_hashes", "command")}),
        flush=True,
    )
    print(log_path.read_text()[-4500:], flush=True)
    if completed.returncode != 0 or not record["source_unchanged"]:
        raise SystemExit(completed.returncode or 1)
