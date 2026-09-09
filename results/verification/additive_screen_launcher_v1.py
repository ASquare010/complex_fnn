"""Run H061 once with durable phase records and no automatic cell retries."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path("results/verification")
phases = [
    ("tests", ["-m", "pytest", "-p", "no:anyio", "-q"]),
    ("preflight", ["-m", "src.core.additive_screen", "preflight"]),
]
for rate in (300, 600, 1200):
    phases.append(
        (
            f"lr{rate}",
            [
                "-m",
                "src.core.frozen_train_worker",
                "--protocol",
                "results/additive_screen_v1/protocol.json",
                "--qualification",
                "results/additive_screen_v1/worker_qualification.json",
                "--cell",
                f"wikitext2_additive_block_lowrank_lr{rate}_s17_200",
            ],
        )
    )
phases.append(("finish", ["-m", "src.core.additive_screen", "finish"]))
for label, args in phases:
    record_path = root / f"additive_screen_{label}_v1.json"
    log_path = root / f"additive_screen_{label}_v1.log"
    assert not record_path.exists() and not log_path.exists(), (
        f"Never repeat a recorded phase: {label}"
    )
    paths = [
        *Path("src").rglob("*.py"),
        *Path("tests").glob("*.py"),
        *Path("configs").glob("*.json"),
        Path("pyproject.toml"),
        Path("results/verification/additive_control_probe_v1.py"),
    ]
    sources = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    command = [sys.executable, "-X", "faulthandler", *args]
    record = {"status": "RUNNING", "phase": label, "command": command, "source_hashes": sources}
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"phase": label, "status": "STARTED"}), flush=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"),
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
        json.dumps({k: v for k, v in record.items() if k not in ("source_hashes", "command")}),
        flush=True,
    )
    print(log_path.read_text()[-6500:], flush=True)
    if process.returncode != 0 or not record["source_unchanged"]:
        raise SystemExit(process.returncode or 1)
