"""Durable stdlib-only phase launcher; no automatic retries or training restarts."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path("results/verification")
for label, arguments, timeout in (
    ("focused_tests", ["-m", "pytest", "-p", "no:anyio", "-q", "tests/test_duration_decay.py", "tests/test_frozen_train_worker.py"], 120),
    ("prepare", ["-m", "src.core.duration_decay", "prepare"], 240),
    ("train", ["-m", "src.core.frozen_train_worker", "--protocol", "results/duration_decay_v1/protocol.json",
               "--qualification", "results/duration_decay_v1/qualification.json", "--cell", "product_s17"], 2400),
    ("verify", ["-m", "src.core.duration_decay", "verify"], 240),
):
    record_path = root / f"duration_decay_{label}_v1.json"
    assert not record_path.exists()
    command = [sys.executable, "-X", "faulthandler", *arguments]
    source = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for base in ("src", "tests") for p in Path(base).rglob("*.py")}
    record = {"status": "running", "command": command, "source_files": source, "started_unix": time.time()}
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    begin = time.perf_counter()
    try:
        with (root / f"duration_decay_{label}_v1.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        record.update(status="complete", returncode=process.returncode)
    except BaseException as exc:
        record.update(status="failed", exception=repr(exc), returncode=None)
    record.update(seconds=time.perf_counter() - begin,
                  source_unchanged=all(hashlib.sha256(Path(n).read_bytes()).hexdigest() == h for n,h in source.items()))
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in record.items() if k != "source_files"}), flush=True)
    if record["returncode"] != 0:
        sys.exit(1)
