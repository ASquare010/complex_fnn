"""Bounded CPU import factorial and wheel-record verification; no training retry."""

import base64
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("results/runtime_import_diagnosis_v1")
assert not (ROOT / "protocol.json").exists()
conditions = [(version, fresh) for version in ("3.12.9", "3.12.12") for fresh in (False, True)]
sources = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "source").glob("*.py")}
protocol = {"conditions": conditions, "replications_each": 3, "allocator": "malloc", "pythonhashseed": 107, "sources": sources,
            "fresh_bytecode": "Use an absent pycache_prefix plus -B; existing caches untouched", "neural_updates": 0}
(ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
base = Path(".venv/Lib/site-packages")
checked, missing, mismatched = 0, [], []
with (base / "sympy-1.14.0.dist-info/RECORD").open(newline="", encoding="utf-8") as stream:
    for filename, digest, size in csv.reader(stream):
        if not digest:
            continue
        path = base / filename
        if not path.exists():
            missing.append(filename)
            continue
        algorithm, expected = digest.split("=",1)
        assert algorithm == "sha256"
        actual = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip("=")
        if actual != expected:
            mismatched.append(filename)
        checked += 1
integrity = {"checked_hashed_files": checked, "missing": missing, "mismatched": mismatched}
(ROOT / "package_integrity.json").write_text(json.dumps(integrity, indent=2)+"\n")
env = os.environ.copy()
env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + str(base.resolve())
env["PYTHONMALLOC"] = "malloc"
env["PYTHONHASHSEED"] = "107"
rows = []
for replication in range(3):
    order = conditions[replication:] + conditions[:replication]
    for version, fresh in order:
        label = f"py{version}_{'fresh' if fresh else 'existing'}_r{replication}"
        python = f"C:/Users/Cuebric/AppData/Roaming/uv/python/cpython-{version}-windows-x86_64-none/python.exe"
        args = [python, "-X", "faulthandler"]
        if fresh:
            prefix = (ROOT / "unused_cache" / label).resolve()
            assert not prefix.exists()
            args += ["-B", "-X", f"pycache_prefix={prefix}"]
        args += ["-u", "results/runtime_import_diagnosis_v1/source/probe.py"]
        started = time.perf_counter()
        with (ROOT / f"{label}.log").open("x") as stream:
            completed = subprocess.run(args, env=env, stdout=stream, stderr=subprocess.STDOUT)
        row = {"label": label, "python": version, "fresh_bytecode": fresh, "replication": replication, "exit_code": completed.returncode, "elapsed_seconds": time.perf_counter()-started}
        rows.append(row)
        print(json.dumps(row), flush=True)
        (ROOT / "progress.json").write_text(json.dumps(rows, indent=2)+"\n")
(ROOT / "result.json").write_text(json.dumps({"status": "COMPLETE", "integrity": integrity, "rows": rows, "neural_updates": 0, "root_cause_proved": False}, indent=2)+"\n")
