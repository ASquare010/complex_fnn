"""Freeze and run one bounded workspace diagnostic; preserve the first outcome."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path("results/cublas_boundary_diagnosis_v1")
files = [
    *ROOT.glob("source/*.py"),
    Path("results/token_memory_duration_v1/source/common.py"),
    Path("results/token_memory_v1/source/execution.py"),
    Path("src/core/token_memory.py"),
]
with (ROOT / "protocol.json").open("x") as stream:
    json.dump(
        {
            "sources": {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "repetitions": 2,
            "optimizer_updates": 0,
            "question": "Does only clearing cuBLAS workspaces release all model-free post-qualification active tensor bytes?",
        },
        stream,
        indent=2,
    )
env = os.environ.copy()
env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + str(Path.cwd() / ".venv/Lib/site-packages")
env["PYTHONMALLOC"] = "malloc"
env["PYTHONHASHSEED"] = "107"
python = "C:/Users/Cuebric/AppData/Roaming/uv/python/cpython-3.12.9-windows-x86_64-none/python.exe"
with (ROOT / "study.log").open("x") as log:
    code = subprocess.run(
        [
            python,
            "-B",
            "-X",
            "pycache_prefix=" + str(ROOT.resolve() / "unused_cache"),
            "-X",
            "faulthandler",
            "-u",
            "-m",
            "results.cublas_boundary_diagnosis_v1.source.probe",
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    ).returncode
(ROOT / "study_exit.txt").write_text(str(code))
raise SystemExit(code)
