"""Use the existing UV-managed Python runtime with exclusive stage logs."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("results/paired_workspace_timing_v1")
stage = sys.argv[1]
env = os.environ.copy()
env.update(
    PYTHONPATH=str(Path.cwd()) + os.pathsep + str(Path.cwd() / ".venv/Lib/site-packages"),
    PYTHONMALLOC="malloc",
    PYTHONHASHSEED="107",
    CUBLAS_WORKSPACE_CONFIG=":4096:8",
)
with (ROOT / (stage + ".log")).open("x") as log:
    code = subprocess.run(
        [
            "C:/Users/Cuebric/AppData/Roaming/uv/python/cpython-3.12.9-windows-x86_64-none/python.exe",
            "-B",
            "-X",
            "pycache_prefix=" + str(ROOT.resolve() / "unused_cache"),
            "-X",
            "faulthandler",
            "-u",
            "-m",
            "results.paired_workspace_timing_v1." + stage,
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    ).returncode
(ROOT / (stage + "_exit.txt")).write_text(str(code))
raise SystemExit(code)
