"""Run a stage once through the existing UV-managed runtime."""

import os
import subprocess
import sys
from pathlib import Path

root = Path("results/compact_rmsnorm_v1")
mode = sys.argv[1]
env = os.environ.copy()
env.update(
    PYTHONPATH=str(Path.cwd()) + os.pathsep + str(Path.cwd() / ".venv/Lib/site-packages"),
    PYTHONMALLOC="malloc",
    PYTHONHASHSEED="107",
)
with (root / (mode + ".log")).open("x") as log:
    code = subprocess.run(
        [
            "C:/Users/Cuebric/AppData/Roaming/uv/python/cpython-3.12.9-windows-x86_64-none/python.exe",
            "-B",
            "-X",
            "pycache_prefix=" + str(root.resolve() / "unused_cache"),
            "-X",
            "faulthandler",
            "-u",
            "-m",
            "results.compact_rmsnorm_v1." + mode,
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    ).returncode
(root / (mode + "_exit.txt")).write_text(str(code))
raise SystemExit(code)
