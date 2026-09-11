"""Launch each execution policy once through UV's managed interpreter."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("results/native_buffer_training_v1")
mode = sys.argv[1]
module = "worker" if mode in ("ordinary", "deterministic") else mode
env = os.environ.copy()
env.update(
    PYTHONPATH=str(Path.cwd()) + os.pathsep + str(Path.cwd() / ".venv/Lib/site-packages"),
    PYTHONMALLOC="malloc",
    PYTHONHASHSEED="107",
)
env.pop("CUBLAS_WORKSPACE_CONFIG", None)
if mode in ("deterministic", "audit"):
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
with (ROOT / (mode + ".log")).open("x") as log:
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
            "results.native_buffer_training_v1." + module,
            *([mode] if module == "worker" else []),
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    ).returncode
(ROOT / (mode + "_exit.txt")).write_text(str(code))
raise SystemExit(code)
