"""Run workspace policies in separate UV-managed processes."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("results/blas_workspace_mid_v1")
stage = sys.argv[1]
mode = stage.removeprefix("audit_")
is_policy = mode in ("high", "low")
module = ("audit" if stage.startswith("audit_") else "worker") if is_policy else stage
env = os.environ.copy()
env.update(
    PYTHONPATH=str(Path.cwd()) + os.pathsep + str(Path.cwd() / ".venv/Lib/site-packages"),
    PYTHONMALLOC="malloc",
    PYTHONHASHSEED="107",
)
if is_policy:
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
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
            "results.blas_workspace_mid_v1." + module,
            *([mode] if is_policy else []),
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    ).returncode
(ROOT / (stage + "_exit.txt")).write_text(str(code))
raise SystemExit(code)
