"""Launch each policy in a fresh UV-managed process with explicit environment."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("results/attention_reproducibility_v1")
mode = sys.argv[1]
policies = ("original_default", "deterministic_default", "deterministic_math")
env = os.environ.copy()
env.update(
    PYTHONPATH=str(Path.cwd()) + os.pathsep + str(Path.cwd() / ".venv/Lib/site-packages"),
    PYTHONMALLOC="malloc",
    PYTHONHASHSEED="107",
)
module = "worker" if mode in policies else mode
args = [mode] if mode in policies else []
if mode in policies:
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    if mode != "original_default":
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
            "results.attention_reproducibility_v1." + module,
            *args,
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    ).returncode
(ROOT / (mode + "_exit.txt")).write_text(str(code))
raise SystemExit(code)
