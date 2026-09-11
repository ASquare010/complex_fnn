"""UV-managed isolated subprocesses; stop on any failed case, never overwrite logs."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("results/fused_reconstruction_v1")
PYTHON = "C:/Users/Cuebric/AppData/Roaming/uv/python/cpython-3.12.9-windows-x86_64-none/python.exe"
stage = sys.argv[1]
env = os.environ.copy()
env.update(
    PYTHONPATH=str(Path.cwd()) + os.pathsep + str(Path.cwd() / ".venv/Lib/site-packages"),
    PYTHONMALLOC="pymalloc",
    PYTHONHASHSEED="107",
)
env.pop("CUBLAS_WORKSPACE_CONFIG", None)
env["CC"] = str(Path(".venv/Lib/site-packages/triton/runtime/tcc/tcc.exe").resolve())
env["TRITON_CACHE_DIR"] = str((ROOT / "triton_cache").resolve())
module, args = stage, []
if stage.startswith("case"):
    module, args = "worker", [str(int(stage[4:]))]
with (ROOT / (stage + ".log")).open("x") as log:
    code = subprocess.run(
        [
            PYTHON,
            "-B",
            "-X",
            "pycache_prefix=" + str(ROOT.resolve() / "unused_cache"),
            "-X",
            "faulthandler",
            "-u",
            "-m",
            "results.fused_reconstruction_v1." + module,
            *args,
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    ).returncode
(ROOT / (stage + "_exit.txt")).write_text(str(code))
raise SystemExit(code)
