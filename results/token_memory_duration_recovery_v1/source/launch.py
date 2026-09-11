"""One explicit recovery stage, preserving every original and new log."""

import os
import subprocess
import sys
from pathlib import Path

root = Path("results/token_memory_duration_recovery_v1")
mode = sys.argv[1] if len(sys.argv) > 1 else "study"
env = os.environ.copy()
env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + str(Path.cwd() / ".venv/Lib/site-packages")
env["PYTHONMALLOC"] = "malloc"
env["PYTHONHASHSEED"] = "107"
python = "C:/Users/Cuebric/AppData/Roaming/uv/python/cpython-3.12.9-windows-x86_64-none/python.exe"
with (root / f"{mode}.log").open("x") as log:
    process = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            python,
            "python",
            "-X",
            "faulthandler",
            "-u",
            "-m",
            f"results.token_memory_duration_recovery_v1.source.{mode}",
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
(root / f"{mode}_exit.txt").write_text(str(process.returncode))
raise SystemExit(process.returncode)
