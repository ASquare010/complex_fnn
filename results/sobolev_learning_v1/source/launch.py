"""Preserve one UV subprocess log and exit record for each named stage."""

import os
import subprocess
import sys
from pathlib import Path

folder = Path("results/sobolev_learning_v1")
mode = sys.argv[1] if len(sys.argv) > 1 else "study"
env = os.environ.copy()
env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + str(Path.cwd() / ".venv/Lib/site-packages")
env["PYTHONMALLOC"] = "malloc"
env["PYTHONHASHSEED"] = "107"
python = "C:/Users/Cuebric/AppData/Roaming/uv/python/cpython-3.12.9-windows-x86_64-none/python.exe"
with (folder / f"{mode}.log").open("x") as log:
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
            f"results.sobolev_learning_v1.source.{mode}",
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
(folder / f"{mode}_exit.txt").write_text(str(process.returncode))
raise SystemExit(process.returncode)
