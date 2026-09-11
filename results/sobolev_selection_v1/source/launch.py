"""Single-process UV launcher with preserved, exclusive logs and exit receipts."""

import os
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
folder = Path("results/sobolev_selection_v1")
mode = sys.argv[1] if len(sys.argv) > 1 else "study"
env = os.environ.copy()
env["PYTHONPATH"] = str(root) + os.pathsep + str(root / ".venv/Lib/site-packages")
env["PYTHONMALLOC"] = "malloc"
env["PYTHONHASHSEED"] = "107"
python = "C:/Users/Cuebric/AppData/Roaming/uv/python/cpython-3.12.9-windows-x86_64-none/python.exe"
with (folder / f"{mode}.log").open("x") as log:
    result = subprocess.run(
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
            f"results.sobolev_selection_v1.source.{mode}",
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
(folder / f"{mode}_exit.txt").write_text(str(result.returncode))
raise SystemExit(result.returncode)
