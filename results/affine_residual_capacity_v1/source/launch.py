"""Run this read-only capacity screen with the already qualified UV runtime."""

import os
import subprocess
from pathlib import Path

root = Path.cwd()
env = os.environ.copy()
env["PYTHONPATH"] = str(root) + os.pathsep + str(root / ".venv/Lib/site-packages")
python = "C:/Users/Cuebric/AppData/Roaming/uv/python/cpython-3.12.9-windows-x86_64-none/python.exe"
folder = Path("results/affine_residual_capacity_v1")
with (folder / "study.log").open("w") as log:
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
            "results.affine_residual_capacity_v1.source.study",
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
(folder / "study_exit.txt").write_text(str(result.returncode))
raise SystemExit(result.returncode)
