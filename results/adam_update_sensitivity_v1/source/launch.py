"""One exclusive UV-managed subprocess; retain logs and raw exit status."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("results/adam_update_sensitivity_v1")
mode = sys.argv[1]
env = os.environ.copy()
env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + str(Path.cwd() / ".venv/Lib/site-packages")
env["PYTHONMALLOC"] = "malloc"
env["PYTHONHASHSEED"] = "107"
python = "C:/Users/Cuebric/AppData/Roaming/uv/python/cpython-3.12.9-windows-x86_64-none/python.exe"
with (ROOT / f"{mode}.log").open("x") as log:
    code = subprocess.run(
        [
            python,
            "-B",
            "-X",
            "pycache_prefix=" + str(ROOT.resolve() / "unused_cache"),
            "-X",
            "faulthandler",
            "-u",
            "-m",
            f"results.adam_update_sensitivity_v1.source.{mode}",
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    ).returncode
(ROOT / f"{mode}_exit.txt").write_text(str(code))
raise SystemExit(code)
