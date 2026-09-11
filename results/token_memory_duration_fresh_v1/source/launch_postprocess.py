"""One logged, cache-bypassing postprocessing invocation; no implicit retries."""

import os
import subprocess
import sys
from pathlib import Path

root = Path("results/token_memory_duration_fresh_v1")
mode = sys.argv[1]
assert mode in ("audit", "analyze", "plot")
env = os.environ.copy()
env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + str(Path.cwd() / ".venv/Lib/site-packages")
env["PYTHONMALLOC"] = "malloc"
env["PYTHONHASHSEED"] = "107"
python = "C:/Users/Cuebric/AppData/Roaming/uv/python/cpython-3.12.9-windows-x86_64-none/python.exe"
with (root / f"{mode}.log").open("x") as stream:
    code = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            python,
            "python",
            "-B",
            "-X",
            "pycache_prefix=" + str(root.resolve() / "unused_postprocess_cache"),
            "-X",
            "faulthandler",
            "-u",
            "-m",
            f"results.token_memory_duration_fresh_v1.source.{mode}",
        ],
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
    ).returncode
(root / f"{mode}_exit.txt").write_text(str(code))
raise SystemExit(code)
