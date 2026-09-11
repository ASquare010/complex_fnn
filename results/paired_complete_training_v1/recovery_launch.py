"""UV-managed isolated subprocesses; stop on any failed case, never overwrite logs."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("results/paired_complete_training_v1")
PYTHON = "C:/Users/Cuebric/AppData/Roaming/uv/python/cpython-3.12.9-windows-x86_64-none/python.exe"
stage = sys.argv[1]
if stage == "train":
    for row in json.loads((ROOT / "protocol.json").read_text())["schedule"]:
        code = subprocess.run(
            [PYTHON, "-B", str(ROOT / "recovery_launch.py"), f"case{row['index']:02d}"]
        ).returncode
        assert code == 0, (row["index"], code)
    raise SystemExit(0)
env = os.environ.copy()
env.update(
    PYTHONPATH=str(Path.cwd()) + os.pathsep + str(Path.cwd() / ".venv/Lib/site-packages"),
    PYTHONMALLOC="pymalloc",
    PYTHONHASHSEED="107",
)
module, args = stage, []
if stage.startswith("case"):
    index = int(stage[4:])
    arm = json.loads((ROOT / "protocol.json").read_text())["schedule"][index]["arm"]
    if arm == "ordinary":
        env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    module, args = "worker", [str(index)]
else:
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
with (ROOT / ("recovery_" + stage + ".log")).open("x") as log:
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
            "results.paired_complete_training_v1." + module,
            *args,
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    ).returncode
(ROOT / ("recovery_" + stage + "_exit.txt")).write_text(str(code))
raise SystemExit(code)
