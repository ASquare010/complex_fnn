"""One explicitly recorded CPU-bytecode-bypassing, zero-update audit attempt."""

import os
import subprocess
from pathlib import Path

root = Path("results/token_memory_duration_v1")
env = os.environ.copy()
env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + str(Path.cwd() / ".venv/Lib/site-packages")
env["PYTHONMALLOC"] = "malloc"
env["PYTHONHASHSEED"] = "107"
python = "C:/Users/Cuebric/AppData/Roaming/uv/python/cpython-3.12.9-windows-x86_64-none/python.exe"
with (root / "audit.log").open("x") as stream:
    code = subprocess.run(
        [
            "uv", "run", "--no-project", "--python", python, "python", "-B", "-X",
            "pycache_prefix=" + str(root.resolve() / "unused_audit_cache"),
            "-X", "faulthandler", "-u", "-m", "results.token_memory_duration_v1.source.audit",
        ],
        env=env, stdout=stream, stderr=subprocess.STDOUT,
    ).returncode
(root / "audit_exit.txt").write_text(str(code))
raise SystemExit(code)
