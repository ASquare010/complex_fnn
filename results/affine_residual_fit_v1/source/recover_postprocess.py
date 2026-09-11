"""Explicit postprocessing recovery; never rerun fitting or modify thresholds."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
folder = Path("results/affine_residual_fit_v1")
mode = sys.argv[1]
assert mode in ("audit", "plot", "diagnostics", "tests")
receipt = folder / f"{mode}_recovery_before.json"
assert not receipt.exists(), "Each recovery attempt needs a separate reviewed record"
files = [folder / "result.json", folder / "protocol.json", folder / "source" / f"{mode}.py"]
files += [folder / f"{mode}{suffix}" for suffix in (".log", "_exit.txt") if (folder / f"{mode}{suffix}").exists()]
receipt.write_text(json.dumps({"purpose": "Postprocessing only; no retraining, checkpoint rewrite or gate change", "runtime_change": "Fresh process, PYTHONMALLOC=malloc, PYTHONHASHSEED=107; root cause of native access violations remains unproven", "before": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}, indent=2) + "\n")
env = os.environ.copy()
env["PYTHONPATH"] = str(root) + os.pathsep + str(root / ".venv/Lib/site-packages")
env["PYTHONMALLOC"] = "malloc"
env["PYTHONHASHSEED"] = "107"
python = "C:/Users/Cuebric/AppData/Roaming/uv/python/cpython-3.12.9-windows-x86_64-none/python.exe"
with (folder / f"{mode}_recovery.log").open("x") as log:
    result = subprocess.run(["uv", "run", "--no-project", "--python", python, "python", "-X", "faulthandler", "-u", "-m", f"results.affine_residual_fit_v1.source.{mode}"], env=env, stdout=log, stderr=subprocess.STDOUT)
(folder / f"{mode}_recovery_exit.txt").write_text(str(result.returncode))
raise SystemExit(result.returncode)
