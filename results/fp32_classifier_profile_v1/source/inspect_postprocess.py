"""Read-only inspection of this study's unexpectedly slow report process."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / ".venv/Lib/site-packages"))
import psutil  # noqa: E402

for process in psutil.process_iter(["pid", "name", "cmdline", "cpu_times", "status"]):
    info = process.info
    if (
        info["name"]
        and "python" in info["name"].lower()
        and "fp32_classifier_profile_v1" in " ".join(info["cmdline"] or [])
    ):
        print(json.dumps(info, default=str))
