"""Fresh CPU-only execution of the unchanged plot after an import access violation."""

import faulthandler
import hashlib
import importlib
import json
from pathlib import Path

ROOT = Path("results/fp32_training_replication_recovery_v1")
protocol = json.loads((ROOT / "plot_recovery_protocol.json").read_text())
for name, digest in protocol["files"].items():
    with Path(name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, name
faulthandler.dump_traceback_later(45, repeat=True)
try:
    importlib.import_module("results.fp32_training_replication_recovery_v1.source.plot")
finally:
    faulthandler.cancel_dump_traceback_later()
