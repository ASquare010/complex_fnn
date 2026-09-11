"""Fresh CPU-only execution of the unchanged analyzer after its access violation."""

import faulthandler
import hashlib
import json
from pathlib import Path

ROOT = Path("results/fp32_decoder_resource_v1")
record = json.loads((ROOT / "analysis_recovery_protocol.json").read_text())
for name, digest in record["files"].items():
    with Path(name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, name
faulthandler.dump_traceback_later(45, repeat=True)
try:
    from results.fp32_decoder_resource_v1.source.analyze import run

    run()
finally:
    faulthandler.cancel_dump_traceback_later()
