"""Fix publication newline conversion and expose only compact H117 artifacts."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path("results/fp32_training_replication_recovery_v1")
protocol = json.loads((ROOT / "protocol.json").read_text())
receipt = json.loads(
    Path("results/verification/fp32_training_replication_final_v1.json").read_text()
)
pending = {}
for name in protocol["before_documents"]:
    raw = Path(name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt["files"][name], name
    pending[name] = re.sub(rb"\r+\n", b"\n", raw)
extra = """!/results/fp32_training_replication_v1/*.log.gz
!/results/fp32_training_replication_v1/*_exit.txt
!/results/fp32_training_replication_v1/failure.json
!/results/fp32_training_replication_v1/qualification.json
!/results/fp32_training_replication_v1/environment.json
!/results/fp32_training_replication_recovery_v1/*.json.gz
!/results/fp32_training_replication_recovery_v1/*.csv.gz
!/results/fp32_training_replication_recovery_v1/*.log.gz
!/results/fp32_training_replication_recovery_v1/*_exit.txt
!/results/fp32_training_replication_recovery_v1/*_protocol.json
!/results/fp32_training_replication_recovery_v1/qualification.json
!/results/fp32_training_replication_recovery_v1/environment.json
!/results/fp32_training_replication_recovery_v1/initializations.json
"""
assert extra.splitlines()[0].encode() not in pending[".gitignore"]
pending[".gitignore"] += extra.encode()
for name, data in pending.items():
    Path(name).write_bytes(data)
print("Normalized seven navigation files to LF and retained compact-artifact allow rules.")
