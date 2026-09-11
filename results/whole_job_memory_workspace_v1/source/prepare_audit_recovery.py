"""Preserve the first audit failure and freeze the bounded post-training recovery."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/whole_job_memory_workspace_v1")
assert (ROOT / "study_exit.txt").read_text().strip() == "0"
assert (ROOT / "audit_exit.txt").read_text().strip() == "3221225477"
assert not (ROOT / "audit.json").exists()
paths = [
    ROOT / name
    for name in ("audit.log", "audit_exit.txt", "result.json", "protocol.json", "study_exit.txt")
]
paths += [*ROOT.glob("source/*.py"), Path("research/whole_job_memory_audit_recovery.md")]
record = dict(
    reason="native access violation during module loading before audit output",
    training_repeated=False,
    root_cause_proved=False,
    unchanged_scoring_source=(ROOT / "source/audit.py").as_posix(),
    files={p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
)
with (ROOT / "audit_recovery_before.json").open("x") as handle:
    json.dump(record, handle, indent=2)
print("Preserved first audit failure and all current sources before recovery")
