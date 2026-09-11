"""Freeze the narrower completion audit; keep both gradient-audit failures."""

import hashlib
import json
from pathlib import Path

root = Path("results/fp32_classifier_profile_v1")
assert not (root / "completion_audit_protocol.json").exists()
paths = [
    root / name
    for name in (
        "source/completion_audit.py",
        "source/prepare_completion_audit.py",
        "audit_recovery_protocol.json",
        "audit_recovery_failure.json",
        "audit_recovery.log",
        "audit_recovery_exit.txt",
    )
]
paths.append(Path("research/fp32_classifier_profile_completion_audit.md"))
assert (root / "audit_recovery_exit.txt").read_text() == "1"
files = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(root / "completion_audit_protocol.json").write_text(
    json.dumps(
        dict(
            files=files,
            optimizer_updates=0,
            backward_passes=0,
            full_gradient_replay_qualified=False,
            no_further_tolerance_relaxation=True,
        ),
        indent=2,
    )
    + "\n"
)
print("Completion audit frozen with gradient advancement hold")
