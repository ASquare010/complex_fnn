"""Preserve the failed audit and freeze the explicit zero-update recovery."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/fp32_classifier_profile_v1")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert not (ROOT / "audit_recovery_protocol.json").exists()
original = json.loads((ROOT / "audit_protocol.json").read_text())
for path, digest in original["files"].items():
    assert sha(path) == digest, path
assert (ROOT / "audit_exit.txt").read_text() == "1"
paths = [
    ROOT / name
    for name in (
        "source/audit_recovery.py",
        "source/prepare_audit_recovery.py",
        "source/gradient_replay_diagnosis.py",
        "audit_protocol.json",
        "audit_failure.json",
        "audit.log",
        "audit_exit.txt",
        "gradient_replay_diagnosis.json",
        "gradient_replay_diagnosis.log",
        "gradient_replay_diagnosis_exit.txt",
    )
]
paths.append(Path("research/fp32_classifier_profile_audit_recovery.md"))
(ROOT / "audit_recovery_protocol.json").write_text(
    json.dumps(
        dict(
            files={p.as_posix(): sha(p) for p in paths},
            optimizer_updates=0,
            prospective_scientific_gates_changed=False,
            posthoc_replay_global_tolerance=1e-5,
            posthoc_replay_per_tensor_tolerance=1e-4,
        ),
        indent=2,
    )
    + "\n"
)
print("Audit recovery frozen; original study, failed audit and diagnosis preserved")
