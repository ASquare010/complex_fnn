"""Freeze complete fresh-training evidence before independent audit."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/fp32_training_replication_recovery_v1")
assert not (ROOT / "audit_protocol.json").exists()
result = json.loads((ROOT / "result.json").read_text())
assert result["status"] == "COMPLETE" and len(result["cases"]) == 18
assert (ROOT / "study_exit.txt").read_text().strip() == "0"
initials = json.loads((ROOT / "initializations.json").read_text())
paths = [
    ROOT / name
    for name in (
        "source/audit.py",
        "source/prepare_audit.py",
        "protocol.json",
        "result.json",
        "qualification.json",
        "initializations.json",
    )
]
paths.append(Path("results/fp32_decoder_resource_v1/source/audit.py"))
paths.extend(Path(r["path"]) for r in initials["initializations"])
for case in result["cases"]:
    paths.extend(Path(case[k]["path"]) for k in ("initial_probe", "checkpoint"))
    paths.extend(Path(c["path"]) for c in case["intermediate_checkpoints"])
hashes = {}
for path in paths:
    with path.open("rb") as stream:
        hashes[path.as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
(ROOT / "audit_protocol.json").write_text(
    json.dumps(
        dict(
            files=hashes,
            tensor_artifacts=78,
            gradient_replay_backward_passes=12,
            native_validation_scores=60,
            regenerated_initializations=6,
            trained_checkpoints=54,
            optimizer_updates=0,
            replay_global_limit=1e-5,
            replay_tensor_limit=1e-4,
            loss_and_score_relative_limit=1e-6,
        ),
        indent=2,
    )
    + "\n"
)
print("Frozen audit: 78 tensor artifacts, 12 FP32 replays and 60 native scores")
