"""Freeze the completed study and audit before any independent replay/scoring."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/fp32_decoder_resource_v1")
assert not (ROOT / "audit_protocol.json").exists()
result = json.loads((ROOT / "result.json").read_text())
assert result["status"] == "COMPLETE" and len(result["cases"]) == 30
assert (ROOT / "study_exit.txt").read_text().strip() == "0"
paths = [
    ROOT / "source/audit.py",
    Path(__file__),
    ROOT / "result.json",
    ROOT / "qualification.json",
]
for case in result["cases"]:
    paths.extend(Path(case[key]["path"]) for key in ("initial_probe", "checkpoint"))
hashes = {}
for path in paths:
    with path.open("rb") as stream:
        hashes[path.as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
(ROOT / "audit_protocol.json").write_text(
    json.dumps(
        dict(
            files=hashes,
            gradient_replay_backward_passes=24,
            native_validation_scores=36,
            optimizer_updates=0,
            replay_global_limit=1e-5,
            replay_tensor_limit=1e-4,
            loss_and_score_relative_limit=1e-6,
        ),
        indent=2,
    )
    + "\n"
)
print("Audit frozen: 60 tensor artifacts, 24 FP32 gradient replays, 36 native scores")
