"""Freeze completed repeats and their artifacts before independent audit."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/training_variability_v1")
assert not (ROOT / "audit_protocol.json").exists()
assert (ROOT / "study_exit.txt").read_text().strip() == "0"
result = json.loads((ROOT / "result.json").read_text())
protocol = json.loads((ROOT / "protocol.json").read_text())
assert result["status"] == "COMPLETE" and len(result["cases"]) == 4
paths = [
    ROOT / n for n in ("protocol.json", "result.json", "source/audit.py", "source/prepare_audit.py")
]
for case in result["cases"]:
    paths.extend(Path(case[k]["path"]) for k in ("initial_probe", "checkpoint"))
    paths.extend(Path(c["path"]) for c in case["intermediate_checkpoints"])
hashes = protocol["input_hashes"].copy()
for path in paths:
    with path.open("rb") as stream:
        hashes[path.as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
(ROOT / "audit_protocol.json").write_text(
    json.dumps(
        dict(
            files=hashes,
            new_tensor_artifacts=16,
            original_tensor_artifacts=9,
            native_validation_scores=13,
            gradient_replay_backward_passes=4,
            checkpoint_pair_distances=45,
            initial_gradient_pair_distances=15,
            optimizer_updates=0,
            replay_global_limit=1e-5,
            replay_tensor_limit=1e-4,
            loss_score_limit=1e-6,
        ),
        indent=2,
    )
    + "\n",
    encoding="utf-8",
    newline="\n",
)
print("Frozen 16 new tensor artifacts plus original inputs; four replays and 13 native scores.")
