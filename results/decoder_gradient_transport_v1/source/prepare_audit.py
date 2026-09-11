"""Freeze independent forward recapture and arithmetic checks after the study."""

import hashlib
import json
from pathlib import Path

root = Path("results/decoder_gradient_transport_v1")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert not (root / "audit_protocol.json").exists()
protocol = json.loads((root / "protocol.json").read_text())
for path, digest in protocol["sources"].items():
    assert sha(path) == digest, path
result = json.loads((root / "result.json").read_text())
assert result["status"] == "COMPLETE" and len(result["conditions"]) == 30
files = {
    p.as_posix(): sha(p)
    for p in (
        root / "source/audit.py",
        Path(__file__),
        root / "result.json",
        root / "qualification.json",
    )
}
for condition in result["conditions"]:
    for artifact in condition["artifacts"]:
        assert sha(artifact["path"]) == artifact["sha256"]
        files[artifact["path"]] = artifact["sha256"]
(root / "audit_protocol.json").write_text(
    json.dumps(
        dict(
            files=files,
            native_forward_recaptures=30,
            backward_passes=0,
            optimizer_updates=0,
            note="Native Transformer forward with norm hook; saved-tensor sums of squares independent of the main norm helper. No new full gradient replay claim.",
        ),
        indent=2,
    )
    + "\n"
)
print("Frozen zero-backward audit and all 60 tensor artifacts")
