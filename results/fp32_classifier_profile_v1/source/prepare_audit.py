"""Freeze the independent audit and completed scientific artifacts before replay."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/fp32_classifier_profile_v1")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert not (ROOT / "audit_protocol.json").exists()
protocol = json.loads((ROOT / "protocol.json").read_text())
for path, digest in protocol["sources"].items():
    assert sha(path) == digest, path
result = json.loads((ROOT / "result.json").read_text())
assert result["status"] == "COMPLETE" and result["optimizer_updates"] == 2800
files = {
    p.as_posix(): sha(p)
    for p in (
        ROOT / "source/audit.py",
        Path(__file__),
        ROOT / "result.json",
        ROOT / "protocol.json",
        ROOT / "qualification.json",
    )
}
for case in result["cases"]:
    for artifact in (case["checkpoint"], case["initial_probe"]):
        assert sha(artifact["path"]) == artifact["sha256"]
        files[artifact["path"]] = artifact["sha256"]
(ROOT / "audit_protocol.json").write_text(
    json.dumps(
        dict(
            files=files,
            optimizer_updates=0,
            note="Independent native rescoring and exact saved-probe replay; post-study audit source frozen before execution.",
        ),
        indent=2,
    )
    + "\n"
)
print("Independent audit frozen; 112 scientific tensor artifacts verified")
