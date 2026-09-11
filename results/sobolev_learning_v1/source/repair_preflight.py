"""Preserve and correct the pre-protocol path-type failure; no training ran."""

import hashlib
import json
from pathlib import Path

root = Path("results/sobolev_learning_v1")
assert not (root / "protocol.json").exists()
assert not (root / "qualification.json").exists()
assert not (root / "pairs").exists()
original = root / "source/study.py"
archive = root / "preflight_failure/study.py"
archive.parent.mkdir()
archive.write_bytes(original.read_bytes())
files = [archive, root / "study.log", root / "study_exit.txt", root / "failure.json"]
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
source = original.read_text()
source = source.replace("sha256(path)", "sha256(Path(path))")
source = source.replace('sha256(teacher_record["path"])', 'sha256(Path(teacher_record["path"]))')
source = source.replace('sha256(original["path"])', 'sha256(Path(original["path"]))')
original.write_text(source)
(root / "preflight_recovery.json").write_text(
    json.dumps(
        {
            "before": before,
            "neural_updates_before_failure": 0,
            "protocol_written_before_failure": False,
            "change": "Wrap string paths in pathlib.Path before calling maintained sha256",
        },
        indent=2,
    )
    + "\n"
)
