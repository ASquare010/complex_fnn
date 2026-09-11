"""Freeze completed scientific outputs before independent native replay."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/native_buffer_loss_v1")
p = read(ROOT / "protocol.json")
for key in ("sources", "input_hashes", "maintained_files"):
    hashes(p[key])
assert (ROOT / "study_exit.txt").read_text().strip() == "0"
r = read(ROOT / "result.json")
files = [
    ROOT / "protocol.json",
    ROOT / "result.json",
    ROOT / "study.log",
    ROOT / "study_exit.txt",
    ROOT / "environment.json",
    ROOT / "qualification.json",
]
for row in r["cases"]:
    files.extend(
        [
            Path(row["measurement"]["gradient"]["path"]),
            Path(row["measurement"]["gradient"]["path"]).parent / "result.json",
        ]
    )
assert not (ROOT / "audit_protocol.json").exists()
(ROOT / "audit_protocol.json").write_text(
    json.dumps(
        dict(
            files={f.as_posix(): sha(f) for f in files},
            backwards=len(r["cases"]),
            training_updates=0,
        ),
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
