"""Freeze completed states before independent audit, with no GPU work."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/compact_training_v1")
assert not (ROOT / "audit_protocol.json").exists()
p, r = read(ROOT / "protocol.json"), read(ROOT / "result.json")
hashes(p["sources"])
assert (ROOT / "study_exit.txt").read_text().strip() == "0"
files = [
    ROOT / n
    for n in ("protocol.json", "result.json", "qualification.json", "study.log", "study_exit.txt")
]
for row in r["cases"]:
    c = row["measurement"]
    files.extend(Path(c[k]["path"]) for k in ("first_gradients", "first_state", "final_state"))
    files.append(ROOT / row["arm"] / "runs" / c["label"] / "result.json")
(ROOT / "audit_protocol.json").write_text(
    json.dumps(
        dict(
            files={f.as_posix(): sha(f) for f in files},
            audit_backwards=0,
            native_scores=4,
            batches=120,
        ),
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
