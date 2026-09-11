"""Freeze every completed profile state and metric before independent checks."""

import json
from pathlib import Path

from results.optimizer_memory_v1.source.prepare import ROOT, hashes, read, sha

assert not (ROOT / "audit_protocol.json").exists()
assert (ROOT / "profile_exit.txt").read_text().strip() == "0"
p, result = read(ROOT / "protocol.json"), read(ROOT / "result.json")
hashes(p["sources"])
assert result["status"] == "COMPLETE" and len(result["cases"]) == 12
paths = [
    ROOT / n
    for n in ("protocol.json", "result.json", "environment.json", "profile.log", "profile_exit.txt")
]
for case in result["cases"]:
    paths += [Path(case[k]["path"]) for k in ("first_gradients", "first_state", "final_state")]
    paths.append(ROOT / "runs" / case["label"] / "result.json")
(ROOT / "audit_protocol.json").write_text(
    json.dumps(
        dict(
            files={f.as_posix(): sha(f) for f in paths},
            tensor_artifacts=36,
            native_scores=12,
            audit_updates=0,
            audit_backwards=0,
        ),
        indent=2,
    )
    + "\n",
    encoding="utf-8",
    newline="\n",
)
print("Frozen 36 artifacts, 360 updates and all phase measurements.")
