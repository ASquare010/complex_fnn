"""Seal 16 gradients and all completed probe observations before replay."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import ROOT, hashes, read, sha

assert not (ROOT / "audit_protocol.json").exists()
p, r = read(ROOT / "protocol.json"), read(ROOT / "result.json")
hashes(p["sources"])
assert r["status"] == "COMPLETE" and len(r["cases"]) == 8
assert (ROOT / "profile_exit.txt").read_text().strip() == "0"
paths = [
    ROOT / n
    for n in (
        "protocol.json",
        "qualification.json",
        "result.json",
        "environment.json",
        "profile.log",
        "profile_exit.txt",
    )
]
for c in r["cases"]:
    paths += [
        Path(c["untraced"]["path"]),
        Path(c["traced"]["gradient"]["path"]),
        ROOT / "runs" / c["label"] / "result.json",
    ]
(ROOT / "audit_protocol.json").write_text(
    json.dumps(
        dict(
            files={f.as_posix(): sha(f) for f in paths},
            audit_backwards=8,
            audit_updates=0,
            audit_scores=0,
            replay_gradient_artifacts=8,
        ),
        indent=2,
    )
    + "\n",
    encoding="utf-8",
    newline="\n",
)
print("Frozen 248 full-model probes and 16 gradients before independent replay.")
