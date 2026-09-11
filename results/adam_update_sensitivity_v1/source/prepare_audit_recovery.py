"""Freeze completed continuation without rewriting the original failed exit."""

import json
from pathlib import Path

from results.adam_update_sensitivity_v1.source.prepare import ROOT, hashes, read, sha

assert not (ROOT / "audit_protocol.json").exists()
assert (ROOT / "study_exit.txt").read_text().strip() == "1"
assert all((ROOT / f"{s}_exit.txt").read_text().strip() == "0" for s in ("recover", "analyze"))
recovery = read(ROOT / "recovery_protocol.json")
hashes(recovery["files"])
result = read(ROOT / "result.json")
assert result["status"] == "COMPLETE" and len(result["cases"]) == 12
assert result["cases"][:6] == recovery["cpu_cases"]
paths = [
    ROOT / name
    for name in (
        "protocol.json",
        "recovery_protocol.json",
        "result.json",
        "analysis.json",
        "environment.json",
        "recover.log",
        "recover_exit.txt",
        "analyze.log",
        "analyze_exit.txt",
    )
]
for case in result["cases"]:
    paths += [Path(case[k]["path"]) for k in ("clipped", "updated")]
    paths.append(ROOT / "runs" / case["label"] / "result.json")
(ROOT / "audit_protocol.json").write_text(
    json.dumps(
        dict(
            files={p.as_posix(): sha(p) for p in paths},
            tensor_artifacts=24,
            audit_optimizer_steps=0,
            audit_forwards=0,
            audit_backwards=0,
            original_failure_preserved=True,
        ),
        indent=2,
    )
    + "\n",
    encoding="utf-8",
    newline="\n",
)
print("Frozen all 24 artifacts, continuation and analysis for the unchanged auditor.")
