"""Freeze every actual step and the primary analysis before independent audit."""

import json
from pathlib import Path

from results.adam_update_sensitivity_v1.source.prepare import ROOT, read, sha

assert not (ROOT / "audit_protocol.json").exists()
assert all((ROOT / f"{s}_exit.txt").read_text().strip() == "0" for s in ("study", "analyze"))
result = read(ROOT / "result.json")
assert result["status"] == "COMPLETE" and len(result["cases"]) == 12
paths = [
    ROOT / name for name in ("protocol.json", "result.json", "analysis.json", "environment.json")
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
        ),
        indent=2,
    )
    + "\n",
    encoding="utf-8",
    newline="\n",
)
print("Frozen 24 tensor artifacts and all primary analysis outputs.")
