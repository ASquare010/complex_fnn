"""Freeze both finished workspace policies before replay."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/blas_workspace_v1")
p = read(ROOT / "protocol.json")
for key in ("sources", "input_hashes", "maintained_files"):
    hashes(p[key])
files = [ROOT / "protocol.json"]
for mode in ("high", "low"):
    assert (ROOT / (mode + "_exit.txt")).read_text().strip() == "0"
    r = read(ROOT / (mode + "_result.json"))
    files += [ROOT / (mode + x) for x in ("_result.json", ".log", "_exit.txt")]
    files += [Path(row["measurement"]["gradient"]["path"]) for row in r["cases"]]
assert not (ROOT / "audit_protocol.json").exists()
(ROOT / "audit_protocol.json").write_text(
    json.dumps(
        dict(files={f.as_posix(): sha(f) for f in files}, backwards=8, training_updates=0), indent=2
    )
    + "\n",
    encoding="utf-8",
)
