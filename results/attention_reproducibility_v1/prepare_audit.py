"""Freeze the finished policy results before independent CPU verification."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/attention_reproducibility_v1")
p = read(ROOT / "protocol.json")
for key in ("sources", "input_hashes", "maintained_files"):
    hashes(p[key])
files = [ROOT / "protocol.json"]
for mode in p["policies"]:
    assert (ROOT / (mode + "_exit.txt")).read_text().strip() == "0"
    r = read(ROOT / mode / "result.json")
    files += [
        ROOT / mode / "result.json",
        ROOT / mode / "environment.json",
        ROOT / (mode + ".log"),
        ROOT / (mode + "_exit.txt"),
    ]
    files += [Path(row["artifact"]["path"]) for row in r["cases"] if "artifact" in row]
assert not (ROOT / "audit_protocol.json").exists()
(ROOT / "audit_protocol.json").write_text(
    json.dumps(
        dict(files={f.as_posix(): sha(f) for f in files}, backwards=0, training_updates=0), indent=2
    )
    + "\n",
    encoding="utf-8",
)
