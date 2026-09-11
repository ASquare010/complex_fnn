"""Seal collected measurements and the independent CPU audit before analysis."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/paired_workspace_timing_v1")
p = read(ROOT / "protocol.json")
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
assert (ROOT / "worker_exit.txt").read_text().strip() == "0"
assert not (ROOT / "audit_protocol.json").exists()
files = [
    f
    for f in ROOT.iterdir()
    if f.is_file() and f.suffix in (".py", ".json", ".jsonl", ".pt", ".csv", ".stderr")
]
(ROOT / "audit_protocol.json").write_text(
    json.dumps(dict(files={f.as_posix(): sha(f) for f in files}), indent=2) + "\n", encoding="utf-8"
)
