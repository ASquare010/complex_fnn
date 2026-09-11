"""Seal all twelve complete continuations before independent audit."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/paired_complete_training_v1")
p = read(ROOT / "protocol.json")
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
assert not (ROOT / "audit_protocol.json").exists()
cases = []
for row in p["schedule"]:
    assert (ROOT / f"case{row['index']:02d}_exit.txt").read_text().strip() == "0"
    cases.append(read(ROOT / f"case{row['index']:02d}" / "case.json"))
(ROOT / "result.json").write_text(
    json.dumps(
        dict(cases=cases, training_updates=360, backwards=360, training_targets=1474560), indent=2
    )
    + "\n",
    encoding="utf-8",
)
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and "unused_cache" not in f.parts
    and f.suffix in (".json", ".pt", ".csv", ".stderr", ".py")
]
(ROOT / "audit_protocol.json").write_text(
    json.dumps(dict(files={f.as_posix(): sha(f) for f in files}), indent=2) + "\n", encoding="utf-8"
)
