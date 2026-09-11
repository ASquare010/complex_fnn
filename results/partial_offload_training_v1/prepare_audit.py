"""Seal all sixteen completed short continuations."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/partial_offload_training_v1")
p = read(ROOT / "protocol.json")
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
assert not (ROOT / "audit_protocol.json").exists()
cases = []
for row in p["schedule"]:
    assert (ROOT / f"case{row['index']:02d}_exit.txt").read_text().strip() == "0"
    cases.append(read(ROOT / f"case{row['index']:02d}" / "case.json"))
write_json(
    ROOT / "result.json",
    dict(cases=cases, training_updates=480, backwards=480, training_targets=1966080),
)
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and "unused_cache" not in f.parts
    and f.suffix in (".py", ".json", ".pt", ".csv", ".stderr")
]
write_json(ROOT / "audit_protocol.json", dict(files={f.as_posix(): sha(f) for f in files}))
