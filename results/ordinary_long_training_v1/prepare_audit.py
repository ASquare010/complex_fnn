"""Seal eighteen completed runs before independent native verification."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/ordinary_long_training_v1")
p = read(ROOT / "protocol.json")
for field in ("sources", "input_hashes", "maintained_files", "checkpoint_hashes"):
    hashes(p[field])
assert not (ROOT / "audit_protocol.json").exists()
cases = []
for row in p["schedule"]:
    assert (ROOT / f"case{row['index']:02d}_exit.txt").read_text().strip() == "0"
    cases.append(read(ROOT / f"case{row['index']:02d}" / "case.json"))
write_json(
    ROOT / "result.json",
    dict(
        cases=cases,
        training_updates=14400,
        backwards=14418,
        training_targets=58982400,
        study_scores=72,
    ),
)
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and "unused_cache" not in f.parts
    and f.suffix in (".py", ".json", ".pt", ".jsonl", ".csv", ".stderr")
]
write_json(ROOT / "audit_protocol.json", dict(files={f.as_posix(): sha(f) for f in files}))
