"""Aggregate original case JSON verbatim and seal the independent audit."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/batch_scale_long_v1")
p = read(ROOT / "protocol.json")
for field in ("sources", "input_hashes", "maintained_files", "checkpoint_hashes"):
    hashes(p[field])
assert not (ROOT / "result.json").exists() and not (ROOT / "audit_protocol.json").exists()
paths = []
for row in p["schedule"]:
    assert (ROOT / f"case{row['index']:02d}_exit.txt").read_text().strip() == "0"
    paths.append(ROOT / f"case{row['index']:02d}" / "case.json")
with (ROOT / "result.json").open("x", encoding="utf-8") as stream:
    stream.write('{"cases":[')
    for i, path in enumerate(paths):
        if i:
            stream.write(",")
        stream.write(path.read_text(encoding="utf-8").strip())
    stream.write(
        '],"training_updates":9600,"backwards":9612,"training_targets":78643200,"study_scores":48}\n'
    )
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and "unused_cache" not in f.parts
    and f.suffix in (".py", ".json", ".pt", ".jsonl", ".csv", ".stderr")
]
write_json(ROOT / "audit_protocol.json", dict(files={f.as_posix(): sha(f) for f in files}))
