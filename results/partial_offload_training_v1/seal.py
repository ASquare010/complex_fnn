"""Stream existing case JSON verbatim after an aggregate-encoder crash."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/partial_offload_training_v1")
p = read(ROOT / "protocol.json")
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
assert (ROOT / "prepare_audit_exit.txt").read_text().strip() == "3221225477"
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
    stream.write('],"training_updates":480,"backwards":480,"training_targets":1966080}\n')
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and "unused_cache" not in f.parts
    and f.suffix in (".py", ".json", ".pt", ".csv", ".stderr", ".md")
]
write_json(ROOT / "audit_protocol.json", dict(files={f.as_posix(): sha(f) for f in files}))
print("Sealed sixteen unchanged cases without aggregate re-encoding")
