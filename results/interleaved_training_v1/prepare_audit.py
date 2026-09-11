"""Seal raw segment JSON and all artifacts without re-encoding large tensors' metadata."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/interleaved_training_v1")
p = read(ROOT / "protocol.json")
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
assert (ROOT / "worker_exit.txt").read_text().strip() == "0"
assert not (ROOT / "result.json").exists() and not (ROOT / "audit_protocol.json").exists()
with (ROOT / "result.json").open("x", encoding="utf-8") as stream:
    stream.write('{"cases":[')
    for i in range(24):
        path = ROOT / f"case{i:02d}/case.json"
        if i:
            stream.write(",")
        stream.write(path.read_text().strip())
    stream.write('],"training_updates":720,"backwards":720}\n')
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and "unused_cache" not in f.parts
    and f.suffix in (".py", ".json", ".pt", ".csv", ".stderr")
]
write_json(ROOT / "audit_protocol.json", dict(files={f.as_posix(): sha(f) for f in files}))
