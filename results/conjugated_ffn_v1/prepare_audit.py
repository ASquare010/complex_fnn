"""Seal all fit/data artifacts before a differently batched explicit-model replay."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/conjugated_ffn_v1")
p = read(ROOT / "protocol.json")
hashes(p["sources"])
hashes(p["maintained_files"])
assert (ROOT / "worker_exit.txt").read_text().strip() == "0"
assert len(list(ROOT.glob("fit*.pt"))) == 54 and len(list(ROOT.glob("data_*.pt"))) == 9
assert not (ROOT / "audit_protocol.json").exists()
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file() and "unused_cache" not in f.parts and f.suffix in (".py", ".json", ".pt", ".md")
]
write_json(
    ROOT / "audit_protocol.json",
    dict(
        files={f.as_posix(): sha(f) for f in files},
        score_relative_tolerance=1e-5,
        positive_control="Both wide controls must beat zero by20%; failure means inconclusive learning probe.",
    ),
)
