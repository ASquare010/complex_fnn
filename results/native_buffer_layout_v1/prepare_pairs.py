"""Freeze the CPU-only post-hoc question separately from original gates."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import read, sha

ROOT = Path("results/native_buffer_layout_v1")
r = read(ROOT / "result.json")
files = [
    ROOT / "analyze_pairs.py",
    ROOT / "prepare_pairs.py",
    ROOT / "result.json",
    ROOT / "audit.json",
    ROOT / "summary.json",
]
files += [Path(row["measurement"]["gradient"]["path"]) for row in r["cases"]]
assert not (ROOT / "pair_protocol.json").exists()
(ROOT / "pair_protocol.json").write_text(
    json.dumps(
        dict(
            files={f.as_posix(): sha(f) for f in files},
            question="Quantify stored native/reuse differences without repeating any GPU work or changing failed gates",
            backwards=0,
            training_updates=0,
        ),
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
