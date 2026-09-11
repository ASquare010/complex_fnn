"""Freeze all completed long-run states before independent replay/scoring."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/compact_long_training_v1")
assert not (ROOT / "audit_protocol.json").exists()
p, r = read(ROOT / "protocol.json"), read(ROOT / "result.json")
hashes(p["sources"])
assert (ROOT / "study_exit.txt").read_text().strip() == "0"
files = [
    ROOT / n
    for n in ("protocol.json", "result.json", "study.log", "study_exit.txt", "environment.json")
]
for row in r["cases"]:
    c = row["measurement"]
    files.extend(
        Path(v["path"])
        for v in [c["initial_probe"], *c["intermediate_checkpoints"], c["checkpoint"]]
    )
    files.extend(
        [ROOT / row["arm"] / "runs" / c["label"] / n for n in ("metrics.json", "history.jsonl")]
    )
(ROOT / "audit_protocol.json").write_text(
    json.dumps(
        dict(
            files={f.as_posix(): sha(f) for f in files},
            audit_backwards=6,
            native_scores=20,
            training_batches=4800,
        ),
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
