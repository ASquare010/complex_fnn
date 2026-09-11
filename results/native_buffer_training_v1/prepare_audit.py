"""Freeze all eight completed training cases before independent audit."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/native_buffer_training_v1")
p = read(ROOT / "protocol.json")
for key in ("sources", "input_hashes", "maintained_files"):
    hashes(p[key])
files = [ROOT / "protocol.json"]
cases = []
boundaries = []
for mode in ("ordinary", "deterministic"):
    assert (ROOT / (mode + "_exit.txt")).read_text().strip() == "0"
    r = read(ROOT / (mode + "_result.json"))
    cases += r["cases"]
    boundaries += r["boundaries"]
    files += [ROOT / (mode + x) for x in ("_result.json", "_environment.json", ".log", "_exit.txt")]
for row in cases:
    c = row["measurement"]
    files += [Path(c[key]["path"]) for key in ("first_gradients", "first_state", "final_state")]
    files += [Path(c["final_state"]["path"]).parent / "result.json"]
assert len(cases) == 8
(ROOT / "result.json").write_text(
    json.dumps(
        dict(
            cases=cases,
            boundaries=boundaries,
            training_updates=240,
            backwards=240,
            training_targets=983040,
        ),
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
files.append(ROOT / "result.json")
assert not (ROOT / "audit_protocol.json").exists()
(ROOT / "audit_protocol.json").write_text(
    json.dumps(
        dict(
            files={f.as_posix(): sha(f) for f in files},
            backwards=0,
            training_updates=0,
            native_scores=8,
            batches=240,
        ),
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
