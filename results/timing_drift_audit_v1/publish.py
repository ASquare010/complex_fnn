"""Check analysis against original records and publish the drift audit."""

import csv
import datetime as dt
import statistics as st
from pathlib import Path

from audit import ROOT, SOURCE, read, sha, write

p = read(ROOT / "protocol.json")
a = read(ROOT / "analysis.json")
assert (ROOT / "audit_exit.txt").read_text().strip() == "0"
for group in ("input_hashes", "sources"):
    for path, digest in p[group].items():
        assert sha(path) == digest, path
assert len(a["runs"]) == 12
for run in a["runs"]:
    folder = SOURCE / f"case{run['index']:02d}"
    c = read(folder / "case.json")["measurement"]
    records = c["records"][20:]
    # Independent linear lookup checks the audit's binary interval assignment.
    counts = [0] * 30
    states = [{} for _ in counts]
    for row in csv.reader((folder / "telemetry.csv").read_text().splitlines()):
        stamp, gpu, _, state, *_ = [v.strip() for v in row]
        if int(gpu) != 0:
            continue
        at = int(dt.datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f").timestamp() * 1e9)
        matches = [i for i, r in enumerate(records) if r["at_ns"] <= at <= r["until_ns"]]
        assert len(matches) <= 1
        if matches:
            j = matches[0] // 26
            counts[j] += 1
            states[j][state] = states[j].get(state, 0) + 1
    for j, window in enumerate(run["windows"]):
        assert counts[j] == window["telemetry_samples"]
        assert states[j] == window["pstates"]
        subset = records[j * 26 : (j + 1) * 26]
        for key in ("event_update_ms", "wall_update_ms"):
            assert st.mean(r[key] for r in subset) == window[key]["mean"]
    assert sum(counts) == run["active_samples"]
last = a["runs"][11]
assert all(w["pstates"] == {"P0": w["telemetry_samples"]} for w in last["windows"][:20])
assert set(last["windows"][20]["pstates"]) == {"P0", "P4"}
assert all(w["pstates"] == {"P4": w["telemetry_samples"]} for w in last["windows"][21:])
report = Path("research/timing_drift_audit_results.md")
text = report.read_text(encoding="utf-8")
text = text.replace(
    "## Next experiment",
    """## Localized failure

The last ordinary control (TinyStories seed127) has only P0 samples through
update540. Its541-566 window contains both P0 and P4; all subsequent sampled
windows contain only P4. The three original block means are78.04,79.94 and142.00
ms/update, while active-sample median SM clocks are1680,1620 and915MHz. Its
stability ratio1.8196 fails the unchanged1.15 limit. This is a measured change
in the control's execution conditions, not evidence of candidate instability.
The first WikiText pair also has different clock distributions; its16.7% runtime
overhead remains a failed observation, not a corrected or excused measurement.

A separate linear interval lookup independently reproduced every window's sample
count and P-state composition; original records reproduced all window timing
means. All12 runs remain included. The device-state cause remains unestablished.

## Next experiment""",
)
report.write_text(text, encoding="utf-8")
current, readme = Path("research/CURRENT_STATE.md"), Path("README.md")
assert sha(current) == sha(ROOT / "CURRENT_STATE.before.md")
assert sha(readme) == sha(ROOT / "README.before.md")
head, body = current.read_text(encoding="utf-8").split("\n\n", 1)
body = body.replace("## Latest:", "## Previous:", 1)
current.write_text(
    head
    + """

## Latest: timing-drift audit

[H139](timing_drift_audit_results.md): verified all12 H138 runs and9,360 measured
updates with zero GPU training. The final ordinary control transitions P0 to P4
around updates541-566; its last timing block rises to142ms from78-80ms. The first
WikiText pair also has different clock distributions. Cause remains unknown;
H138 remains failed and no candidate is promoted.

Next: prospectively freeze balanced, temporally interleaved complete-update
comparisons at saved states, retaining every round and the15% runtime limit.
Do not repeat long convergence runs to chase favorable timing. This must remain
separate from H138 and cannot by itself prove deployment throughput or a new FFN.
The broader VRAM/parameter-efficiency research goal remains open.

"""
    + body,
    encoding="utf-8",
)
head, body = readme.read_text(encoding="utf-8").split("\n\n", 1)
body = body.replace("Latest:", "Earlier:", 1)
readme.write_text(
    head
    + """

Latest: [H139 timing-drift audit](research/timing_drift_audit_results.md) identifies
a P0-to-P4 transition in H138's unstable control. Zero further GPU training;
H138 remains failed. See the report for the next controlled timing experiment.

"""
    + body,
    encoding="utf-8",
)
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file() and f.name not in ("receipt.json", "publish.log", "publish_exit.txt")
]
files += [report, current, readme, Path("research/timing_drift_audit_plan.md")]
receipt = dict(
    study="H139",
    status="EVIDENCE_VERIFIED",
    prior_gate_unchanged=True,
    training_updates=0,
    backwards=0,
    goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files},
)
write(ROOT / "receipt.json", receipt)
for path, digest in receipt["files"].items():
    assert sha(path) == digest
print("Verified telemetry assignment; report and provenance published. H138 stays failed.")
