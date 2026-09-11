"""Reconstruct allocated bytes from raw events; reject inconsistent attribution."""

import collections
import gzip
import json
from pathlib import Path

ROOT = Path("results/backward_allocation_v1")


def read(path):
    return json.loads(path.read_text())


def unpack(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def site(block):
    frames = block.get("frames", [])
    for frame in frames:
        filename = frame["filename"].replace("\\", "/")
        if "/src/" in filename or filename.endswith("autograd/graph.py"):
            return (
                filename.split("/complex_fnn/")[-1] + ":" + str(frame["line"]) + " " + frame["name"]
            )
    return "unattributed native/backward allocation"


def reconstruct(case):
    folder = ROOT / "runs" / case["label"]
    initial, final = [unpack(folder / n) for n in ("initial.json.gz", "final.json.gz")]
    live = {
        b["address"]: b
        for s in initial["segments"]
        for b in s["blocks"]
        if b["state"] == "active_allocated"
    }
    size = sum(b["size"] for b in live.values())
    assert size == case["baseline"]
    peak, index, peak_live, trigger = size, -1, dict(live), None
    events = final["device_traces"][0]
    assert len(events) == case["events"] < 100000
    marker_map = {m["index"]: m for m in case["markers"]}
    for i, event in enumerate(events):
        action = event["action"]
        if action == "alloc":
            assert event["addr"] not in live
            live[event["addr"]] = dict(size=event["size"], frames=event.get("frames", []))
            size += event["size"]
        elif action == "free_requested":
            old = live.pop(event["addr"])
            assert old["size"] == event["size"]
            size -= event["size"]
        if size > peak:
            peak, index, peak_live, trigger = size, i, dict(live), event
        if i in marker_map:
            assert size == marker_map[i]["allocated"]
        assert size >= 0
    assert size == case["final_allocated"]
    assert size == sum(
        b["size"]
        for s in final["segments"]
        for b in s["blocks"]
        if b["state"] == "active_allocated"
    )
    assert peak == case["peak"]
    before = [m for m in case["markers"] if m["index"] <= index]
    after = [m for m in case["markers"] if m["index"] > index]
    groups = collections.Counter()
    for b in peak_live.values():
        groups[site(b)] += b["size"]
    return dict(
        label=case["label"],
        peak_bytes=peak,
        peak_index=index,
        previous_marker=before[-1]["name"] if before else None,
        next_marker=after[0]["name"] if after else None,
        trigger_size=trigger["size"],
        trigger_site=site(trigger),
        trigger_frames=trigger.get("frames", []),
        peak_live_groups=dict(groups.most_common()),
        live_allocations=len(peak_live),
        accounting_passed=True,
        gradient_global=case["gradient_global"],
        gradient_max_tensor=case["gradient_max_tensor"],
    )


def run():
    r = read(ROOT / "result.json")
    rows = [reconstruct(c) for c in r["cases"]]
    (ROOT / "summary.json").write_text(
        json.dumps(
            dict(
                rows=rows,
                backwards=12,
                training_updates=0,
                diagnostic_targets=49152,
                broad_goal_achieved=False,
            ),
            indent=2,
        )
        + "\n"
    )
    for row in rows:
        print(
            row["label"],
            round(row["peak_bytes"] / 2**20, 3),
            row["previous_marker"],
            row["next_marker"],
            row["trigger_site"],
            flush=True,
        )
        print([(k, round(v / 2**20, 3)) for k, v in list(row["peak_live_groups"].items())[:5]])


if __name__ == "__main__":
    run()
