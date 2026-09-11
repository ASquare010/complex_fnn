"""Offline requested-byte accounting; the original allocated-byte gate stays failed."""

import collections
import gzip
import json
from pathlib import Path

from results.backward_allocation_v1.analyze import ROOT, read, site, unpack
from results.checkpoint_input_offload_v1.source.prepare import hashes, sha


def analyze(c):
    folder = ROOT / "runs" / c["label"]
    initial, final = [unpack(folder / n) for n in ("initial.json.gz", "final.json.gz")]
    live = {
        b["address"]: dict(size=b["requested_size"], frames=b.get("frames", []))
        for s in initial["segments"]
        for b in s["blocks"]
        if b["state"] == "active_allocated"
    }
    size = sum(b["size"] for b in live.values())
    peak, index, peak_live = size, -1, dict(live)
    events = final["device_traces"][0]
    markers = {m["index"] + 1: m for m in c["markers"]}
    marker_rows = []
    for i, e in enumerate(events):
        if e["action"] == "alloc":
            assert e["addr"] not in live
            live[e["addr"]] = e
            size += e["size"]
        elif e["action"] == "free_requested":
            old = live.pop(e["addr"])
            assert old["size"] == e["size"]
            size -= e["size"]
        if size > peak:
            peak, index, peak_live = size, i, dict(live)
        if i in markers:
            m = markers[i]
            assert e["action"] == "snapshot" and size <= m["allocated"]
            marker_rows.append(
                dict(name=m["name"], index=i, requested=size, allocated=m["allocated"])
            )
    expected = sum(
        b["requested_size"]
        for s in final["segments"]
        for b in s["blocks"]
        if b["state"] == "active_allocated"
    )
    assert size == expected
    assert len(marker_rows) == len(c["markers"]) == 19
    assert (
        sum(
            b["size"]
            for s in final["segments"]
            for b in s["blocks"]
            if b["state"] == "active_allocated"
        )
        == c["final_allocated"]
    )
    before = [m for m in marker_rows if m["index"] <= index]
    after = [m for m in marker_rows if m["index"] > index]
    groups = collections.Counter()
    for b in peak_live.values():
        groups[site(b)] += b["size"]
    assert sum(groups.values()) == peak
    return dict(
        label=c["label"],
        requested_peak=peak,
        allocated_peak=c["peak"],
        peak_index=index,
        previous=before[-1]["name"],
        next=after[0]["name"],
        trigger_site=site(events[index]),
        trigger_bytes=events[index]["size"],
        groups=dict(groups.most_common()),
        markers=marker_rows,
        requested_endpoint_verified=True,
        allocated_peak_attribution_verified=False,
        events=len(events),
    )


def run():
    assert not (ROOT / "reanalysis_protocol.json").exists()
    p, r = [read(ROOT / n) for n in ("protocol.json", "result.json")]
    for field in ("sources", "inputs", "maintained_files"):
        hashes(p[field])
    files = [
        *ROOT.glob("runs/*/*"),
        ROOT / "analyze.log",
        ROOT / "analyze_exit.txt",
        ROOT / "inspect_events.log",
        ROOT / "inspect_sizes.log",
        ROOT / "payload.py",
        Path("research/backward_allocation_reanalysis.md"),
    ]
    manifest = {f.as_posix(): sha(f) for f in files if f.is_file()}
    (ROOT / "reanalysis_protocol.json").write_text(
        json.dumps(
            dict(
                files=manifest,
                additional_gpu_backwards=0,
                original_gate="FAIL_ALLOCATED_ACCOUNTING",
                scope="requested payload only",
            ),
            indent=2,
        )
        + "\n"
    )
    rows = [analyze(c) for c in r["cases"]]
    hashes(manifest)
    result = dict(
        rows=rows,
        original_gate="FAIL_ALLOCATED_ACCOUNTING",
        backwards=12,
        training_updates=0,
        diagnostic_targets=49152,
        broad_goal_achieved=False,
    )
    (ROOT / "payload_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(
            row["label"],
            "requested/allocated MiB",
            round(row["requested_peak"] / 2**20, 3),
            round(row["allocated_peak"] / 2**20, 3),
            row["previous"],
            row["next"],
            row["trigger_site"],
        )
        print([(k, round(v / 2**20, 3)) for k, v in list(row["groups"].items())[:6]])
    (ROOT / "payload_summary.json.gz").write_bytes(
        gzip.compress((ROOT / "payload_summary.json").read_bytes(), mtime=0)
    )


if __name__ == "__main__":
    run()
