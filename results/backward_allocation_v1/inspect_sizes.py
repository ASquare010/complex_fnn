"""Inspect requested versus allocator block sizes in existing traces only."""

from results.backward_allocation_v1.analyze import ROOT, read, unpack

for c in read(ROOT / "result.json")["cases"]:
    d = unpack(ROOT / "runs" / c["label"] / "final.json.gz")
    print(c["label"])
    for event in [e for e in d["device_traces"][0] if e["action"] == "alloc"][:2]:
        print({k: v for k, v in event.items() if k != "frames"})
    print({k: v for k, v in d["segments"][0]["blocks"][0].items() if k != "frames"})
    print("segments", len(d["segments"]), "events", len(d["device_traces"][0]))
