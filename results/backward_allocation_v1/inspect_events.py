"""Read-only diagnosis of allocator marker indexing in saved snapshots."""

from results.backward_allocation_v1.analyze import ROOT, read, unpack

for case in read(ROOT / "result.json")["cases"]:
    folder = ROOT / "runs" / case["label"]
    initial, final = [unpack(folder / n) for n in ("initial.json.gz", "final.json.gz")]
    size = sum(
        b["size"]
        for s in initial["segments"]
        for b in s["blocks"]
        if b["state"] == "active_allocated"
    )
    peak, values = size, []
    events = final["device_traces"][0]
    for event in events:
        size += (
            event["size"]
            if event["action"] == "alloc"
            else -event["size"]
            if event["action"] == "free_requested"
            else 0
        )
        values.append(size)
        peak = max(peak, size)
    bad = [
        dict(
            marker=m,
            event=events[m["index"]]["action"],
            nearby=values[max(0, m["index"] - 1) : m["index"] + 3],
        )
        for m in case["markers"]
        if values[m["index"]] != m["allocated"]
    ]
    print(case["label"], "initial", len(initial["device_traces"][0]), "bad markers", bad[:2])
    print(
        "final",
        size,
        case["final_allocated"],
        "peak",
        peak,
        case["peak"],
        "first events",
        [(e["action"], e["size"]) for e in events[:4]],
    )
