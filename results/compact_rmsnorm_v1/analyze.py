"""Validate accounting and apply the prospective gates to every arm."""

import gzip
import json
import statistics as st
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/compact_rmsnorm_v1")
p, r = [read(ROOT / n) for n in ("protocol.json", "result.json")]
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
assert r["qualification"]["passed"] and r["qualification"]["backwards"] == 6
assert len(r["cases"]) == 8 and sum(c["measurement"]["backwards"] for c in r["cases"]) + 6 == 86
assert len(r["boundaries"]) == 10 and all(
    b["gpu"] == dict(allocated=0, reserved=0) for b in r["boundaries"]
)
for row in r["cases"]:
    c = row["measurement"]
    assert len(c["rows"]) == 10 and len(c["phases"]) == 43
    assert c["parameters"] == 9099648 and c["state_unchanged"] and c["training_updates"] == 0
    assert c["global_error"] <= 1e-5 and c["tensor_error"] <= 1e-4 and c["loss_error"] <= 1e-6
    assert sha(c["gradient"]["path"]) == c["gradient"]["sha256"]
    assert read(ROOT / row["normalization"] / "runs" / c["label"] / "result.json") == c
    assert c["peak"] == max(v["peak_allocated_bytes"] for v in c["phases"])
    assert c["host_allocated_peak"] == max(v["host"]["allocated_bytes.peak"] for v in c["phases"])
    for key in ("wall_ms", "event_sum_ms"):
        values = [v[key] for v in c["rows"][3:]]
        assert c["timing"][key] == dict(
            mean=st.mean(values), median=st.median(values), variance=st.variance(values)
        )
    if c["mode"] == "staged":
        assert all(v["hook_count"] == 50 and v["grad_bytes_after_backward"] == 0 for v in c["rows"])
comparisons = []
for f in p["fixtures"]:
    peers = [row for row in r["cases"] if row["measurement"]["dataset"] == f["dataset"]]
    baseline = next(
        row["measurement"]
        for row in peers
        if row["normalization"] == "ordinary" and row["measurement"]["mode"] == "resident"
    )
    for row in peers:
        c = row["measurement"]
        memory = c["peak"] / baseline["peak"]
        event = c["timing"]["event_sum_ms"]["median"] / baseline["timing"]["event_sum_ms"]["median"]
        wall = c["timing"]["wall_ms"]["median"] / baseline["timing"]["wall_ms"]["median"]
        gates = dict(
            memory=memory <= p["memory_ratio_max"],
            event=event <= p["time_ratio_max"],
            wall=wall <= p["time_ratio_max"],
            host=c["host_allocated_peak"] <= p["host_peak_max"],
        )
        comparisons.append(
            dict(
                dataset=f["dataset"],
                normalization=row["normalization"],
                gradients=c["mode"],
                peak_mib=c["peak"] / 2**20,
                host_mib=c["host_allocated_peak"] / 2**20,
                memory_ratio=memory,
                event_ratio=event,
                wall_ratio=wall,
                gates=gates,
                passed=all(gates.values()),
            )
        )
combined = [
    c for c in comparisons if c["normalization"] == "compact" and c["gradients"] == "staged"
]
summary = dict(
    comparisons=comparisons,
    combined_qualified=all(c["passed"] for c in combined),
    backwards=86,
    training_updates=0,
    broad_goal_achieved=False,
)
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
for name in ("result.json", "summary.json"):
    (ROOT / (name + ".gz")).write_bytes(gzip.compress((ROOT / name).read_bytes(), mtime=0))
for c in comparisons:
    print(
        c["dataset"],
        c["normalization"],
        c["gradients"],
        round(c["peak_mib"], 3),
        "MiB",
        round(100 * (1 - c["memory_ratio"]), 3),
        "% saved",
        round(c["event_ratio"], 4),
        "event ratio",
        c["gates"],
    )
print("Combined qualified:", summary["combined_qualified"])
