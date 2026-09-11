"""Apply fixed memory/time gates and report the four measured cases."""

import gzip
import json
import statistics as st
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/gradient_staging_v1")
p, r = [read(ROOT / n) for n in ("protocol.json", "result.json")]
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
assert r["qualification"]["passed"] and r["qualification"]["backwards"] == 2
assert sum(c["backwards"] for c in r["cases"]) + 2 == 42
assert len(r["boundaries"]) == 6 and all(
    b["gpu"] == dict(allocated=0, reserved=0) for b in r["boundaries"]
)
comparisons = []
for c in r["cases"]:
    assert c["state_unchanged"] and c["training_updates"] == 0 and c["parameters"] == 9099648
    assert c["global_error"] <= 1e-5 and c["tensor_error"] <= 1e-4 and c["loss_error"] <= 1e-6
    assert sha(c["gradient"]["path"]) == c["gradient"]["sha256"]
    assert len(c["rows"]) == 10 and len(c["phases"]) == 43
    assert c["peak"] == max(v["peak_allocated_bytes"] for v in c["phases"])
    for key in ("wall_ms", "event_sum_ms"):
        values = [v[key] for v in c["rows"][3:]]
        assert c["timing"][key] == dict(
            mean=st.mean(values), median=st.median(values), variance=st.variance(values)
        )
    if c["mode"] == "staged":
        assert c["pinned_gradient_payload"] == 36398592
        assert all(v["grad_bytes_after_backward"] == 0 and v["hook_count"] == 50 for v in c["rows"])
    else:
        assert all(v["grad_bytes_after_backward"] == 36398592 for v in c["rows"])
for f in p["fixtures"]:
    resident, staged = [
        next(c for c in r["cases"] if c["dataset"] == f["dataset"] and c["mode"] == m)
        for m in ("resident", "staged")
    ]
    memory = staged["peak"] / resident["peak"]
    event = (
        staged["timing"]["event_sum_ms"]["median"] / resident["timing"]["event_sum_ms"]["median"]
    )
    wall = staged["timing"]["wall_ms"]["median"] / resident["timing"]["wall_ms"]["median"]
    gates = dict(
        memory=memory <= p["memory_ratio_max"],
        event_time=event <= p["time_ratio_max"],
        wall_time=wall <= p["time_ratio_max"],
        host_memory=staged["host_allocated_peak"] <= p["host_peak_max"],
    )
    comparisons.append(
        dict(
            dataset=f["dataset"],
            resident_mib=resident["peak"] / 2**20,
            staged_mib=staged["peak"] / 2**20,
            memory_ratio=memory,
            event_ratio=event,
            wall_ratio=wall,
            host_mib=staged["host_allocated_peak"] / 2**20,
            gates=gates,
            passed=all(gates.values()),
        )
    )
result = dict(
    comparisons=comparisons,
    passed=all(c["passed"] for c in comparisons),
    backwards=42,
    training_updates=0,
    broad_goal_achieved=False,
)
(ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
for n in ("result.json", "summary.json"):
    (ROOT / (n + ".gz")).write_bytes(gzip.compress((ROOT / n).read_bytes(), mtime=0))
print(json.dumps(result, indent=2))
