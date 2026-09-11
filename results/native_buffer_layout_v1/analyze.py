"""Check exactness, budgets and resource gates; retain every comparison."""

import gzip
import json
import statistics as st
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/native_buffer_layout_v1")
p, r, a = [read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json")]
for key in ("sources", "input_hashes", "maintained_files"):
    hashes(p[key])
hashes(read(ROOT / "audit_protocol.json")["files"])
assert r["qualification"]["passed"] and r["qualification"]["backwards"] == 18
assert len(r["qualification"]["records"]) == 8 and r["qualification"]["full_shape"]["passed"]
assert all(
    all(x["bitwise_equal"]) and x["finite"] and x["inputs_unchanged"]
    for x in r["qualification"]["records"]
)
assert r["backwards"] == 58 and a["backwards"] == 4
assert r["training_updates"] == a["training_updates"] == 0
assert len(r["cases"]) == len(a["cases"]) == 4
assert r["diagnostic_targets"] == 163840
assert len(r["boundaries"]) + len(a["boundaries"]) == 11
assert all(x["gpu"] == dict(allocated=0, reserved=0) for x in r["boundaries"] + a["boundaries"])
metrics = []
for row in r["cases"]:
    c = row["measurement"]
    assert c["backwards"] == 10 and c["training_updates"] == 0 and c["parameters"] == 9099648
    assert c["state_unchanged"] and len(c["rows"]) == 10
    assert sha(c["gradient"]["path"]) == c["gradient"]["sha256"]
    assert c["peak"] == max(x["peak_allocated_bytes"] for x in c["phases"])
    for key, timing in c["timing"].items():
        values = [x[key] for x in c["rows"][3:]]
        assert timing == dict(
            mean=st.mean(values), median=st.median(values), variance=st.variance(values)
        )
    matching = next(
        x for x in a["cases"] if x["dataset"] == c["dataset"] and x["arm"] == row["arm"]
    )
    metrics.append(
        dict(
            dataset=c["dataset"],
            arm=row["arm"],
            peak_mib=c["peak"] / 2**20,
            event_ms=c["timing"]["event_sum_ms"]["median"],
            wall_ms=c["timing"]["wall_ms"]["median"],
            host_mib=c["host_allocated_peak"] / 2**20,
            exact=matching["passed"],
            peak_phases=[x["phase"] for x in c["phases"] if x["peak_allocated_bytes"] == c["peak"]],
        )
    )
comparisons = []
for dataset in ("wikitext2", "tinystories"):
    base = next(x for x in metrics if x["dataset"] == dataset and x["arm"] == "native")
    candidate = next(x for x in metrics if x["dataset"] == dataset and x["arm"] == "reuse")
    ratios = {k: candidate[k] / base[k] for k in ("peak_mib", "event_ms", "wall_ms")}
    gates = dict(
        exact=base["exact"] and candidate["exact"],
        memory=ratios["peak_mib"] <= p["memory_ratio_max"],
        event=ratios["event_ms"] <= p["time_ratio_max"],
        wall=ratios["wall_ms"] <= p["time_ratio_max"],
        host=candidate["host_mib"] * 2**20 <= p["host_peak_max"],
    )
    comparisons.append(
        dict(dataset=dataset, ratios=ratios, gates=gates, passed=all(gates.values()))
    )
s = dict(
    metrics=metrics,
    comparisons=comparisons,
    passed=a["passed"] and all(x["passed"] for x in comparisons),
    backwards=62,
    training_updates=0,
    model_probe_targets=163840,
    replay_targets=16384,
    broad_goal_achieved=False,
)

(ROOT / "summary.json").write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
for name in ("result.json", "audit.json", "summary.json"):
    (ROOT / (name + ".gz")).write_bytes(gzip.compress((ROOT / name).read_bytes(), mtime=0))
print(json.dumps(s, indent=2))
