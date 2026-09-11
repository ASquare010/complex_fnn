"""Apply all predeclared complete-job, numerical and quality gates."""

import gzip
import json
import statistics as st
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/native_buffer_training_v1")
p, r, a = [read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json")]
for key in ("sources", "input_hashes", "maintained_files"):
    hashes(p[key])
hashes(read(ROOT / "audit_protocol.json")["files"])
assert (
    len(r["cases"]) == 8
    and r["backwards"] == r["training_updates"] == 240
    and r["training_targets"] == 983040
)
assert (
    a["backwards"] == a["training_updates"] == 0 and a["native_scores"] == 8 and a["batches"] == 240
)
assert len(r["boundaries"]) == 10 and len(a["boundaries"]) == 9
assert all(b["gpu"] == dict(allocated=0, reserved=0) for b in r["boundaries"] + a["boundaries"])
metrics = []
for row in r["cases"]:
    c = row["measurement"]
    assert (
        len(c["updates"]) == 30
        and len(c["phases"]) == 157
        and c["parameters"] == 9099648
        and c["finite"]
    )
    assert c["peak_allocated_bytes"] == max(v["peak_allocated_bytes"] for v in c["phases"])
    for phase in c["phases"]:
        assert (
            sum(phase["storage_bytes"].values()) + phase["unattributed_live_bytes"]
            == phase["live_allocated_bytes"]
        )
    for key in ("wall_ms", "event_sum_ms"):
        values = [v[key] for v in c["updates"][10:]]
        for field, fn in (
            ("median", st.median),
            ("mean", st.mean),
            ("sample_variance", st.variance),
        ):
            assert c["timing"][key][field] == fn(values)
    for key in ("first_gradients", "first_state", "final_state"):
        assert sha(c[key]["path"]) == c[key]["sha256"]
    metrics.append(
        dict(
            dataset=c["fixture"]["dataset"],
            arm=row["arm"],
            peak_mib=c["peak_allocated_bytes"] / 2**20,
            event_ms=c["timing"]["event_sum_ms"]["median"],
            wall_ms=c["timing"]["wall_ms"]["median"],
            nll=c["after_score"]["nll"],
            host_mib=max(v["host"]["allocated_bytes.peak"] for v in c["phases"]) / 2**20,
            stability=max(c["timing_stability"].values()),
        )
    )
comparisons = []
for f in p["fixtures"]:
    peers = [row["measurement"] for row in r["cases"] if row["measurement"]["fixture"] == f]
    for key in ("initial_model_hash", "initial_moment_hash", "initial_sampler_hash"):
        assert len({c[key] for c in peers}) == 1
    orders = [[(x["tokens_hash"], x["targets_hash"]) for x in c["updates"]] for c in peers]
    assert all(v == orders[0] for v in orders)
    candidate = next(x for x in metrics if x["dataset"] == f["dataset"] and x["arm"] == "reuse")
    for arm in ("ordinary", "native", "offload"):
        control = next(x for x in metrics if x["dataset"] == f["dataset"] and x["arm"] == arm)
        ratios = {k: candidate[k] / control[k] for k in ("peak_mib", "event_ms", "wall_ms", "nll")}
        gates = dict(
            audit=a["passed"],
            memory=ratios["peak_mib"] <= p["memory_ratio_max"],
            event=ratios["event_ms"] <= p["time_ratio_max"],
            wall=ratios["wall_ms"] <= p["time_ratio_max"],
            quality=ratios["nll"] <= p["nll_ratio_max"],
            host=candidate["host_mib"] * 2**20 <= p["host_peak_max"],
            stability=all(
                x["stability"] <= p["timing_stability_max"]
                for x in metrics
                if x["dataset"] == f["dataset"]
            ),
        )
        comparisons.append(
            dict(
                dataset=f["dataset"],
                control=arm,
                ratios=ratios,
                gates=gates,
                passed=all(gates.values()),
            )
        )
s = dict(
    metrics=metrics,
    comparisons=comparisons,
    passed=all(x["passed"] for x in comparisons),
    training_updates=240,
    training_targets=983040,
    backwards=240,
    broad_goal_achieved=False,
)
(ROOT / "summary.json").write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
for name in ("result.json", "audit.json", "summary.json"):
    (ROOT / (name + ".gz")).write_bytes(gzip.compress((ROOT / name).read_bytes(), mtime=0))
print(json.dumps(s, indent=2))
