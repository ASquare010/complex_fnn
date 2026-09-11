"""Apply the complete-job resource, quality and independent-audit gates."""

import gzip
import json
import statistics as st
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/compact_training_v1")
p, r, a = [read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json")]
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
hashes(read(ROOT / "audit_protocol.json")["files"])
assert (
    len(r["cases"]) == 4
    and sum(row["measurement"]["training_updates"] for row in r["cases"]) == 120
)
assert r["backwards"] == 122 and a["backwards"] == a["optimizer_updates"] == 0
assert a["native_scores"] == 4 and a["batches"] == 120
assert len(r["boundaries"]) == 7 and len(a["boundaries"]) == 5
assert all(b["gpu"] == dict(allocated=0, reserved=0) for b in r["boundaries"] + a["boundaries"])
for row in r["cases"]:
    c = row["measurement"]
    assert len(c["updates"]) == 30 and len(c["phases"]) == 157 and c["parameters"] == 9099648
    assert c["peak_allocated_bytes"] == max(m["peak_allocated_bytes"] for m in c["phases"])
    for m in c["phases"]:
        assert (
            sum(m["storage_bytes"].values()) + m["unattributed_live_bytes"]
            == m["live_allocated_bytes"]
        )
    for key in ("wall_ms", "event_sum_ms"):
        values = [v[key] for v in c["updates"][10:]]
        assert c["timing"][key]["median"] == st.median(values)
        assert c["timing"][key]["mean"] == st.mean(values)
        assert c["timing"][key]["sample_variance"] == st.variance(values)
    for key in ("first_gradients", "first_state", "final_state"):
        assert sha(c[key]["path"]) == c[key]["sha256"]
comparisons = []
for f in p["fixtures"]:
    baseline, combined = [
        next(
            row["measurement"]
            for row in r["cases"]
            if row["arm"] == arm and row["measurement"]["fixture"] == f
        )
        for arm in ("baseline", "combined")
    ]
    assert [(v["tokens_hash"], v["targets_hash"]) for v in baseline["updates"]] == [
        (v["tokens_hash"], v["targets_hash"]) for v in combined["updates"]
    ]
    for key in ("initial_model_hash", "initial_moment_hash", "initial_sampler_hash"):
        assert baseline[key] == combined[key]
    memory = combined["peak_allocated_bytes"] / baseline["peak_allocated_bytes"]
    times = {
        k: combined["timing"][k]["median"] / baseline["timing"][k]["median"]
        for k in ("wall_ms", "event_sum_ms")
    }
    nll = combined["after_score"]["nll"] / baseline["after_score"]["nll"]
    host = max(m["host"]["allocated_bytes.peak"] for m in combined["phases"])
    gates = dict(
        audit=a["passed"],
        memory=memory <= p["memory_ratio_max"],
        wall=times["wall_ms"] <= p["time_ratio_max"],
        event=times["event_sum_ms"] <= p["time_ratio_max"],
        nll=nll <= p["nll_ratio_max"],
        host=host <= p["host_peak_max"],
        stability=all(
            v <= p["timing_stability_max"]
            for c in (baseline, combined)
            for v in c["timing_stability"].values()
        ),
    )
    comparisons.append(
        dict(
            dataset=f["dataset"],
            baseline_mib=baseline["peak_allocated_bytes"] / 2**20,
            combined_mib=combined["peak_allocated_bytes"] / 2**20,
            memory_ratio=memory,
            time_ratios=times,
            baseline_nll=baseline["after_score"]["nll"],
            combined_nll=combined["after_score"]["nll"],
            nll_ratio=nll,
            host_mib=host / 2**20,
            gates=gates,
            passed=all(gates.values()),
        )
    )
s = dict(
    comparisons=comparisons,
    passed=all(c["passed"] for c in comparisons),
    training_updates=120,
    training_targets=491520,
    backwards=122,
    broad_goal_achieved=False,
)
(ROOT / "summary.json").write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
for name in ("result.json", "audit.json", "summary.json"):
    (ROOT / (name + ".gz")).write_bytes(gzip.compress((ROOT / name).read_bytes(), mtime=0))
print(json.dumps(s, indent=2))
