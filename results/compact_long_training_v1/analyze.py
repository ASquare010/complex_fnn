"""Check all long-run budgets and compare candidate with both controls."""

import gzip
import json
import statistics as st
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/compact_long_training_v1")
p, r, a = [read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json")]
for field in ("sources", "input_hashes", "maintained_files", "checkpoint_hashes"):
    hashes(p[field])
hashes(read(ROOT / "audit_protocol.json")["files"])
assert len(r["cases"]) == 6 and r["training_updates"] == 4800 and r["training_targets"] == 19660800
assert r["backwards"] + a["backwards"] == 4812 and a["optimizer_updates"] == 0
assert a["native_scores"] == 20 and a["training_batches"] == 4800
assert len(r["boundaries"]) == 7 and len(a["boundaries"]) == 9
assert all(b["gpu"] == dict(allocated=0, reserved=0) for b in r["boundaries"] + a["boundaries"])
metrics = []
for row in r["cases"]:
    c = row["measurement"]
    assert (
        c["status"] == "COMPLETE"
        and c["optimizer_updates"] == 800
        and c["profile_backward_passes"] == 801
    )
    assert c["weights_gradients_moments_finite"] and c["parameter_count"] == 9099648
    assert [v["step"] for v in c["records"]] == list(range(1, 801))
    assert c["peak_job_allocated_bytes"] == max(
        v["peak_allocated_bytes"] for v in c["memory_phases"]
    )
    for key, summary in c["timing"].items():
        values = [v[key] for v in c["records"][20:]]
        assert (
            summary["median"] == st.median(values)
            and summary["mean"] == st.mean(values)
            and summary["sample_variance"] == st.variance(values)
        )
    for checkpoint in [c["initial_probe"], *c["intermediate_checkpoints"], c["checkpoint"]]:
        assert sha(checkpoint["path"]) == checkpoint["sha256"]
    metrics.append(
        dict(
            dataset=c["fixture"]["dataset"],
            arm=row["arm"],
            nll=c["final_validation"]["nll"],
            peak_mib=c["peak_job_allocated_bytes"] / 2**20,
            event_ms=c["timing"]["event_update_ms"]["median"],
            wall_ms=c["timing"]["wall_update_ms"]["median"],
            host_mib=max(v["host"]["allocated_bytes.peak"] for v in c["memory_phases"]) / 2**20,
            stability=c["timing_stability_ratio"],
        )
    )
comparisons = []
for f in p["fixtures"]:
    peers = [row for row in r["cases"] if row["measurement"]["fixture"] == f]
    for key in (
        "initial_model_hash",
        "initial_optimizer_hash",
        "initial_sampler_hash",
        "data_order_hash",
        "final_sampler_hash",
    ):
        assert len({row["measurement"][key] for row in peers}) == 1
    candidate = next(c for c in metrics if c["dataset"] == f["dataset"] and c["arm"] == "combined")
    for control in ("native", "chunks"):
        base = next(c for c in metrics if c["dataset"] == f["dataset"] and c["arm"] == control)
        ratios = {
            key: candidate[key] / base[key] for key in ("nll", "peak_mib", "event_ms", "wall_ms")
        }
        gates = dict(
            audit=a["passed"],
            quality=ratios["nll"] <= p["nll_ratio_max"],
            memory=ratios["peak_mib"] <= p["memory_ratio_max"],
            event=ratios["event_ms"] <= p["time_ratio_max"],
            wall=ratios["wall_ms"] <= p["time_ratio_max"],
            host=candidate["host_mib"] * 2**20 <= p["host_peak_max"],
            stability=all(
                c["stability"] <= p["stability_max"]
                for c in metrics
                if c["dataset"] == f["dataset"]
            ),
        )
        comparisons.append(
            dict(
                dataset=f["dataset"],
                control=control,
                ratios=ratios,
                gates=gates,
                passed=all(gates.values()),
            )
        )
s = dict(
    metrics=metrics,
    comparisons=comparisons,
    passed=all(c["passed"] for c in comparisons),
    training_updates=4800,
    training_targets=19660800,
    backwards=4812,
    broad_goal_achieved=False,
)
(ROOT / "summary.json").write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
for name in ("result.json", "audit.json", "summary.json"):
    (ROOT / (name + ".gz")).write_bytes(gzip.compress((ROOT / name).read_bytes(), mtime=0))
print(json.dumps(s, indent=2))
