"""Apply frozen paired timing, quality, memory and stability gates."""

import csv
import datetime as dt
import statistics as st
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/batch_scale_training_v1")
p, r, a = [read(ROOT / name) for name in ("protocol.json", "result.json", "audit.json")]
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
hashes(read(ROOT / "audit_protocol.json")["files"])
assert len(r["cases"]) == 24 and r["training_updates"] == r["backwards"] == 720
assert (
    a["backwards"] == a["training_updates"] == 0
    and a["native_scores"] == 24
    and a["batches"] == 720
)
boundaries = read(ROOT / "boundaries.json") + a["boundaries"]
assert len(boundaries) == 50 and all(b["gpu"] == dict(allocated=0, reserved=0) for b in boundaries)
assert read(ROOT / "telemetry_metadata.json")["monitor_stopped"]
assert not (ROOT / "telemetry.stderr").read_text().strip()
samples = []
for row in csv.reader((ROOT / "telemetry.csv").read_text().splitlines()):
    stamp, gpu, uuid, state, temp, sm, mem, power, limit, util = [v.strip() for v in row]
    if int(gpu) != 0:
        continue
    values = dict(
        at_ns=int(dt.datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f").timestamp() * 1e9),
        pstate=state,
    )
    for key, value in [("clock", sm), ("temperature", temp), ("power", power)]:
        values[key] = None if value in ("[N/A]", "N/A", "[Not Supported]") else float(value)
    samples.append(values)
assert p["batch_size"] == 16 and p["training_targets"] == 5898240
metrics = []
for row in r["cases"]:
    c = row["measurement"]
    assert c["training_targets"] == 30 * 8192
    assert (
        len(c["updates"]) == 30
        and len(c["phases"]) == 67
        and c["finite"]
        and c["parameters"] == 9099648
    )
    assert c["offloaded_blocks"] == ([4, 5, 6, 7] if row["arm"] == "buffer4" else [])
    settings = row["settings"]
    assert (
        settings["deterministic"] is False
        and settings["tf32"] is False
        and settings["threads"] == 4
    )
    assert settings["workspace_bytes"] == 8.125 * 2**20
    for artifact in ("first_gradients", "first_state", "final_state"):
        assert sha(c[artifact]["path"]) == c[artifact]["sha256"]
    for phase in c["phases"]:
        assert (
            sum(phase["storage_bytes"].values()) + phase["unattributed_live_bytes"]
            == phase["live_allocated_bytes"]
        )
    assert c["peak_allocated_bytes"] == max(v["peak_allocated_bytes"] for v in c["phases"])
    timed = c["updates"][10:]
    for key in ("event_complete_ms", "wall_ms", "event_sum_ms"):
        values = [v[key] for v in timed]
        for field, fn in [
            ("mean", st.mean),
            ("median", st.median),
            ("sample_variance", st.variance),
        ]:
            assert c["timing"][key][field] == fn(values)
        halves = [st.median(values[:10]), st.median(values[10:])]
        assert c["timing_stability"][key] == max(halves) / min(halves)
    active = [s for s in samples if any(v["at_ns"] <= s["at_ns"] <= v["until_ns"] for v in timed)]
    sensor = {
        k: (st.median(v) if (v := [s[k] for s in active if s[k] is not None]) else None)
        for k in ("clock", "temperature", "power")
    }
    metrics.append(
        dict(
            index=row["index"],
            dataset=row["fixture"]["dataset"],
            seed=row["fixture"]["seed"],
            arm=row["arm"],
            repeat=row["repeat"],
            peak_mib=c["peak_allocated_bytes"] / 2**20,
            host_mib=max(v["host"]["allocated_bytes.peak"] for v in c["phases"]) / 2**20,
            event_ms=c["timing"]["event_complete_ms"]["median"],
            wall_ms=c["timing"]["wall_ms"]["median"],
            stability=max(c["timing_stability"][k] for k in ("wall_ms", "event_complete_ms")),
            nll=c["after_score"]["nll"],
            setup_wall_ms=c["setup_wall_ms"],
            segment_wall_ms=row["segment_wall_ms"],
            excluded_wall_ms=row["segment_wall_ms"] - sum(v["wall_ms"] for v in c["updates"]),
            telemetry_samples=len(active),
            sensors=sensor,
            pstates=sorted({s["pstate"] for s in active}),
        )
    )
comparisons = []
for f in p["fixtures"]:
    peers = [row for row in r["cases"] if row["fixture"] == f]
    assert len(peers) == 4
    for key in ("initial_model_hash", "initial_moment_hash", "initial_sampler_hash"):
        assert len({row["measurement"][key] for row in peers}) == 1
    orders = [
        [(v["tokens_hash"], v["targets_hash"]) for v in row["measurement"]["updates"]]
        for row in peers
    ]
    assert all(order == orders[0] for order in orders)
    group = {
        arm: [
            m
            for m in metrics
            if m["dataset"] == f["dataset"] and m["seed"] == f["seed"] and m["arm"] == arm
        ]
        for arm in ("ordinary", "buffer4")
    }
    times = {
        arm: {
            key: [
                v[key]
                for row in peers
                if row["arm"] == arm
                for v in row["measurement"]["updates"][10:]
            ]
            for key in ("event_complete_ms", "wall_ms")
        }
        for arm in group
    }
    ratios = dict(
        event=st.median(times["buffer4"]["event_complete_ms"])
        / st.median(times["ordinary"]["event_complete_ms"]),
        wall=st.median(times["buffer4"]["wall_ms"]) / st.median(times["ordinary"]["wall_ms"]),
        memory=max(m["peak_mib"] for m in group["buffer4"])
        / max(m["peak_mib"] for m in group["ordinary"]),
    )
    pairs = [
        dict(
            repeat=i,
            event=group["buffer4"][i]["event_ms"] / group["ordinary"][i]["event_ms"],
            wall=group["buffer4"][i]["wall_ms"] / group["ordinary"][i]["wall_ms"],
            nll=group["buffer4"][i]["nll"] / group["ordinary"][i]["nll"],
        )
        for i in (0, 1)
    ]
    repeat_stability = {
        arm: {
            key: max(m[key] for m in group[arm]) / min(m[key] for m in group[arm])
            for key in ("event_ms", "wall_ms")
        }
        for arm in group
    }
    gates = dict(
        audit=a["passed"],
        memory=ratios["memory"] <= p["memory_ratio_max"],
        event=ratios["event"] <= p["time_ratio_max"],
        wall=ratios["wall"] <= p["time_ratio_max"],
        quality=all(pair["nll"] <= p["nll_ratio_max"] for pair in pairs),
        host=all(m["host_mib"] * 2**20 <= p["host_peak_max"] for m in group["buffer4"]),
        within_segment_stability=all(
            m["stability"] <= p["stability_max"] for g in group.values() for m in g
        ),
        repeat_stability=all(
            v <= p["stability_max"] for g in repeat_stability.values() for v in g.values()
        ),
        telemetry=all(m["telemetry_samples"] > 0 for g in group.values() for m in g),
    )
    statistics = {
        arm: {
            key: dict(mean=st.mean(v), median=st.median(v), sample_variance=st.variance(v))
            for key, v in times[arm].items()
        }
        for arm in group
    }
    comparisons.append(
        dict(
            dataset=f["dataset"],
            seed=f["seed"],
            ratios=ratios,
            pairs=pairs,
            repeat_stability=repeat_stability,
            statistics=statistics,
            gates=gates,
            passed=all(gates.values()),
        )
    )
aggregates = []
for dataset in ("wikitext2", "tinystories"):
    rows = [c for c in comparisons if c["dataset"] == dataset]
    assert sorted(c["seed"] for c in rows) == [101, 113, 127]
    aggregates.append(
        dict(
            dataset=dataset,
            statistics={
                k: dict(mean=st.mean(v), median=st.median(v), sample_variance=st.variance(v))
                for k in ("event", "wall", "memory")
                if (v := [c["ratios"][k] for c in rows])
            },
        )
    )
s = dict(
    study="H142",
    passed=all(c["passed"] for c in comparisons),
    audit_passed=a["passed"],
    metrics=metrics,
    comparisons=comparisons,
    aggregates=aggregates,
    training_updates=720,
    backwards=720,
    broad_goal_achieved=False,
)
write_json(ROOT / "summary.json", s)
print("Paired timing gate:", s["passed"])
for c in comparisons:
    print(c["dataset"], c["seed"], c["ratios"], [k for k, v in c["gates"].items() if not v])
