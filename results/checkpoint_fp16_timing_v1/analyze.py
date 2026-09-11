"""Apply the prospective all-fixture timing/resource/short-NLL gates."""

import csv
import datetime as dt
import statistics as st
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/checkpoint_fp16_timing_v1")
p = read(ROOT / "protocol.json")
a = read(ROOT / "audit.json")
assert (ROOT / "audit_exit.txt").read_text().strip() == "0"
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
hashes(read(ROOT / "audit_protocol.json")["files"])
cases = [read(ROOT / f"case{i:02d}" / "case.json") for i in range(36)]
bounds = read(ROOT / "boundaries.json") + a["boundaries"]
assert len(bounds) == 74 and all(b["gpu"] == dict(allocated=0, reserved=0) for b in bounds)
assert (
    read(ROOT / "telemetry_metadata.json")["monitor_stopped"]
    and not (ROOT / "telemetry.stderr").read_text().strip()
)
samples = []
for row in csv.reader((ROOT / "telemetry.csv").read_text().splitlines()):
    stamp, gpu, uuid, state, temp, sm, mem, power, limit, util = [v.strip() for v in row]
    if int(gpu) != 0:
        continue
    samples.append(
        dict(
            at_ns=int(dt.datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f").timestamp() * 1e9),
            pstate=state,
            sm=sm,
            temperature=temp,
            power=power,
        )
    )
metrics = []
for r, expected in zip(cases, p["schedule"], strict=True):
    for key in ("index", "arm", "repeat", "fixture"):
        assert r[key] == expected[key]
    c = r["measurement"]
    assert (
        c["finite"]
        and len(c["updates"]) == 30
        and len(c["phases"]) == 67
        and c["parameters"] == 9099648
    )
    assert c["training_targets"] == 245760 and c["training_updates"] == 30
    assert c["initial_model_hash"] == r["fixture"]["model_hash"]
    assert c["initial_sampler_hash"] == r["fixture"]["sampler_hash"]
    assert c["compressed_inputs"] == c["unpacked_inputs"] == (240 if r["arm"] == "fp16" else 0)
    assert c["wrapped_blocks"] == (
        list(range(8)) if r["arm"] == "fp16" else [4, 5, 6, 7] if r["arm"] == "buffer4" else []
    )
    s = r["settings"]
    assert (
        s["deterministic"] is False
        and s["tf32"] is False
        and s["threads"] == 4
        and s["workspace_bytes"] == 8.125 * 2**20
    )
    for v in c["phases"]:
        assert (
            sum(v["storage_bytes"].values()) + v["unattributed_live_bytes"]
            == v["live_allocated_bytes"]
        )
    assert c["peak_allocated_bytes"] == max(v["peak_allocated_bytes"] for v in c["phases"])
    warm = c["updates"][10:]
    for key in ("wall_ms", "event_complete_ms", "event_sum_ms"):
        values = [v[key] for v in warm]
        for field, fn in (
            ("mean", st.mean),
            ("median", st.median),
            ("sample_variance", st.variance),
        ):
            assert c["timing"][key][field] == fn(values)
        medians = [st.median(values[:10]), st.median(values[10:])]
        assert c["timing_stability"][key] == max(medians) / min(medians)
    active = [v for v in samples if any(t["at_ns"] <= v["at_ns"] <= t["until_ns"] for t in warm)]
    metrics.append(
        dict(
            index=r["index"],
            dataset=r["fixture"]["dataset"],
            seed=r["fixture"]["seed"],
            arm=r["arm"],
            repeat=r["repeat"],
            peak_bytes=c["peak_allocated_bytes"],
            host_bytes=max(v["host"]["allocated_bytes.peak"] for v in c["phases"]),
            timing=c["timing"],
            stability=c["timing_stability"],
            nll=c["after_score"]["nll"],
            telemetry_samples=len(active),
            pstates=sorted(set(v["pstate"] for v in active)),
        )
    )
fixtures = []
for f in p["fixtures"]:
    peers = {(r["arm"], r["repeat"]): r["measurement"] for r in cases if r["fixture"] == f}
    ms = [m for m in metrics if (m["dataset"], m["seed"]) == (f["dataset"], f["seed"])]
    for c in peers.values():
        assert [(v["tokens_hash"], v["targets_hash"]) for v in c["updates"]] == [
            (v["tokens_hash"], v["targets_hash"]) for v in peers["ordinary", 0]["updates"]
        ]
    stats = {}
    between = {}
    for arm in ("ordinary", "buffer4", "fp16"):
        stats[arm] = {}
        between[arm] = {}
        for key in ("wall_ms", "event_complete_ms"):
            vals = [v[key] for rep in (0, 1) for v in peers[arm, rep]["updates"][10:]]
            stats[arm][key] = dict(
                mean=st.mean(vals), median=st.median(vals), sample_variance=st.variance(vals)
            )
            repmed = [peers[arm, rep]["timing"][key]["median"] for rep in (0, 1)]
            between[arm][key] = max(repmed) / min(repmed)
    comparisons = []
    for other in ("ordinary", "buffer4"):
        timing = {
            key: stats["fp16"][key]["median"] / stats[other][key]["median"]
            for key in ("wall_ms", "event_complete_ms")
        }
        memory = max(peers["fp16", rep]["peak_allocated_bytes"] for rep in (0, 1)) / max(
            peers[other, rep]["peak_allocated_bytes"] for rep in (0, 1)
        )
        quality = [
            peers["fp16", rep]["after_score"]["nll"] / peers[other, rep]["after_score"]["nll"]
            for rep in (0, 1)
        ]
        repeats = [
            {
                key: peers["fp16", rep]["timing"][key]["median"]
                / peers[other, rep]["timing"][key]["median"]
                for key in timing
            }
            for rep in (0, 1)
        ]
        comparisons.append(
            dict(
                reference=other,
                timing=timing,
                memory_ratio=memory,
                nll_ratios=quality,
                repeat_timing_ratios=repeats,
                passed=all(v <= (1.15 if other == "ordinary" else 1.05) for v in timing.values())
                and memory <= (0.9 if other == "ordinary" else 1.02)
                and max(quality) <= 1.01,
            )
        )
    stability = all(
        m["stability"][k] <= 1.15 for m in ms for k in ("wall_ms", "event_complete_ms")
    ) and all(v <= 1.15 for arm in between.values() for v in arm.values())
    host = max(m["host_bytes"] for m in ms) <= 128 * 2**20
    coverage = all(m["telemetry_samples"] > 0 for m in ms)
    passed = (
        a["passed"] and all(v["passed"] for v in comparisons) and stability and host and coverage
    )
    fixtures.append(
        dict(
            dataset=f["dataset"],
            seed=f["seed"],
            comparisons=comparisons,
            statistics=stats,
            between_repeat_stability=between,
            stability_pass=stability,
            host_pass=host,
            telemetry_pass=coverage,
            passed=passed,
        )
    )
write_json(
    ROOT / "summary.json",
    dict(
        status="EVIDENCE_VERIFIED",
        passed=all(f["passed"] for f in fixtures),
        fixtures=fixtures,
        metrics=metrics,
        training_updates=1080,
        training_backwards=1080,
        training_targets=8847360,
    ),
)
print("All-fixture timing/resource gate:", all(f["passed"] for f in fixtures))
