"""Apply prospective whole-training gates to every mirrored repeat."""

import csv
import datetime
import json
import statistics as st
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/paired_complete_training_v1")
p, r, a = [read(ROOT / name) for name in ("protocol.json", "result.json", "audit.json")]
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
hashes(read(ROOT / "audit_protocol.json")["files"])
assert len(r["cases"]) == 12 and r["backwards"] == r["training_updates"] == 360
assert len(a["boundaries"]) == 13 and a["native_scores"] == 12 and a["batches"] == 360
assert a["training_updates"] == a["backwards"] == 0
boundaries = a["boundaries"] + [b for row in r["cases"] for b in row["boundaries"]]
assert len(boundaries) == 37 and all(b["gpu"] == dict(allocated=0, reserved=0) for b in boundaries)
assert read(ROOT / "derivation_audit.json")["passed"]
metrics = []
for row in r["cases"]:
    c = row["measurement"]
    assert (
        len(c["updates"]) == 30
        and len(c["phases"]) == 67
        and c["parameters"] == 9099648
        and c["finite"]
    )
    assert row["settings"]["deterministic"] == (row["arm"] != "ordinary")
    assert row["settings"]["workspace_bytes"] == (8.125 if row["arm"] == "ordinary" else 8) * 2**20
    assert row["settings"]["threads"] == 4 and row["settings"]["tf32"] is False
    for phase in c["phases"]:
        assert (
            sum(phase["storage_bytes"].values()) + phase["unattributed_live_bytes"]
            == phase["live_allocated_bytes"]
        )
    assert c["peak_allocated_bytes"] == max(x["peak_allocated_bytes"] for x in c["phases"])
    for key in ("first_gradients", "first_state", "final_state"):
        assert sha(c[key]["path"]) == c[key]["sha256"]
    for key in ("wall_ms", "event_sum_ms", "event_complete_ms"):
        values = [v[key] for v in c["updates"][10:]]
        for field, fn in (
            ("mean", st.mean),
            ("median", st.median),
            ("sample_variance", st.variance),
        ):
            assert c["timing"][key][field] == fn(values)
        halves = [st.median(values[:10]), st.median(values[10:])]
        assert c["timing_stability"][key] == max(halves) / min(halves)
    folder = ROOT / f"case{row['index']:02d}"
    samples = []
    for values in csv.reader((folder / "telemetry.csv").read_text().splitlines()):
        assert len(values) == 10
        stamp, index, uuid, pstate, *numbers = [v.strip() for v in values]
        if int(index) != 0:
            continue
        item = dict(
            at_ns=int(datetime.datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f").timestamp() * 1e9),
            uuid=uuid,
            pstate=pstate,
        )
        for name, value in zip(
            (
                "temperature_c",
                "sm_mhz",
                "memory_mhz",
                "power_w",
                "power_limit_w",
                "utilization_pct",
            ),
            numbers,
            strict=True,
        ):
            item[name] = None if value in ("[N/A]", "N/A", "[Not Supported]") else float(value)
        samples.append(item)
    active = [
        s
        for s in samples
        if c["updates"][10]["at_ns"] <= s["at_ns"] <= c["updates"][-1]["until_ns"]
    ]
    assert read(folder / "telemetry_metadata.json")["monitor_stopped"]
    assert not (folder / "telemetry.stderr").read_text().strip()
    sensor = {}
    for key in ("sm_mhz", "power_w", "temperature_c"):
        values = [s[key] for s in active if s[key] is not None]
        sensor[key] = (
            dict(min=min(values), median=st.median(values), max=max(values)) if values else None
        )
    metrics.append(
        dict(
            index=row["index"],
            dataset=c["fixture"]["dataset"],
            arm=row["arm"],
            repeat=row["repeat"],
            peak_mib=c["peak_allocated_bytes"] / 2**20,
            reserved_mib=c["peak_reserved_bytes"] / 2**20,
            event_ms=c["timing"]["event_complete_ms"]["median"],
            wall_ms=c["timing"]["wall_ms"]["median"],
            nll=c["after_score"]["nll"],
            host_mib=max(v["host"]["allocated_bytes.peak"] for v in c["phases"]) / 2**20,
            stability=max(c["timing_stability"][k] for k in ("wall_ms", "event_complete_ms")),
            telemetry_samples=len(active),
            telemetry=sensor,
        )
    )
comparisons = []
for f in p["fixtures"]:
    peers = [row["measurement"] for row in r["cases"] if row["fixture"] == f]
    for key in ("initial_model_hash", "initial_moment_hash", "initial_sampler_hash"):
        assert len({v[key] for v in peers}) == 1
    orders = [[(v["tokens_hash"], v["targets_hash"]) for v in c["updates"]] for c in peers]
    assert all(order == orders[0] for order in orders)
    for repeat in (0, 1):
        group = {
            v["arm"]: v for v in metrics if v["dataset"] == f["dataset"] and v["repeat"] == repeat
        }
        candidate = group["reuse"]
        for arm in ("ordinary", "native"):
            control = group[arm]
            ratios = {
                k: candidate[k] / control[k] for k in ("peak_mib", "event_ms", "wall_ms", "nll")
            }
            gates = dict(
                audit=a["passed"],
                memory=ratios["peak_mib"] <= p["memory_ratio_max"],
                event=ratios["event_ms"] <= p["time_ratio_max"],
                wall=ratios["wall_ms"] <= p["time_ratio_max"],
                quality=ratios["nll"] <= p["nll_ratio_max"],
                host=candidate["host_mib"] * 2**20 <= p["host_peak_max"],
                stability=all(
                    v["stability"] <= p["timing_stability_max"] for v in (candidate, control)
                ),
                telemetry=all(v["telemetry_samples"] > 0 for v in (candidate, control)),
            )
            comparisons.append(
                dict(
                    dataset=f["dataset"],
                    repeat=repeat,
                    control=arm,
                    ratios=ratios,
                    gates=gates,
                    passed=all(gates.values()),
                )
            )
s = dict(
    study="H134",
    metrics=metrics,
    comparisons=comparisons,
    passed=all(v["passed"] for v in comparisons),
    audit_passed=a["passed"],
    training_updates=360,
    backwards=360,
    training_targets=1474560,
    broad_goal_achieved=False,
)
(ROOT / "summary.json").write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
print(json.dumps(s))
