"""Verify raw long-run summaries and apply every per-seed gate."""

import csv
import datetime
import statistics as st
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/batch_scale_long_v1")
p, r, a = [read(ROOT / name) for name in ("protocol.json", "result.json", "audit.json")]
for field in ("sources", "input_hashes", "maintained_files", "checkpoint_hashes"):
    hashes(p[field])
hashes(read(ROOT / "audit_protocol.json")["files"])
assert len(r["cases"]) == 12 and r["training_updates"] == 9600 and r["backwards"] == 9612
assert a["backwards"] == 12 and a["optimizer_updates"] == 0 and a["native_scores"] == 42
assert len(a["initials"]) == 6 and len(a["cases"]) == 12 and a["training_batches"] == 9600
boundaries = a["boundaries"] + [b for c in r["cases"] for b in c["boundaries"]]
assert len(boundaries) == 43 and all(b["gpu"] == dict(allocated=0, reserved=0) for b in boundaries)
metrics = []
for row in r["cases"]:
    c = row["measurement"]
    assert len(c["records"]) == 800 and len(c["memory_phases"]) == 1611
    assert c["optimizer_updates"] == 800 and c["profile_backward_passes"] == 801
    assert c["fixture"]["source_step"] == 0 and c["parameter_count"] == 9099648
    assert c["offloaded_blocks"] == ([4, 5, 6, 7] if row["arm"] == "buffer4" else [])
    assert c["weights_gradients_moments_finite"]
    assert (
        row["settings"]["deterministic"] is False
        and row["settings"]["workspace_bytes"] == 8.125 * 2**20
    )
    assert row["settings"]["threads"] == 4 and row["settings"]["tf32"] is False
    assert c["peak_job_allocated_bytes"] == max(
        x["peak_allocated_bytes"] for x in c["memory_phases"]
    )
    folder = ROOT / f"case{row['index']:02d}"
    import json

    history = [
        json.loads(line)
        for line in (Path(c["checkpoint"]["path"]).parent / "history.jsonl")
        .read_text()
        .splitlines()
    ]
    assert history == c["records"]
    for artifact in [c["initial_probe"], *c["intermediate_checkpoints"], c["checkpoint"]]:
        assert sha(artifact["path"]) == artifact["sha256"]
    timed = c["records"][20:]
    for key in c["timing"]:
        values = [v[key] for v in timed]
        for field, fn in (
            ("median", st.median),
            ("mean", st.mean),
            ("sample_variance", st.variance),
        ):
            assert c["timing"][key][field] == fn(values)
    blocks = [st.mean(v["wall_update_ms"] for v in timed[j : j + 260]) for j in (0, 260, 520)]
    assert c["timing_block_means"] == blocks and c["timing_stability_ratio"] == max(blocks) / min(
        blocks
    )
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
    active = [v for v in samples if timed[0]["at_ns"] <= v["at_ns"] <= timed[-1]["until_ns"]]
    assert read(folder / "telemetry_metadata.json")["monitor_stopped"]
    assert not (folder / "telemetry.stderr").read_text().strip()
    telemetry = {}
    for key in ("sm_mhz", "temperature_c", "power_w"):
        values = [v[key] for v in active if v[key] is not None]
        telemetry[key] = (
            dict(min=min(values), median=st.median(values), max=max(values)) if values else None
        )
    metrics.append(
        dict(
            index=row["index"],
            dataset=row["fixture"]["dataset"],
            seed=row["fixture"]["seed"],
            arm=row["arm"],
            peak_mib=c["peak_job_allocated_bytes"] / 2**20,
            reserved_mib=c["peak_job_reserved_bytes"] / 2**20,
            host_mib=max(v["host"]["allocated_bytes.peak"] for v in c["memory_phases"]) / 2**20,
            event_ms=c["timing"]["event_update_ms"]["median"],
            wall_ms=c["timing"]["wall_update_ms"]["median"],
            nll=c["final_validation"]["nll"],
            stability=c["timing_stability_ratio"],
            telemetry_samples=len(active),
            telemetry=telemetry,
        )
    )
comparisons = []
for f in p["fixtures"]:
    peers = [row["measurement"] for row in r["cases"] if row["fixture"] == f]
    for key in (
        "initial_model_hash",
        "initial_optimizer_hash",
        "initial_sampler_hash",
        "data_order_hash",
        "final_sampler_hash",
    ):
        assert len({v[key] for v in peers}) == 1
    orders = [[(v["tokens_hash"], v["targets_hash"]) for v in c["records"]] for c in peers]
    assert all(order == orders[0] for order in orders)
    group = {
        m["arm"]: m for m in metrics if m["dataset"] == f["dataset"] and m["seed"] == f["seed"]
    }
    candidate = group["buffer4"]
    for arm in ("ordinary",):
        control = group[arm]
        ratios = {k: candidate[k] / control[k] for k in ("peak_mib", "event_ms", "wall_ms", "nll")}
        gates = dict(
            audit=a["passed"],
            memory=ratios["peak_mib"] <= p["memory_ratio_max"],
            event=ratios["event_ms"] <= p["time_ratio_max"],
            wall=ratios["wall_ms"] <= p["time_ratio_max"],
            quality=ratios["nll"] <= p["nll_ratio_max"],
            host=candidate["host_mib"] * 2**20 <= p["host_peak_max"],
            stability=all(v["stability"] <= p["stability_max"] for v in (candidate, control)),
            telemetry=all(v["telemetry_samples"] > 0 for v in (candidate, control)),
        )
        comparisons.append(
            dict(
                dataset=f["dataset"],
                seed=f["seed"],
                control=arm,
                ratios=ratios,
                gates=gates,
                passed=all(gates.values()),
            )
        )
aggregates = []
for dataset in ("wikitext2", "tinystories"):
    for arm in ("ordinary", "buffer4"):
        rows = [m for m in metrics if m["dataset"] == dataset and m["arm"] == arm]
        assert sorted(m["seed"] for m in rows) == [101, 113, 127]
        stats = {
            key: dict(
                mean=st.mean(v),
                median=st.median(v),
                variance=st.variance(v),
                min=min(v),
                max=max(v),
            )
            for key in ("nll", "peak_mib", "event_ms", "wall_ms")
            if (v := [m[key] for m in rows])
        }
        aggregates.append(dict(dataset=dataset, arm=arm, stats=stats))
s = dict(
    study="H156",
    passed=all(c["passed"] for c in comparisons),
    audit_passed=a["passed"],
    metrics=metrics,
    comparisons=comparisons,
    aggregates=aggregates,
    training_updates=9600,
    training_targets=78643200,
    backwards=9624,
    broad_goal_achieved=False,
)
write_json(ROOT / "summary.json", s)
print("Long-training gate:", s["passed"])
for c in comparisons:
    print(
        c["dataset"],
        c["seed"],
        c["control"],
        c["ratios"],
        [k for k, v in c["gates"].items() if not v],
    )
