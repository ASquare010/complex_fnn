"""Verify workspace accounting and apply the prospective diagnostic gates."""

import gzip
import json
import statistics as st
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/blas_workspace_v1")
p = read(ROOT / "protocol.json")
for key in ("sources", "input_hashes", "maintained_files"):
    hashes(p[key])
hashes(read(ROOT / "audit_protocol.json")["files"])
results = {mode: read(ROOT / (mode + "_result.json")) for mode in ("high", "low")}
audits = {mode: read(ROOT / ("audit_" + mode + ".json")) for mode in ("high", "low")}
metrics = []
for mode, r in results.items():
    a = audits[mode]
    assert (
        r["backwards"] == 40
        and a["backwards"] == 4
        and r["training_updates"] == a["training_updates"] == 0
    )
    assert r["settings"] == a["settings"] and len(r["cases"]) == len(a["cases"]) == 4
    assert len(r["boundaries"]) + len(a["boundaries"]) == 10
    assert all(b["gpu"] == dict(allocated=0, reserved=0) for b in r["boundaries"] + a["boundaries"])
    for row in r["cases"]:
        c = row["measurement"]
        assert (
            len(c["rows"]) == 10
            and c["backwards"] == 10
            and c["training_updates"] == 0
            and c["parameters"] == 9099648
        )
        assert c["state_unchanged"] and sha(c["gradient"]["path"]) == c["gradient"]["sha256"]
        for key, timing in c["timing"].items():
            vals = [x[key] for x in c["rows"][3:]]
            assert timing == dict(
                mean=st.mean(vals), median=st.median(vals), variance=st.variance(vals)
            )
        assert c["peak"] == max(x["peak_allocated_bytes"] for x in c["phases"])
        replay = next(
            x
            for x in a["cases"]
            if x["dataset"] == row["fixture"]["dataset"] and x["arm"] == row["arm"]
        )
        release = row["workspace_release"]
        assert release["after"] == 0 and release["before"] == release["released"]
        metrics.append(
            dict(
                workspace=mode,
                arm=row["arm"],
                dataset=c["dataset"],
                peak_mib=c["peak"] / 2**20,
                event_ms=c["timing"]["event_sum_ms"]["median"],
                wall_ms=c["timing"]["wall_ms"]["median"],
                host_mib=c["host_allocated_peak"] / 2**20,
                released_bytes=release["released"],
                queried_workspace_bytes=r["settings"]["workspace_bytes"],
                exact=replay["passed"],
            )
        )
comparisons = []
accounting = []
for dataset in ("wikitext2", "tinystories"):
    candidate = next(
        x for x in metrics if (x["workspace"], x["arm"], x["dataset"]) == ("low", "reuse", dataset)
    )
    for workspace, arm in (("low", "native"), ("high", "reuse")):
        base = next(
            x
            for x in metrics
            if (x["workspace"], x["arm"], x["dataset"]) == (workspace, arm, dataset)
        )
        ratios = {k: candidate[k] / base[k] for k in ("peak_mib", "event_ms", "wall_ms")}
        gates = dict(
            audit=all(x["passed"] for x in audits.values()),
            memory=ratios["peak_mib"] <= p["memory_ratio_max"],
            event=ratios["event_ms"] <= p["time_ratio_max"],
            wall=ratios["wall_ms"] <= p["time_ratio_max"],
            host=candidate["host_mib"] * 2**20 <= p["host_peak_max"],
        )
        comparisons.append(
            dict(
                dataset=dataset,
                control=workspace + "_" + arm,
                ratios=ratios,
                gates=gates,
                passed=all(gates.values()),
            )
        )
    for arm in ("native", "reuse"):
        high, low = [
            next(
                x for x in metrics if (x["workspace"], x["arm"], x["dataset"]) == (m, arm, dataset)
            )
            for m in ("high", "low")
        ]
        delta = high["released_bytes"] - low["released_bytes"]
        predicted = 2 * (high["queried_workspace_bytes"] - low["queried_workspace_bytes"])
        accounting.append(
            dict(
                dataset=dataset,
                arm=arm,
                release_difference_bytes=delta,
                predicted_bytes=predicted,
                exact_prediction=delta == predicted,
                peak_difference_bytes=round((high["peak_mib"] - low["peak_mib"]) * 2**20),
            )
        )
s = dict(
    metrics=metrics,
    comparisons=comparisons,
    accounting=accounting,
    accounting_passed=all(x["exact_prediction"] for x in accounting),
    passed=all(x["passed"] for x in comparisons),
    backwards=88,
    training_updates=0,
    broad_goal_achieved=False,
)
(ROOT / "summary.json").write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
for name in (
    "high_result.json",
    "low_result.json",
    "audit_high.json",
    "audit_low.json",
    "summary.json",
):
    (ROOT / (name + ".gz")).write_bytes(gzip.compress((ROOT / name).read_bytes(), mtime=0))
print(json.dumps(s, indent=2))
