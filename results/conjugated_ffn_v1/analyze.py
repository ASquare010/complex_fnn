"""Retain every fixed recipe and apply the prospective falsification gates."""

import statistics as st
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/conjugated_ffn_v1")
p, r, a = [read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json")]
hashes(p["sources"])
hashes(p["maintained_files"])
hashes(read(ROOT / "audit_protocol.json")["files"])
assert a["passed"] and a["scores"] == 108 and a["regenerated_datasets"] == 9
assert r["training_updates"] == 16200 and len(r["fits"]) == 54
assert len(read(ROOT / "boundaries.json")) == 55 and len(a["boundaries"]) == 55
assert all(
    b["gpu"] == dict(allocated=0, reserved=0)
    for b in read(ROOT / "boundaries.json") + a["boundaries"]
)
checks = []
for task in p["tasks"]:
    for seed in p["seeds"]:
        group = {c["arm"]: c for c in r["fits"] if (c["task"], c["seed"]) == (task, seed)}
        assert set(group) == set(p["arms"])
        c = group["shared_permuted"]
        wide = [group[k] for k in ("full_gelu", "full_swiglu")]
        for v in group.values():
            assert (
                v["parameters"] == p["counts"][v["arm"]]["parameters"]
                and v["macs"] == p["counts"][v["arm"]]["macs"]
            )
            assert v["all_finite"] and sha(v["state_path"]) == v["state_sha256"]
            assert v["median_update_ms"] == st.median(t["wall_ms"] for t in v["history"][20:])
        gates = dict(
            positive_control=all(v["reporting"] <= 0.8 * v["zero_reporting"] for v in wide),
            parameters=all(c["parameters"] <= 0.6 * v["parameters"] for v in wide),
            wide_quality=all(c["reporting"] <= 1.05 * v["reporting"] for v in wide),
            narrow_quality=c["reporting"] < group["narrow_gelu"]["reporting"],
            sharing_ablation=c["reporting"] < group["shared_same"]["reporting"],
            memory=c["peak_bytes"] <= 0.9 * group["full_gelu"]["peak_bytes"],
            runtime=c["median_update_ms"] <= 1.15 * group["full_gelu"]["median_update_ms"],
        )
        checks.append(
            dict(
                task=task,
                seed=seed,
                gates=gates,
                passed=all(gates.values()),
                candidate_mse=c["reporting"],
                wide_gelu_ratio=c["reporting"] / group["full_gelu"]["reporting"],
                narrow_ratio=c["reporting"] / group["narrow_gelu"]["reporting"],
                memory_ratio=c["peak_bytes"] / group["full_gelu"]["peak_bytes"],
                time_ratio=c["median_update_ms"] / group["full_gelu"]["median_update_ms"],
            )
        )
aggregates = []
for task in p["tasks"]:
    for arm in p["arms"]:
        rows = [c for c in r["fits"] if c["task"] == task and c["arm"] == arm]
        vals = [c["reporting"] for c in rows]
        aggregates.append(
            dict(
                task=task,
                arm=arm,
                mean=st.mean(vals),
                median=st.median(vals),
                sample_variance=st.variance(vals),
                mean_peak_mib=st.mean(c["peak_bytes"] / 2**20 for c in rows),
                mean_update_ms=st.mean(c["median_update_ms"] for c in rows),
                ranks=[c["output_rank"] for c in rows],
            )
        )
s = dict(
    study="H143",
    passed=all(c["passed"] for c in checks),
    checks=checks,
    aggregates=aggregates,
    training_updates=16200,
    broad_goal_achieved=False,
)
write_json(ROOT / "summary.json", s)
print("Structural screen:", s["passed"])
for c in checks:
    print(
        c["task"],
        c["seed"],
        round(c["wide_gelu_ratio"], 4),
        round(c["narrow_ratio"], 4),
        round(c["memory_ratio"], 4),
        round(c["time_ratio"], 4),
        [k for k, v in c["gates"].items() if not v],
    )
