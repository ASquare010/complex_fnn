"""Independent CPU audit of frozen paired timings, gradients and telemetry."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch, tensor_hash, tree_hash
from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
import csv
import datetime
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path("results/paired_workspace_timing_v1")


def stats(values):
    return dict(
        mean=st.mean(values),
        median=st.median(values),
        variance=st.variance(values) if len(values) > 1 else 0,
        minimum=min(values),
        maximum=max(values),
        n=len(values),
    )


def run():
    p, raw = read(ROOT / "protocol.json"), read(ROOT / "result.json")
    assert not torch.cuda.is_initialized()
    hashes(read(ROOT / "audit_protocol.json")["files"])
    for field in ("sources", "input_hashes", "maintained_files"):
        hashes(p[field])
    for values in p["base_verified"].values():
        hashes(values)
    assert (ROOT / "worker_exit.txt").read_text().strip() == "0"
    rows = [r for c in raw["corpora"] for r in c["rows"]]
    assert rows == [json.loads(line) for line in (ROOT / "bursts.jsonl").read_text().splitlines()]
    assert len(rows) == 32 and len(raw["references"]) == 4
    refs = {(r["dataset"], r["mode"]): r for r in raw["references"]}
    for r in refs.values():
        artifact = r["artifact"]
        assert sha(artifact["path"]) == artifact["sha256"]
        gradients = torch.load(artifact["path"], map_location="cpu", weights_only=True)
        assert all(bool(torch.isfinite(g).all()) for g in gradients.values())
        assert {k: tensor_hash(v) for k, v in gradients.items()} == r["gradients"]
        assert tree_hash(gradients) == artifact["tree_hash"]
        assert r["unchanged"]
    samples = []
    for row in csv.reader((ROOT / "telemetry.csv").read_text().splitlines()):
        assert len(row) == 10
        stamp, index, uuid, pstate, *metrics = [x.strip() for x in row]
        if int(index) != 0:
            continue
        sample = dict(
            at_ns=int(datetime.datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f").timestamp() * 1e9),
            uuid=uuid,
            pstate=pstate,
        )
        for k, value in zip(
            (
                "temperature_c",
                "sm_mhz",
                "memory_mhz",
                "power_w",
                "power_limit_w",
                "utilization_pct",
            ),
            metrics,
            strict=True,
        ):
            sample[k] = None if value in ("[N/A]", "N/A", "[Not Supported]") else float(value)
        samples.append(sample)
    assert samples and len({x["uuid"] for x in samples}) == 1
    assert read(ROOT / "telemetry_metadata.json")["monitor_stopped"]
    assert not (ROOT / "telemetry.stderr").read_text().strip()
    backwards = len(refs)
    for row in rows:
        ref = refs[row["dataset"], "high" if row["arm"] == "reuse32" else "low"]
        assert row["exact"] and row["gradients"] == ref["gradients"]
        assert row["loss"] == ref["loss"] and row["evaluation"] == ref["evaluation"]
        assert row["settings"]["workspace_bytes"] == (32 if row["arm"] == "reuse32" else 8) * 2**20
        assert len(row["probes"]) == 8
        backwards += len(row["probes"])
        measured = [x for x in row["probes"] if x["measured"]]
        assert len(measured) == 5 and [x["rep"] for x in measured] == list(range(3, 8))
        assert all(
            math.isfinite(x[k]) and x[k] > 0 for x in measured for k in ("event_ms", "wall_ms")
        )
        row["timing"] = {
            k: stats([x[k] for x in measured]) for k in ("event_ms", "wall_ms", "cpu_ms")
        }
        row["at_ns"], row["until_ns"] = measured[0]["at_ns"], measured[-1]["until_ns"]
        active = [x for x in samples if row["at_ns"] <= x["at_ns"] <= row["until_ns"]]
        row["telemetry_samples"] = len(active)
        row["telemetry"] = {
            k: stats(v) if (v := [x[k] for x in active if x[k] is not None]) else None
            for k in (
                "temperature_c",
                "sm_mhz",
                "memory_mhz",
                "power_w",
                "power_limit_w",
                "utilization_pct",
            )
        }
    assert backwards == raw["backwards"] == p["backwards"] == 260
    assert raw["updates"] == 0 and raw["evaluations"] == len(rows) + len(refs) == p["evaluations"]
    assert len(raw["boundaries"]) == p["boundaries"] == 7
    assert all(b["gpu"]["allocated"] == b["gpu"]["reserved"] == 0 for b in raw["boundaries"])
    assert all(c["unchanged"] for c in raw["corpora"])
    comparisons = []
    for f in p["fixtures"]:
        for control in p["controls"]:
            group = [r for r in rows if r["dataset"] == f["dataset"] and r["control"] == control]
            pairs = []
            for cycle, order in enumerate(p["cycles"]):
                cycle_rows = [r for r in group if r["cycle"] == cycle]
                assert "".join(r["letter"] for r in cycle_rows) == order
                assert [r["position"] for r in cycle_rows] == list(range(4))
                for i in (0, 2):
                    couple = cycle_rows[i : i + 2]
                    a, b = (
                        next(r for r in couple if r["letter"] == "A"),
                        next(r for r in couple if r["letter"] == "B"),
                    )
                    pairs.append(
                        dict(
                            cycle=cycle,
                            positions=[i, i + 1],
                            ratios={
                                k: a["timing"][k]["median"] / b["timing"][k]["median"]
                                for k in ("event_ms", "wall_ms")
                            },
                        )
                    )
            ratios = {
                k: stats([pair["ratios"][k] for pair in pairs]) for k in ("event_ms", "wall_ms")
            }
            gates = {k: v["median"] <= p["timing_ratio_limit"] for k, v in ratios.items()}
            gates["telemetry"] = all(r["telemetry_samples"] >= 1 for r in group)
            comparisons.append(
                dict(
                    dataset=f["dataset"],
                    control=control,
                    pairs=pairs,
                    ratios=ratios,
                    gates=gates,
                    passed=all(gates.values()),
                )
            )
    summary = dict(
        study="H133",
        passed=all(x["passed"] for x in comparisons),
        exact=True,
        comparisons=comparisons,
        bursts=rows,
        samples=samples,
        backwards=backwards,
        updates=0,
        evaluations=36,
        boundaries=7,
        evidence_verified=True,
    )
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    assert not torch.cuda.is_initialized()
    hashes(read(ROOT / "audit_protocol.json")["files"])
    print(json.dumps({k: summary[k] for k in ("passed", "exact", "backwards", "comparisons")}))


if __name__ == "__main__":
    run()
