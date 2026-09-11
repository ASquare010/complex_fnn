"""Apply the prospective H121 gates without choosing favorable subsets."""

import csv
import gzip
import io
import json

from results.checkpoint_input_offload_v1.source.prepare import ROOT, read


def trace_gate(c, p):
    t = c["traced"]["trace"]
    rows = t["rows"]
    return (
        len(rows) == 8
        and [r["block"] for r in rows] == list(range(8))
        and sum(r["bytes"] for r in rows) == p["boundary_payload_bytes"]
        and all(
            r["shape"] == [8, 512, 384]
            and r["dtype"] == "torch.float32"
            and r["unpack_count"] == 1
            and r["source_hash"] == r["unpack_hash"]
            and r["packed_device"].startswith("cpu" if c["mode"] == "offload" else "cuda")
            and (r["pinned"] if c["mode"] == "offload" else not r["pinned"])
            for r in rows
        )
        and t["live_payload_bytes"] == 0
        and t["peak_payload_bytes"] <= p["boundary_payload_bytes"]
        and sum(e["kind"] == "release" for e in t["events"]) == 8
    )


def summarize(p, r, a):
    metrics, comparisons = [], []
    for c in r["cases"]:
        metrics.append(
            dict(
                label=c["label"],
                dataset=c["fixture"]["dataset"],
                classifier=c["fixture"]["loss_policy"],
                mode=c["mode"],
                peak_mib=c["peak_allocated_bytes"] / 2**20,
                reserved_peak_mib=c["peak_reserved_bytes"] / 2**20,
                normal_peak_mib=c["normal_peak_allocated_bytes"] / 2**20,
                pinned_allocated_peak_mib=c["host_allocated_peak"] / 2**20,
                pinned_active_peak_mib=c["host_active_peak"] / 2**20,
                event_ms=c["timing"]["event_sum_ms"]["median"],
                wall_ms=c["timing"]["wall_ms"]["median"],
                trace_passed=trace_gate(c, p),
                parameters=c["parameters"],
            )
        )
    for fixture in p["fixtures"]:
        native = next(c for c in r["cases"] if c["fixture"] == fixture and c["mode"] == "native")
        off = next(c for c in r["cases"] if c["fixture"] == fixture and c["mode"] == "offload")
        memory_ratio = off["peak_allocated_bytes"] / native["peak_allocated_bytes"]
        ratios = {
            k: off["timing"][k]["median"] / native["timing"][k]["median"]
            for k in ("wall_ms", "event_sum_ms")
        }
        relevant = [v for v in a["comparisons"] if v["label"] in (native["label"], off["label"])]
        gates = dict(
            numerical=len(relevant) == 6 and all(v["passed"] for v in relevant),
            trace=trace_gate(native, p) and trace_gate(off, p),
            memory=memory_ratio <= p["memory_ratio_max"],
            event_time=ratios["event_sum_ms"] <= p["time_ratio_max"],
            wall_time=ratios["wall_ms"] <= p["time_ratio_max"],
            timing_stability=all(
                v <= p["timing_stability_max"]
                for c in (native, off)
                for v in c["timing_stability"].values()
            ),
            host_memory=off["host_allocated_peak"] <= p["host_allocated_peak_max"],
        )
        comparisons.append(
            dict(
                fixture=fixture["label"],
                memory_ratio=memory_ratio,
                normal_memory_ratio=off["normal_peak_allocated_bytes"]
                / native["normal_peak_allocated_bytes"],
                time_ratios=ratios,
                gates=gates,
                passed=all(gates.values()),
            )
        )
    passed = all(c["passed"] for c in comparisons)
    return dict(
        metrics=metrics,
        comparisons=comparisons,
        decision="QUALIFIED FOR A SEPARATE SHORT TRAINING SCREEN"
        if passed
        else "ELIMINATED FOR THIS JOINT MEMORY/RUNTIME GATE",
        all_numerical_audits_passed=a["passed"],
        component_gate_qualified=passed,
        training_updates=0,
        backwards=262,
        diagnostic_targets=1048576,
        broad_goal_achieved=False,
        maintained_defaults_changed=False,
        original_quality_gate_requalified=False,
    )


def csv_file(name, rows):
    buffer = io.StringIO(newline="")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    writer = csv.DictWriter(buffer, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    raw = buffer.getvalue().encode()
    (ROOT / name).write_bytes(gzip.compress(raw, mtime=0))


def run():
    p, r, a = [read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json")]
    s = summarize(p, r, a)
    (ROOT / "summary.json").write_text(
        json.dumps(s, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    csv_file("metrics.csv.gz", s["metrics"])
    csv_file(
        "repetitions.csv.gz",
        [
            dict(
                label=c["label"],
                **{k: v for k, v in row.items() if k != "event_ms"},
                **{phase + "_ms": value for phase, value in row["event_ms"].items()},
            )
            for c in r["cases"]
            for row in c["repetitions"]
        ],
    )
    phases = []
    for stage, cases in (("profile", r["cases"]), ("audit", a["replays"])):
        for c in cases:
            for row in c["phases"]:
                phases.append(
                    dict(
                        stage=stage,
                        label=c["label"],
                        **{k: v for k, v in row.items() if not isinstance(v, dict)},
                        **{"storage_" + k: v for k, v in row["storage_bytes"].items()},
                        **{"host_" + k: v for k, v in row["host"].items()},
                    )
                )
    csv_file("phases.csv.gz", phases)
    for name in ("qualification.json", "result.json", "audit.json"):
        (ROOT / (name + ".gz")).write_bytes(gzip.compress((ROOT / name).read_bytes(), mtime=0))
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    run()
