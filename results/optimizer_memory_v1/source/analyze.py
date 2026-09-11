"""Apply the fixed all-fixture gate and preserve phase-level resource evidence."""

import csv
import gzip
import io
import json

from results.optimizer_memory_v1.source.prepare import ROOT, read


def csv_gz(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(gzip.compress(stream.getvalue().encode(), mtime=0))


def run():
    p, r, audit = [read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json")]
    assert (ROOT / "audit_exit.txt").read_text().strip() == "0"
    metrics, phases, updates, comparisons = [], [], [], []
    passed = {
        c["label"]: all(
            next(x for x in audit[k] if x["label"] == c["label"])["passed"]
            for k in ("numerical", "gradient_comparisons", "scores")
        )
        for c in r["cases"]
    }
    for c in r["cases"]:
        fixture = c["fixture"]
        peak = max(c["phases"], key=lambda m: m["peak_allocated_bytes"])
        metrics.append(
            dict(
                label=c["label"],
                dataset=fixture["dataset"],
                loss_policy=fixture["loss_policy"],
                optimizer=c["mode"],
                seed=101,
                total_parameters=c["parameters"],
                before_nll=c["before_score"]["nll"],
                final_nll=c["after_score"]["nll"],
                peak_mib=c["peak_allocated_bytes"] / 2**20,
                reserved_mib=c["peak_reserved_bytes"] / 2**20,
                peak_phase=peak["phase"],
                event_median_ms=c["timing"]["event_sum_ms"]["median"],
                wall_median_ms=c["timing"]["wall_ms"]["median"],
                numerical_passed=passed[c["label"]],
                training_updates=30,
                training_targets=122880,
            )
        )
        for m in c["phases"]:
            phases.append(
                dict(
                    label=c["label"],
                    phase=m["phase"],
                    peak_allocated_bytes=m["peak_allocated_bytes"],
                    peak_reserved_bytes=m["peak_reserved_bytes"],
                    live_allocated_bytes=m["live_allocated_bytes"],
                    live_reserved_bytes=m["live_reserved_bytes"],
                    **m["storage_bytes"],
                    known_storage_bytes=m["known_storage_bytes"],
                    unattributed_live_bytes=m["unattributed_live_bytes"],
                )
            )
        for u in c["updates"]:
            updates.append(
                dict(
                    label=c["label"],
                    **{k: v for k, v in u.items() if k != "event_ms"},
                    **{k + "_ms": v for k, v in u["event_ms"].items()},
                )
            )
    for fixture in p["fixtures"]:
        peers = [c for c in r["cases"] if c["fixture"] == fixture]
        base = next(c for c in peers if c["mode"] == "default")
        for c in peers:
            if c["mode"] == "default":
                continue
            memory = c["peak_allocated_bytes"] / base["peak_allocated_bytes"]
            times = {
                k: c["timing"][k]["median"] / base["timing"][k]["median"]
                for k in ("wall_ms", "event_sum_ms")
            }
            nll = c["after_score"]["nll"] / base["after_score"]["nll"]
            stable = all(
                v <= p["timing_stability_max"]
                for peer in (c, base)
                for v in peer["timing_stability"].values()
            )
            gates = dict(
                numerical=passed[c["label"]] and passed[base["label"]],
                memory=memory <= p["memory_ratio_max"],
                quality=nll <= p["nll_ratio_max"],
                wall_time=times["wall_ms"] <= p["time_ratio_max"],
                event_time=times["event_sum_ms"] <= p["time_ratio_max"],
                timing_stability=stable,
            )
            optimizer_ratio = max(
                m["peak_allocated_bytes"]
                for m in c["phases"]
                if m["phase"].startswith("optimizer_")
            ) / max(
                m["peak_allocated_bytes"]
                for m in base["phases"]
                if m["phase"].startswith("optimizer_")
            )
            comparisons.append(
                dict(
                    fixture=fixture["label"],
                    mode=c["mode"],
                    memory_ratio=memory,
                    optimizer_phase_ratio=optimizer_ratio,
                    nll_ratio=nll,
                    time_ratios=times,
                    gates=gates,
                    passed=all(gates.values()),
                )
            )
    decisions = {
        mode: "EARNS UNINSTRUMENTED SCREEN"
        if all(c["passed"] for c in comparisons if c["mode"] == mode)
        else "ELIMINATED FOR THIS MEMORY GATE"
        if any(not c["gates"]["memory"] for c in comparisons if c["mode"] == mode)
        else "DOES NOT QUALIFY"
        for mode in ("single", "fused")
    }
    summary = dict(
        status="COMPLETE",
        decisions=decisions,
        all_audits_passed=audit["passed"],
        comparisons=comparisons,
        training_updates=360,
        training_targets=1474560,
        backwards=360,
        native_audit_scores=12,
        study_scores=24,
        tensor_artifacts=36,
        recorded_memory_intervals=len(phases),
        study_zero_boundaries=len(r["boundaries"]),
        audit_zero_boundaries=len(audit["boundaries"]),
        wall_seconds=r["wall_seconds"],
        original_quality_gate_requalified=False,
        maintained_defaults_changed=False,
        broad_goal_achieved=False,
    )
    (ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    for name, rows in (("metrics", metrics), ("phases", phases), ("updates", updates)):
        csv_gz(ROOT / (name + ".csv.gz"), rows)
    for name in ("result.json", "audit.json"):
        (ROOT / (name + ".gz")).write_bytes(gzip.compress((ROOT / name).read_bytes(), mtime=0))
    for path in ROOT.glob("*.log"):
        if path.name != "analyze.log":
            path.with_suffix(".log.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    print(
        json.dumps(
            dict(decisions=decisions, all_audits_passed=audit["passed"], intervals=len(phases))
        )
    )


if __name__ == "__main__":
    run()
