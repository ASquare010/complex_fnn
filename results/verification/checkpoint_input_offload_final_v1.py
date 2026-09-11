"""Verify H121 evidence, budgets and the failed fixed four-fixture gate."""

import csv
import gzip
import io
import json
import math
import statistics as st
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import ROOT, hashes, read, sha


def table(name):
    return list(csv.DictReader(io.StringIO(gzip.decompress((ROOT / name).read_bytes()).decode())))


def equal_table(expected, actual):
    assert len(expected) == len(actual)
    for e, a in zip(expected, actual, strict=True):
        assert e.keys() == a.keys()
        for key, value in e.items():
            if isinstance(value, bool):
                assert a[key] == str(value)
            elif isinstance(value, (int, float)):
                assert float(a[key]) == value
            else:
                assert a[key] == value


def verify_trace(c):
    t = c["traced"]["trace"]
    assert t["offloaded"] == (c["mode"] == "offload")
    assert len(t["rows"]) == 8 and len(t["events"]) == 24
    assert [v["block"] for v in t["rows"]] == list(range(8))
    for row in t["rows"]:
        assert row["shape"] == [8, 512, 384] and row["dtype"] == "torch.float32"
        assert row["bytes"] == 6 * 2**20 and row["source_device"].startswith("cuda")
        assert row["source_hash"] == row["unpack_hash"] and row["unpack_count"] == 1
        assert row["pinned"] == t["offloaded"]
        assert row["packed_device"].startswith("cpu" if t["offloaded"] else "cuda")
        assert all(
            math.isfinite(row[k]) and row[k] >= 0
            for k in (
                "pack_event_ms",
                "unpack_event_ms",
                "pack_host_wall_ms",
                "unpack_host_wall_ms",
            )
        )
    events = t["events"]
    assert [e["at_ns"] for e in events] == sorted(e["at_ns"] for e in events)
    live, peak, packed, unpacked, released = 0, 0, set(), set(), set()
    for e in events:
        index, kind = e["index"], e["kind"]
        if kind == "pack":
            assert index not in packed
            packed.add(index)
            live += 6 * 2**20
        elif kind == "unpack":
            assert index in packed and index not in unpacked
            unpacked.add(index)
        else:
            assert kind == "release" and index in unpacked and index not in released
            released.add(index)
            live -= 6 * 2**20
        assert e["live_bytes"] == live and live >= 0
        peak = max(peak, live)
    assert packed == unpacked == released == set(range(8))
    assert live == t["live_payload_bytes"] == 0
    assert peak == t["peak_payload_bytes"] == 48 * 2**20
    assert t["audit_extra_d2h_bytes"] == 96 * 2**20


def run():
    p, r, a, q, s = [
        read(ROOT / n)
        for n in (
            "protocol.json",
            "result.json",
            "audit.json",
            "qualification.json",
            "summary.json",
        )
    ]
    assert len(p["sources"]) == 162 and len(p["maintained_files"]) == 61
    for field in ("sources", "maintained_files", "input_hashes", "library_hashes"):
        hashes(p[field])
    hashes(read(ROOT / "audit_protocol.json")["files"])
    recovery = read(ROOT / "recovery_protocol.json")
    hashes(recovery["files"])
    assert recovery["completed_backwards"] == 254 and recovery["remaining_backwards"] == 8
    assert recovery["completed_replays"] == recovery["repeated_completed_cases"] == 0
    prior = read(p["previous_receipt"])
    assert prior["status"] == "PASS" and prior["all_numerical_audits_passed"]
    for path, digest in prior["files"].items():
        target = p["before_documents"].get(path, {}).get("preserved_path", path)
        assert sha(target) == digest
    for field in ("tensor_hashes", "local_metric_hashes"):
        hashes(prior[field])
    for path, info in prior["packed"].items():
        assert sha(path) == info["sha256"]
    for info in p["before_documents"].values():
        assert sha(info["preserved_path"]) == info["sha256"]
    assert q["passed"] and q["backwards"] == 6 and q["finite_difference_forwards"] == 12
    assert len(q["rows"]) == 2
    for row in q["rows"]:
        assert row["passed"] and len(row["comparisons"]) == len(row["directions"]) == 3
        assert all(v["passed"] for v in row["comparisons"])
        for d in row["directions"]:
            assert d["absolute_error"] == abs(d["analytical"] - d["numerical"])
            assert d["absolute_error"] <= 1e-7 * max(1, abs(d["analytical"]))
    assert len(r["cases"]) == len(a["replays"]) == 8
    assert len(q["boundaries"]) == 3 and len(r["boundaries"]) == len(a["boundaries"]) == 9
    for b in q["boundaries"] + r["boundaries"] + a["boundaries"]:
        assert b["gpu"] == dict(allocated=0, reserved=0)
        assert b["host"]["allocated_bytes.current"] >= b["host"]["active_bytes.current"]
    assert r["backwards"] == 248 and a["backwards"] == 8
    assert (
        q["backwards"] + r["backwards"] + a["backwards"]
        == p["total_backwards"]
        == s["backwards"]
        == 262
    )
    assert r["diagnostic_targets"] + a["diagnostic_targets"] == s["diagnostic_targets"] == 1048576
    assert (
        p["training_updates"]
        == r["training_updates"]
        == a["training_updates"]
        == q["optimizer_updates"]
        == 0
    )
    assert p["validation_scores"] == a["validation_scores"] == 0
    assert a["passed"] and len(a["comparisons"]) == 24
    for v in a["comparisons"]:
        assert v["passed"] == (
            v["global_relative"] <= p["gradient_global_tolerance"]
            and v["max_tensor_relative"] <= p["gradient_tensor_tolerance"]
            and v["loss_relative"] <= p["loss_tolerance"]
        )
    tensors, local, metrics, repetitions, phases = {}, {}, {}, [], []
    h120 = read("results/optimizer_memory_v1/result.json")
    for c in r["cases"]:
        assert c["backwards"] == 31 and c["training_updates"] == 0
        assert c["diagnostic_targets"] == 31 * 4096 and c["parameters"] == 9099648
        assert c["state_unchanged"] and c["finite"]
        assert len(c["repetitions"]) == 30
        assert [v["repetition"] for v in c["repetitions"]] == list(range(1, 31))
        source = next(
            v
            for v in h120["cases"]
            if v["fixture"]["label"] == c["fixture"]["label"] and v["mode"] == "default"
        )
        assert c["provenance"]["model_hash"] == source["initial_model_hash"]
        assert c["provenance"]["moments_hash"] == source["initial_moment_hash"]
        assert c["provenance"]["sampler_before"] == source["initial_sampler_hash"]
        for key in ("tokens_hash", "targets_hash"):
            assert c["provenance"][key] == source["updates"][0][key]
        for v in c["repetitions"]:
            assert math.isfinite(v["loss"])
            assert abs(v["loss"] / c["repetitions"][-1]["loss"] - 1) <= p["loss_tolerance"]
            assert sum(v["event_ms"].values()) == v["event_sum_ms"]
            repetitions.append(
                dict(
                    label=c["label"],
                    **{k: x for k, x in v.items() if k != "event_ms"},
                    **{k + "_ms": x for k, x in v["event_ms"].items()},
                )
            )
        for k in ("wall_ms", "event_sum_ms"):
            values = [v[k] for v in c["repetitions"][10:]]
            assert c["timing"][k] == dict(
                n=20,
                mean=st.mean(values),
                median=st.median(values),
                sample_variance=st.variance(values),
                min=min(values),
                max=max(values),
            )
            half = [st.median(values[:10]), st.median(values[10:])]
            assert c["timing_stability"][k] == max(half) / min(half)
        verify_trace(c)
        assert len(c["phases"]) == 98
        assert c["peak_allocated_bytes"] == max(v["peak_allocated_bytes"] for v in c["phases"])
        assert c["peak_reserved_bytes"] == max(v["peak_reserved_bytes"] for v in c["phases"])
        assert c["normal_peak_allocated_bytes"] == max(
            v["peak_allocated_bytes"] for v in c["phases"] if v["phase"].split("_")[-1].isdigit()
        )
        assert c["normal_peak_allocated_bytes"] == c["peak_allocated_bytes"]
        assert c["host_allocated_peak"] == max(
            v["host"]["allocated_bytes.peak"] for v in c["phases"]
        )
        assert c["host_active_peak"] == max(v["host"]["active_bytes.peak"] for v in c["phases"])
        rep = next(v for v in a["replays"] if v["label"] == c["label"])
        assert rep["state_unchanged"] and rep["backwards"] == 1 and rep["training_updates"] == 0
        assert len(rep["phases"]) == 5
        for info in (c["untraced"], c["traced"]["gradient"], rep["gradient"]):
            assert info["finite"]
            tensors[info["path"]] = info["sha256"]
        for name, value in (("result.json", c), ("replay.json", rep)):
            path = ROOT / "runs" / c["label"] / name
            assert read(path) == value
            local[path.as_posix()] = sha(path)
        metrics[c["label"]] = dict(
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
            trace_passed=True,
            parameters=9099648,
        )
    for stage, cases in (("profile", r["cases"]), ("audit", a["replays"])):
        for c in cases:
            for row in c["phases"]:
                assert row["storage_bytes"]["parameters"] == 36398592
                assert (
                    row["storage_bytes"]["moments"] == 72797184
                    and row["storage_bytes"]["counters"] == 0
                )
                assert sum(row["storage_bytes"].values()) == row["known_storage_bytes"]
                assert (
                    row["known_storage_bytes"] + row["unattributed_live_bytes"]
                    == row["live_allocated_bytes"]
                )
                assert row["unattributed_live_bytes"] >= 0
                assert row["peak_allocated_bytes"] >= row["live_allocated_bytes"]
                assert (
                    row["peak_reserved_bytes"]
                    >= row["live_reserved_bytes"]
                    >= row["live_allocated_bytes"]
                )
                for family in ("allocated_bytes", "active_bytes"):
                    assert row["host"][family + ".peak"] >= row["host"][family + ".current"]
                phases.append(
                    dict(
                        stage=stage,
                        label=c["label"],
                        **{k: v for k, v in row.items() if not isinstance(v, dict)},
                        **{"storage_" + k: v for k, v in row["storage_bytes"].items()},
                        **{"host_" + k: v for k, v in row["host"].items()},
                    )
                )
    assert len(tensors) == 24 and len(phases) == 824
    hashes(tensors)
    assert s["metrics"] == list(metrics.values())
    equal_table(list(metrics.values()), table("metrics.csv.gz"))
    equal_table(repetitions, table("repetitions.csv.gz"))
    equal_table(phases, table("phases.csv.gz"))
    for f in p["fixtures"]:
        cases = [c for c in r["cases"] if c["fixture"] == f]
        assert [c["mode"] for c in cases] == f["mode_order"]
        native, off = [next(c for c in cases if c["mode"] == m) for m in ("native", "offload")]
        assert native["provenance"] == off["provenance"]
        comparison = next(v for v in s["comparisons"] if v["fixture"] == f["label"])
        ratio = off["peak_allocated_bytes"] / native["peak_allocated_bytes"]
        times = {
            k: off["timing"][k]["median"] / native["timing"][k]["median"]
            for k in ("wall_ms", "event_sum_ms")
        }
        assert comparison["memory_ratio"] == comparison["normal_memory_ratio"] == ratio
        assert comparison["time_ratios"] == times
        gates = dict(
            numerical=all(
                v["passed"]
                for v in a["comparisons"]
                if v["label"] in (native["label"], off["label"])
            ),
            trace=True,
            memory=ratio <= p["memory_ratio_max"],
            event_time=times["event_sum_ms"] <= p["time_ratio_max"],
            wall_time=times["wall_ms"] <= p["time_ratio_max"],
            timing_stability=all(
                v <= p["timing_stability_max"]
                for c in cases
                for v in c["timing_stability"].values()
            ),
            host_memory=off["host_allocated_peak"] <= p["host_allocated_peak_max"],
        )
        assert comparison["gates"] == gates and comparison["passed"] == all(gates.values())
        if f["loss_policy"].endswith("native"):
            assert (
                comparison["passed"]
                and native["peak_allocated_bytes"] - off["peak_allocated_bytes"] == 48 * 2**20
            )
        else:
            assert not comparison["passed"] and not gates["memory"]
    assert not s["component_gate_qualified"] and s["all_numerical_audits_passed"]
    assert s["decision"] == "ELIMINATED FOR THIS JOINT MEMORY/RUNTIME GATE"
    assert not any(v["broad_goal_achieved"] for v in (p, r, a, s))
    assert not s["maintained_defaults_changed"] and not s["original_quality_gate_requalified"]
    for mode in (
        "prepare",
        "qualify",
        "profile",
        "prepare_audit",
        "prepare_recovery",
        "audit_recovery",
        "analyze",
        "report",
        "plot",
        "publish",
    ):
        assert (ROOT / (mode + "_exit.txt")).read_text().strip() == "0"
    assert (ROOT / "audit_exit.txt").read_text().strip() == recovery["original_audit_exit"] != "0"
    assert "access violation" in (ROOT / "audit.log").read_text()
    assert "independent replay complete" not in (ROOT / "audit.log").read_text()
    assert (ROOT / "audit_recovery.log").read_text().count("independent replay complete") == 8
    packed = {}
    for path in ROOT.glob("*.log"):
        output = path.with_suffix(".log.gz")
        if not output.exists():
            output.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    for path in ROOT.glob("*.gz"):
        raw = gzip.decompress(path.read_bytes())
        if path.with_suffix("").exists():
            assert path.with_suffix("").read_bytes() == raw
        packed[path.as_posix()] = dict(sha256=sha(path), unpacked_bytes=len(raw))
    for name in ("result.json", "audit.json", "qualification.json", "qualification_progress.json"):
        local[(ROOT / name).as_posix()] = sha(ROOT / name)
    files = [
        *ROOT.glob("source/*"),
        *ROOT.glob("*_exit.txt"),
        Path(__file__),
        *[
            ROOT / n
            for n in (
                "protocol.json",
                "audit_protocol.json",
                "recovery_protocol.json",
                "summary.json",
                "environment.json",
            )
        ],
        *[Path(n) for n in p["before_documents"]],
        *[
            Path("research") / n
            for n in (
                "checkpoint_input_offload_plan.md",
                "checkpoint_input_offload_audit_recovery.md",
                "checkpoint_input_offload_results.md",
                "figures/checkpoint_input_offload.png",
            )
        ],
    ]
    receipt = dict(
        status="PASS",
        meaning="Evidence integrity passes; native loss qualifies locally, chunked loss fails fixed memory gate",
        all_numerical_audits_passed=True,
        component_gate_qualified=False,
        training_updates=0,
        backwards=262,
        diagnostic_targets=1048576,
        validation_scores=0,
        tensor_artifacts=24,
        phase_records=824,
        zero_gpu_boundaries=21,
        source_hashes_verified=162,
        maintained_hashes_verified=61,
        failed_pre_execution_imports=1,
        repeated_completed_scientific_cases=0,
        broad_goal_achieved=False,
        files={
            f.relative_to(Path.cwd()).as_posix() if f.is_absolute() else f.as_posix(): sha(f)
            for f in files
            if f.is_file()
        },
        tensor_hashes=tensors,
        local_metric_hashes=local,
        packed=packed,
    )
    Path("results/verification/checkpoint_input_offload_final_v1.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in receipt.items()
                if k not in ("files", "tensor_hashes", "local_metric_hashes", "packed")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
