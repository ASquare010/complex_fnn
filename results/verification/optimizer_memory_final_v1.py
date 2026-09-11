"""Validate H120 provenance, resource accounting and failed fixed memory gates."""

import csv
import gzip
import io
import json
import statistics as st
from pathlib import Path

from results.optimizer_memory_v1.source.prepare import ROOT, hashes, read, sha


def table(name):
    return list(csv.DictReader(io.StringIO(gzip.decompress((ROOT / name).read_bytes()).decode())))


def run():
    p, r, a, s = [
        read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json", "summary.json")
    ]
    assert len(p["sources"]) == 152 and len(p["maintained_files"]) == 61
    for field in ("sources", "maintained_files", "input_hashes", "library_hashes"):
        hashes(p[field])
    hashes(read(ROOT / "audit_protocol.json")["files"])
    prior = read(p["previous_receipt"])
    assert prior["status"] == "PASS" and not prior["scientific_numerical_audit_passed"]
    for path, digest in prior["files"].items():
        target = p["before_documents"].get(path, {}).get("preserved_path", path)
        assert sha(target) == digest
    for field in ("tensor_hashes", "local_metric_hashes"):
        hashes(prior[field])
    for path, info in prior["packed"].items():
        assert sha(path) == info["sha256"]
    for info in p["before_documents"].values():
        assert sha(info["preserved_path"]) == info["sha256"]
    assert len(r["cases"]) == 12 and len({c["label"] for c in r["cases"]}) == 12
    assert a["passed"] and s["all_audits_passed"]
    assert a["optimizer_updates"] == a["backwards"] == 0
    assert r["training_updates"] == r["training_backwards"] == s["training_updates"] == 360
    assert r["training_targets"] == s["training_targets"] == 1474560
    assert r["full_validation_scores"] == 24 and a["native_scores"] == 12
    assert a["batches_verified"] == 360 and a["tensor_artifacts_verified"] == 36
    assert len(r["boundaries"]) == len(a["boundaries"]) == 13
    assert all(b == dict(allocated=0, reserved=0) for b in r["boundaries"] + a["boundaries"])
    metrics, updates, phases = [
        table(n) for n in ("metrics.csv.gz", "updates.csv.gz", "phases.csv.gz")
    ]
    assert (len(metrics), len(updates), len(phases)) == (12, 360, 1884)
    tensors, local_metrics, passes = {}, {}, {}
    for c in r["cases"]:
        assert c["training_updates"] == c["training_backwards"] == 30
        assert c["training_targets"] == 122880 and c["full_validation_scores"] == 2
        assert c["parameters"] == 9099648 and c["finite"]
        for key in ("first_gradients", "first_state", "final_state"):
            tensors[c[key]["path"]] = c[key]["sha256"]
        path = ROOT / "runs" / c["label"] / "result.json"
        assert read(path) == c
        local_metrics[path.as_posix()] = sha(path)
        assert [u["step"] for u in c["updates"]] == list(range(1, 31))
        assert [u["absolute_step"] for u in c["updates"]] == list(range(801, 831))
        warm = c["updates"][10:]
        for k in ("wall_ms", "event_sum_ms"):
            values = [u[k] for u in warm]
            assert c["timing"][k] == dict(
                n=20,
                mean=st.mean(values),
                median=st.median(values),
                sample_variance=st.variance(values),
                min=min(values),
                max=max(values),
            )
            halves = [st.median(values[:10]), st.median(values[10:])]
            assert c["timing_stability"][k] == max(halves) / min(halves)
        for u, tab in zip(
            c["updates"], [u for u in updates if u["label"] == c["label"]], strict=True
        ):
            assert sum(u["event_ms"].values()) == u["event_sum_ms"]
            for k, v in u.items():
                if isinstance(v, dict):
                    for phase, ms in v.items():
                        assert float(tab[phase + "_ms"]) == ms
                elif isinstance(v, (float, int)):
                    assert float(tab[k]) == v
                else:
                    assert tab[k] == v
        assert len(c["phases"]) == 157
        assert c["peak_allocated_bytes"] == max(m["peak_allocated_bytes"] for m in c["phases"])
        assert c["peak_reserved_bytes"] == max(m["peak_reserved_bytes"] for m in c["phases"])
        assert all(
            m["phase"].startswith("backward_")
            for m in c["phases"]
            if m["peak_allocated_bytes"] == c["peak_allocated_bytes"]
        )
        for m, tab in zip(
            c["phases"], [m for m in phases if m["label"] == c["label"]], strict=True
        ):
            assert sum(m["storage_bytes"].values()) == m["known_storage_bytes"]
            assert (
                m["known_storage_bytes"] + m["unattributed_live_bytes"] == m["live_allocated_bytes"]
            )
            assert m["unattributed_live_bytes"] >= 0
            assert m["storage_bytes"]["parameters"] == 36398592
            assert m["storage_bytes"]["moments"] == 72797184
            assert m["storage_bytes"]["counters"] == (200 if c["mode"] == "fused" else 0)
            assert m["peak_allocated_bytes"] >= m["live_allocated_bytes"]
            assert m["peak_reserved_bytes"] >= m["live_reserved_bytes"] >= m["live_allocated_bytes"]
            for k, v in m.items():
                if isinstance(v, dict):
                    assert all(int(tab[n]) == value for n, value in v.items())
                elif isinstance(v, int):
                    assert int(tab[k]) == v
                else:
                    assert tab[k] == v
        n = next(n for n in a["numerical"] if n["label"] == c["label"])
        g = next(g for g in a["gradient_comparisons"] if g["label"] == c["label"])
        score = next(v for v in a["scores"] if v["label"] == c["label"])
        assert n["passed"] == (
            n["parameter_error"]["distance"] <= p["parameter_relative_tolerance"]
            and n["parameter_error"]["max_absolute"] <= p["parameter_absolute_tolerance"]
            and all(e["distance"] <= p["moment_tolerance"] for e in n["moment_errors"])
            and n["clip_error"]["distance"] <= p["clip_tolerance"]
        )
        assert n["raw_norm_fp64"] < 1 and n["clip_error"]["distance"] == 0
        assert g["passed"] == (
            g["global_relative"] <= p["gradient_global_tolerance"]
            and g["max_tensor_relative"] <= p["gradient_tensor_tolerance"]
        )
        assert score["batches_verified"] == 30
        assert score["passed"] == (score["relative_error"] <= p["score_tolerance"])
        assert score["native_score"]["targets"] == c["after_score"]["targets"]
        assert score["native_score"]["order_sha256"] == c["after_score"]["order_sha256"]
        passes[c["label"]] = n["passed"] and g["passed"] and score["passed"]
        metric = next(m for m in metrics if m["label"] == c["label"])
        assert float(metric["final_nll"]) == c["after_score"]["nll"]
        assert float(metric["peak_mib"]) == c["peak_allocated_bytes"] / 2**20
        assert metric["numerical_passed"] == "True"
    assert len(tensors) == 36 and all(passes.values())
    hashes(tensors)
    for fixture in p["fixtures"]:
        cases = [c for c in r["cases"] if c["fixture"] == fixture]
        assert len(cases) == 3 and [c["mode"] for c in cases] == fixture["mode_order"]
        for key in ("initial_model_hash", "initial_moment_hash", "initial_sampler_hash"):
            assert len({c[key] for c in cases}) == 1
        base = next(c for c in cases if c["mode"] == "default")
        for c in cases:
            assert [(u["tokens_hash"], u["targets_hash"]) for u in c["updates"]] == [
                (u["tokens_hash"], u["targets_hash"]) for u in base["updates"]
            ]
            if c["mode"] == "default":
                continue
            row = next(
                x
                for x in s["comparisons"]
                if x["fixture"] == fixture["label"] and x["mode"] == c["mode"]
            )
            assert c["peak_allocated_bytes"] - base["peak_allocated_bytes"] == (
                25600 if c["mode"] == "fused" else 0
            )
            assert row["memory_ratio"] == c["peak_allocated_bytes"] / base["peak_allocated_bytes"]
            assert row["nll_ratio"] == c["after_score"]["nll"] / base["after_score"]["nll"]
            assert all(
                row["time_ratios"][k] == c["timing"][k]["median"] / base["timing"][k]["median"]
                for k in row["time_ratios"]
            )
            gates = dict(
                numerical=passes[c["label"]] and passes[base["label"]],
                memory=row["memory_ratio"] <= p["memory_ratio_max"],
                quality=row["nll_ratio"] <= p["nll_ratio_max"],
                wall_time=row["time_ratios"]["wall_ms"] <= p["time_ratio_max"],
                event_time=row["time_ratios"]["event_sum_ms"] <= p["time_ratio_max"],
                timing_stability=all(
                    v <= p["timing_stability_max"]
                    for peer in (base, c)
                    for v in peer["timing_stability"].values()
                ),
            )
            assert row["gates"] == gates and row["passed"] == all(gates.values()) is False
            assert not gates["memory"]
    assert set(s["decisions"].values()) == {"ELIMINATED FOR THIS MEMORY GATE"}
    assert not any(x["broad_goal_achieved"] for x in (p, r, a, s))
    assert not s["maintained_defaults_changed"] and not s["original_quality_gate_requalified"]
    for mode in (
        "prepare",
        "profile",
        "prepare_audit",
        "audit",
        "analyze",
        "plot",
        "report",
        "publish",
    ):
        assert (ROOT / (mode + "_exit.txt")).read_text().strip() == "0"
    packed = {}
    for path in ROOT.glob("*.log"):
        output = path.with_suffix(".log.gz")
        if not output.exists():
            output.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    for path in ROOT.glob("*.gz"):
        raw = gzip.decompress(path.read_bytes())
        original = path.with_suffix("")
        if original.exists():
            assert original.read_bytes() == raw
        packed[path.as_posix()] = dict(sha256=sha(path), unpacked_bytes=len(raw))
    for name in ("result.json", "audit.json"):
        path = ROOT / name
        local_metrics[path.as_posix()] = sha(path)
    files = [
        *ROOT.glob("source/*"),
        *ROOT.glob("*_exit.txt"),
        Path(__file__),
        *[
            ROOT / n
            for n in (
                "protocol.json",
                "audit_protocol.json",
                "summary.json",
                "environment.json",
                "observation_failure.txt",
            )
        ],
        *[Path(n) for n in p["before_documents"]],
        *[
            Path("research") / n
            for n in (
                "optimizer_memory_plan.md",
                "optimizer_memory_results.md",
                "figures/optimizer_memory.png",
            )
        ],
    ]
    receipt = dict(
        status="PASS",
        meaning="Evidence integrity and fixed decisions verified; both memory candidates fail",
        all_numerical_audits_passed=True,
        component_memory_gate_qualified=False,
        training_updates=360,
        training_targets=1474560,
        backwards=360,
        native_audit_scores=12,
        tensor_artifacts=36,
        source_hashes_verified=152,
        maintained_hashes_verified=61,
        scientific_retries=0,
        observation_failure_preserved=True,
        broad_goal_achieved=False,
        files={
            f.relative_to(Path.cwd()).as_posix() if f.is_absolute() else f.as_posix(): sha(f)
            for f in files
            if f.is_file()
        },
        tensor_hashes=tensors,
        local_metric_hashes=local_metrics,
        packed=packed,
    )
    Path("results/verification/optimizer_memory_final_v1.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            dict(
                status="PASS",
                numerical_audit="PASS",
                memory_candidates="ELIMINATED",
                updates=360,
                phase_records=1884,
                tensors=36,
                broad_goal_achieved=False,
            )
        )
    )


if __name__ == "__main__":
    run()
